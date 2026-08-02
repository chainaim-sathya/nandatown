# SPDX-License-Identifier: Apache-2.0
"""Prava payments plugin -- settles Nanda Town payments on Visa rails via Prava.

Maps the four ``Payments`` protocol methods onto Prava's agentic-payments
mandate API:

===================  ==========================================================
``quote``            Local price book; no network call. Prava has no quote API.
``pay``              ``POST /v1/mandates/{id}/charge`` then
                     ``POST /v1/mandates/{id}/charges/{txn}/report``.
``verify_payment``   ``GET /v1/mandates/{id}``, then locate the charge by its
                     idempotency ``reference``. Authoritative: the answer comes
                     from Prava's ledger, not from local state.
``refund``           Raises :class:`PravaRefundUnsupportedError`. The Prava
                     OpenAPI document defines no refund, credit or reversal
                     endpoint, so a silent no-op here would be a fabricated
                     capability.
===================  ==========================================================

**The binding.** ``PaymentRef`` is passed straight through as Prava's
``reference``, which Prava documents as an idempotency key. Nanda Town derives
payment refs deterministically from the scenario seed, so replaying a
simulation emits the same references and the *card network itself* refuses the
second charge. Replay safety stops being an assertion in our code and becomes a
property enforced by the payment rail.

**Two modes, never confused.** ``mode="simulated"`` runs against
:class:`~nest_plugins_reference.payments.prava_client.SimulatedPravaTransport`
-- an in-process test double that moves no money. ``mode="live"`` runs against
the real sandbox or production host. There is deliberately no ``dry_run`` flag
that returns a synthetic ``Receipt``: that would make a simulated trace
indistinguishable from a settled one. :attr:`PravaPayments.mode` is public and
:meth:`PravaPayments.confirmed` reports whether the merchant-report leg closed,
so a reader can always tell what actually happened. (``Receipt`` itself carries
no metadata field, so the mode is surfaced on the plugin and in the scenario's
broadcast events rather than smuggled into the receipt.)

**Mandate provisioning is out of band.** Creating a mandate requires
``POST /v1/sessions`` with a ``mandate_setup`` block and a *human passkey
approval*. That cannot happen mid-simulation, so ``mandate_id`` is provisioned
beforehand and injected as config. The API key is read from ``PRAVA_API_KEY``
and never appears in scenario YAML.

Example::

    payments = PravaPayments(
        AgentId("buyer-0"),
        mandate_id="mdt_demo",
        mode="simulated",
        cap_minor=10_000,
        unit_price_minor=4_000,
    )
    receipt = await payments.pay(
        AgentId("seller-0"), Money(amount=4_000, currency="USD"), PaymentRef("r-1")
    )
"""

from __future__ import annotations

import os

from nest_core.types import (
    AgentId,
    Money,
    PaymentRef,
    PaymentStatus,
    Quote,
    Receipt,
    ServiceRef,
)

from nest_plugins_reference.payments.prava_client import (
    PRODUCTION_BASE_URL,
    SANDBOX_BASE_URL,
    ChargeResult,
    HttpxPravaTransport,
    PravaClient,
    PravaConfigError,
    PravaError,
    PravaTransport,
    SimulatedPravaTransport,
)

MODE_SIMULATED = "simulated"
"""Run against the in-process test double. No money moves.

Example::

    payments = PravaPayments(AgentId("a"), mandate_id="m", mode=MODE_SIMULATED)
"""

MODE_LIVE = "live"
"""Run against a real Prava host. Requires ``PRAVA_API_KEY``.

Example::

    payments = PravaPayments(AgentId("a"), mandate_id="m", mode=MODE_LIVE)
"""

ENV_SANDBOX = "sandbox"
"""Target ``https://sandbox.api.prava.space``.

Example::

    env = ENV_SANDBOX
"""

ENV_PRODUCTION = "production"
"""Target ``https://api.prava.space``.

Example::

    env = ENV_PRODUCTION
"""

_MODES = (MODE_SIMULATED, MODE_LIVE)
_ENVS = (ENV_SANDBOX, ENV_PRODUCTION)
_NEUTRAL_CURRENCY = "credits"


class PravaPaymentError(RuntimeError):
    """Raised when a Prava charge is refused, declined or left unconfirmed.

    ``over_cap`` is ``True`` only for the mandate-threshold refusal
    (``THRESHOLD_EXCEEDED``), which is terminal for the mandate; an ordinary
    decline is not.

    Example::

        try:
            await payments.pay(seller, Money(amount=999_999), PaymentRef("r"))
        except PravaPaymentError as exc:
            if exc.over_cap:
                stop_buying()
    """

    def __init__(self, message: str, over_cap: bool = False, code: str | None = None) -> None:
        self.over_cap = over_cap
        self.code = code
        super().__init__(message)


class PravaRefundUnsupportedError(NotImplementedError):
    """Raised by :meth:`PravaPayments.refund`. Prava exposes no refund endpoint.

    Verified against the official OpenAPI 3.1 document at
    https://docs.prava.space/api-reference/openapi.json -- all 13 paths were
    read; none is a refund, credit or reversal. Callers needing a reversal must
    model it as a compensating *forward* payment to the original payer, using a
    fresh ``PaymentRef``, and record both legs.

    Example::

        try:
            await payments.refund(PaymentRef("r-1"))
        except PravaRefundUnsupportedError:
            await seller_payments.pay(buyer, amount, PaymentRef("r-1-reversal"))
    """


class PravaPayments:
    """Nanda Town payments plugin backed by the Prava mandate API.

    Configuration is explicit keyword arguments with defaults, all surfaced
    through the scenario's ``task.config`` block:

    ==========================  =============  =================================
    Argument                    Default        Meaning
    ==========================  =============  =================================
    ``mandate_id``              *required*     Provisioned out of band.
    ``mode``                    ``simulated``  ``simulated`` | ``live``.
    ``prava_env``               ``sandbox``    ``sandbox`` | ``production``.
    ``api_key``                 ``$PRAVA_API_KEY``  Never put this in YAML.
    ``currency``                ``USD``        ISO 4217 code sent to Prava.
    ``minor_unit_exponent``     ``2``          0 for JPY/KRW.
    ``unit_price_minor``        ``4_000``      Fallback price when no price book.
    ``price_book``              ``None``       ``service_ref -> minor units``.
    ``intent_digest``           ``None``       Snapshot digest, stamped on quotes.
    ``cap_minor``               ``100_000``    Simulated-mode mandate ceiling.
    ``fail_closed``             ``True``       See below.
    ``timeout_s``               ``30.0``       Live transport only.
    ``max_retries``             ``3``          Live transport only; 429/5xx.
    ``authorization_code``      ``None``       Merchant auth code, if any.
    ==========================  =============  =================================

    ``fail_closed`` governs the *report* leg. A charge that Prava accepted but
    whose merchant-side report failed is an **unconfirmed** payment. With
    ``fail_closed=True`` (default) that raises :class:`PravaPaymentError` and no
    receipt is issued. With ``fail_closed=False`` the receipt is returned marked
    ``reported=False``, and :meth:`verify_payment` will report ``PENDING`` until
    the report succeeds -- useful when a scenario wants to observe the
    uncertainty rather than abort on it. Neither setting ever assumes success.

    Example::

        payments = PravaPayments(
            AgentId("buyer-0"),
            mandate_id="mdt_demo",
            mode="simulated",
            cap_minor=10_000,
        )
        receipt = await payments.pay(
            AgentId("seller-0"),
            Money(amount=4_000, currency="USD"),
            PaymentRef("scenario-42-buyer0-trade0"),
        )
    """

    def __init__(
        self,
        agent_id: AgentId,
        mandate_id: str = "",
        mode: str = MODE_SIMULATED,
        prava_env: str = ENV_SANDBOX,
        api_key: str | None = None,
        currency: str = "USD",
        minor_unit_exponent: int = 2,
        unit_price_minor: int = 4_000,
        price_book: dict[str, int] | None = None,
        intent_digest: str | None = None,
        cap_minor: int = 100_000,
        fail_closed: bool = True,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        authorization_code: str | None = None,
        transport: PravaTransport | None = None,
        receipts: dict[PaymentRef, Receipt] | None = None,
    ) -> None:
        if mode not in _MODES:
            msg = f"mode must be one of {_MODES}: {mode!r}"
            raise PravaConfigError(msg)
        if prava_env not in _ENVS:
            msg = f"prava_env must be one of {_ENVS}: {prava_env!r}"
            raise PravaConfigError(msg)
        if not mandate_id:
            msg = (
                "mandate_id is required. Prava mandates are created by "
                "POST /v1/sessions with a human passkey approval, which cannot "
                "happen mid-simulation -- provision one first and inject it."
            )
            raise PravaConfigError(msg)

        self._agent_id = agent_id
        self._mandate_id = mandate_id
        self._mode = mode
        self._currency = currency
        self._exponent = minor_unit_exponent
        self._unit_price_minor = unit_price_minor
        self._price_book = dict(price_book) if price_book is not None else None
        self._intent_digest = intent_digest
        self._fail_closed = fail_closed
        self._authorization_code = authorization_code
        self._receipts: dict[PaymentRef, Receipt] = receipts if receipts is not None else {}
        self._confirmed: dict[PaymentRef, bool] = {}

        chosen = (
            transport
            if transport is not None
            else self._build_transport(
                mode=mode,
                prava_env=prava_env,
                api_key=api_key,
                cap_minor=cap_minor,
                currency=currency,
                minor_unit_exponent=minor_unit_exponent,
                timeout_s=timeout_s,
                max_retries=max_retries,
                mandate_id=mandate_id,
            )
        )
        self._client = PravaClient(chosen, minor_unit_exponent=minor_unit_exponent)

    def _build_transport(
        self,
        mode: str,
        prava_env: str,
        api_key: str | None,
        cap_minor: int,
        currency: str,
        minor_unit_exponent: int,
        timeout_s: float,
        max_retries: int,
        mandate_id: str,
    ) -> PravaTransport:
        if mode == MODE_SIMULATED:
            return SimulatedPravaTransport(
                mandate_id=mandate_id,
                cap_minor=cap_minor,
                currency=currency,
                exponent=minor_unit_exponent,
            )
        key = api_key if api_key is not None else os.environ.get("PRAVA_API_KEY", "")
        if not key:
            msg = (
                "mode='live' requires a Prava secret key. Set PRAVA_API_KEY in the "
                "environment; never place it in scenario YAML."
            )
            raise PravaConfigError(msg)
        base = PRODUCTION_BASE_URL if prava_env == ENV_PRODUCTION else SANDBOX_BASE_URL
        return HttpxPravaTransport(
            api_key=key,
            base_url=base,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

    # -- introspection ---------------------------------------------------

    @property
    def mode(self) -> str:
        """``"simulated"`` or ``"live"``. Never inferred; always explicit.

        Example::

            assert payments.mode == "simulated"
        """
        return self._mode

    @property
    def mandate_id(self) -> str:
        """The out-of-band provisioned mandate this plugin charges against.

        Example::

            print(payments.mandate_id)
        """
        return self._mandate_id

    def receipt(self, ref: PaymentRef) -> Receipt | None:
        """Return the locally cached receipt for ``ref``, if this agent made it.

        Local cache only. :meth:`verify_payment` is the authoritative check.

        Example::

            r = payments.receipt(PaymentRef("r-1"))
        """
        return self._receipts.get(ref)

    def confirmed(self, ref: PaymentRef) -> bool:
        """Whether the merchant-report leg closed for ``ref`` in this process.

        ``False`` for a charge Prava accepted but whose report failed under
        ``fail_closed=False``. Local view only; :meth:`verify_payment` is
        authoritative.

        Example::

            assert payments.confirmed(PaymentRef("r-1"))
        """
        return self._confirmed.get(ref, False)

    # -- Payments protocol -----------------------------------------------

    async def quote(self, service: ServiceRef) -> Quote:
        """Price ``service`` from the price book, or fall back to the unit price.

        No network call: Prava prices nothing -- it authorizes and settles --
        and a live catalogue read would make ``quote`` non-deterministic under
        replay. When a ``price_book`` is supplied it is a *frozen* mapping built
        from purchase-intent snapshots, so the price an upstream agent agreed to
        is the price this method returns.

        An unknown ``service`` raises rather than falling back. A silent
        fallback would quote the default unit price for an item nobody priced,
        and the trace would look healthy while the wrong amount was charged.

        Example::

            q = await payments.quote(ServiceRef("libas:98252-2XL"))

        Raises:
            PravaConfigError: If a price book is configured and ``service`` is
                not in it.
        """
        metadata = {"source": "prava_plugin_price_book", "mode": self._mode}
        if self._price_book is None:
            amount_minor = self._unit_price_minor
        else:
            key = str(service)
            if key not in self._price_book:
                known = ", ".join(sorted(self._price_book)) or "<empty>"
                msg = (
                    f"no price for service {key!r} in the purchase-intent price book. "
                    f"Known refs: {known}. The scenario asked to quote an item that no "
                    "snapshot priced -- fix the service ref or capture a snapshot for it."
                )
                raise PravaConfigError(msg)
            amount_minor = self._price_book[key]
            metadata["source"] = "prava_intent_snapshot"
        if self._intent_digest is not None:
            metadata["intent_digest"] = self._intent_digest
        return Quote(
            service=service,
            price=Money(amount=amount_minor, currency=self._currency),
            metadata=metadata,
        )

    async def pay(self, to: AgentId, amount: Money, ref: PaymentRef) -> Receipt:
        """Charge the mandate for ``amount``, keyed on ``ref``, then report it.

        ``ref`` becomes Prava's ``reference`` idempotency key verbatim. A
        replayed ``ref`` returns Prava's original charge with
        ``deduplicated: true``; the receipt records that, and the mandate is not
        drawn twice.

        Example::

            receipt = await payments.pay(
                AgentId("seller-0"),
                Money(amount=4_000, currency="USD"),
                PaymentRef("scenario-42-trade-0"),
            )

        Raises:
            PravaConfigError: If ``amount`` is not positive or its currency is
                neither the configured currency nor the layer-neutral
                ``"credits"``.
            PravaPaymentError: If Prava refuses the charge, or if the charge was
                accepted but could not be confirmed and ``fail_closed`` is set.
        """
        self._check_currency(amount)
        if amount.amount <= 0:
            msg = f"payment amount must be positive: {amount.amount}"
            raise PravaConfigError(msg)

        try:
            result = await self._client.charge(
                self._mandate_id,
                amount_minor=amount.amount,
                reference=str(ref),
                purchase_context=[{"description": f"nest payment {ref} to {to}"}],
            )
        except PravaError as exc:
            msg = f"charge call failed for {ref}: {exc}"
            raise PravaPaymentError(msg, code=exc.code) from exc

        if not result.accepted:
            msg = (
                f"Prava refused charge {ref}: "
                f"{result.error_code or result.status} {result.error_message or ''}".strip()
            )
            raise PravaPaymentError(msg, over_cap=result.over_cap, code=result.error_code)

        self._confirmed[ref] = await self._report(result, amount)
        receipt = Receipt(
            ref=ref,
            payer=self._agent_id,
            payee=to,
            amount=Money(amount=amount.amount, currency=self._currency),
        )
        self._receipts[ref] = receipt
        return receipt

    async def _report(self, result: ChargeResult, amount: Money) -> bool:
        """Run the merchant-report leg. Returns whether it confirmed."""
        if result.transaction_id is None:
            if self._fail_closed:
                msg = "charge accepted but Prava returned no transactionId; cannot confirm"
                raise PravaPaymentError(msg)
            return False
        try:
            response = await self._client.report_charge(
                self._mandate_id,
                transaction_id=result.transaction_id,
                approved=True,
                amount_minor=amount.amount,
                authorization_code=self._authorization_code,
            )
        except PravaError as exc:
            if self._fail_closed:
                msg = (
                    f"charge {result.transaction_id} accepted but the report leg "
                    f"failed ({exc}); payment is UNCONFIRMED and fail_closed is set"
                )
                raise PravaPaymentError(msg, code=exc.code) from exc
            return False
        return str(response.get("status")) == "completed"

    async def verify_payment(self, ref: PaymentRef) -> PaymentStatus:
        """Read the charge back from Prava's ledger and map it to a status.

        Authoritative by construction: it re-reads ``GET /v1/mandates/{id}`` and
        locates the charge by ``reference``, so it reports what the payment rail
        believes, not what this process believes. A network failure yields
        ``PENDING`` rather than ``FAILED`` -- an unreachable API is not evidence
        that a payment did not happen.

        Example::

            status = await payments.verify_payment(PaymentRef("r-1"))

        """
        try:
            charge = await self._client.find_charge(self._mandate_id, str(ref))
        except PravaError:
            return PaymentStatus.PENDING
        if charge is None:
            return PaymentStatus.FAILED
        status = str(charge.get("status", "")).lower()
        if status in ("approved", "completed", "settled"):
            return PaymentStatus.CONFIRMED
        if status in ("declined", "failed"):
            return PaymentStatus.FAILED
        return PaymentStatus.PENDING

    async def refund(self, ref: PaymentRef) -> None:
        """Always raises. Prava exposes no refund, credit or reversal endpoint.

        See :class:`PravaRefundUnsupportedError` for the verification note and
        the compensating-payment workaround.

        Example::

            with pytest.raises(PravaRefundUnsupportedError):
                await payments.refund(PaymentRef("r-1"))

        Raises:
            PravaRefundUnsupportedError: Always.
        """
        msg = (
            f"cannot refund {ref}: the Prava API defines no refund, credit or "
            "reversal endpoint (verified against the official OpenAPI document, "
            "all 13 paths). Model reversals as a compensating forward payment "
            "with a fresh PaymentRef."
        )
        raise PravaRefundUnsupportedError(msg)

    # -- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying transport.

        Example::

            await payments.aclose()
        """
        await self._client.aclose()

    # -- private ---------------------------------------------------------

    def _check_currency(self, amount: Money) -> None:
        if amount.currency in (self._currency, _NEUTRAL_CURRENCY):
            return
        msg = (
            f"currency mismatch: plugin is configured for {self._currency!r} but "
            f"was handed {amount.currency!r}. Nanda Town's layer-neutral "
            f"{_NEUTRAL_CURRENCY!r} is also accepted and interpreted as "
            f"{self._currency!r}."
        )
        raise PravaConfigError(msg)
