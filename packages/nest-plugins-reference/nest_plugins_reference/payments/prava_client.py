# SPDX-License-Identifier: Apache-2.0
"""Async transport + client for the Prava agentic-payments REST API.

A pure API client: it owns HTTP, error envelopes, retry policy and the
integer-minor-unit <-> decimal-string conversion Prava's wire format requires.
It holds no Nanda Town types and no simulation state, so it is reusable
outside this repo.

Two transports ship here, and the difference matters:

* :class:`HttpxPravaTransport` -- the real network path, against
  ``https://sandbox.api.prava.space`` or ``https://api.prava.space``.
* :class:`SimulatedPravaTransport` -- a deterministic in-process **test
  double**. It is NOT Prava. It moves no money, mints no credentials and
  contacts no network. It exists because Nanda Town grades determinism
  (same seed -> byte-identical trace) and because CI holds no Prava key.
  Never present its output as a settled transaction.

Endpoints covered (transcribed from the official OpenAPI 3.1 document at
https://docs.prava.space/api-reference/openapi.json):

* ``POST /v1/mandates/{id}/charge``
* ``POST /v1/mandates/{id}/charges/{txn_id}/report``
* ``GET  /v1/mandates/{id}``

The spec exposes **no** refund, credit or reversal endpoint. Callers that
need a reversal must model it as a compensating forward payment; this client
deliberately offers no ``refund`` method rather than faking one.

Example::

    transport = SimulatedPravaTransport(mandate_id="mdt_demo", cap_minor=10_000)
    client = PravaClient(transport=transport)
    result = await client.charge("mdt_demo", amount_minor=4_000, reference="r-1")
"""

from __future__ import annotations

import asyncio
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Protocol, cast, runtime_checkable

import httpx

SANDBOX_BASE_URL = "https://sandbox.api.prava.space"
"""Prava sandbox server root.

Example::

    base = SANDBOX_BASE_URL
"""

PRODUCTION_BASE_URL = "https://api.prava.space"
"""Prava production server root.

Example::

    base = PRODUCTION_BASE_URL
"""

THRESHOLD_EXCEEDED = "THRESHOLD_EXCEEDED"
"""Prava ``errorCode`` returned when a charge would exceed the mandate cap.

Example::

    if result.error_code == THRESHOLD_EXCEEDED:
        ...
"""

CHARGE_STATUS_AWAITING = "awaiting_result"
"""``MandateChargeResult.status`` for an accepted, unreported charge.

Example::

    accepted = result.status == CHARGE_STATUS_AWAITING
"""

CHARGE_STATUS_FAILED = "failed"
"""``MandateChargeResult.status`` for a refused charge.

Example::

    refused = result.status == CHARGE_STATUS_FAILED
"""

_DEFAULT_EXPONENT = 2
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _as_dict(value: object) -> dict[str, Any] | None:
    """Narrow a decoded-JSON value to a string-keyed mapping, or ``None``.

    ``isinstance(x, dict)`` alone narrows to ``dict[Unknown, Unknown]`` under
    pyright strict, which propagates as partially-unknown types through every
    subsequent ``.get()``. Every JSON object Prava returns is string-keyed by
    construction, so the cast is sound and this is the single place it happens.
    """
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return None


class PravaError(RuntimeError):
    """Raised on a non-2xx Prava response, carrying its structured envelope.

    Prava returns ``{"error": {"code": ..., "message": ..., "details": ...}}``.
    ``status`` is the HTTP status, ``code`` the Prava error code (``None`` if
    the body was unparseable).

    Example::

        try:
            await client.get_mandate("mdt_missing")
        except PravaError as exc:
            print(exc.status, exc.code)
    """

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        outer = _as_dict(body)
        envelope = _as_dict(outer.get("error")) if outer is not None else None
        code = envelope.get("code") if envelope is not None else None
        message = envelope.get("message") if envelope is not None else None
        self.code: str | None = str(code) if code is not None else None
        self.message: str = str(message) if message is not None else str(body)[:300]
        super().__init__(f"[{status}] {self.code}: {self.message}")


class PravaConfigError(ValueError):
    """Raised when client or plugin configuration is unusable.

    Example::

        raise PravaConfigError("mandate_id is required")
    """


def minor_to_decimal_string(amount_minor: int, exponent: int = _DEFAULT_EXPONENT) -> str:
    """Convert integer minor units to the decimal string Prava expects.

    Nanda Town's ``Money.amount`` is an ``int``. Prava's ``amount`` field is a
    decimal string such as ``"40.00"``. ``exponent`` is the currency's minor-unit
    exponent (2 for USD/EUR/INR, 0 for JPY/KRW); it is a flag rather than a
    constant precisely so zero-decimal currencies are not silently wrong.

    Example::

        assert minor_to_decimal_string(4_000) == "40.00"
        assert minor_to_decimal_string(4_000, exponent=0) == "4000"

    Raises:
        PravaConfigError: If ``exponent`` is negative.
    """
    if exponent < 0:
        msg = f"exponent must be >= 0: {exponent}"
        raise PravaConfigError(msg)
    if exponent == 0:
        return str(amount_minor)
    scaled = Decimal(amount_minor).scaleb(-exponent)
    quantum = Decimal(1).scaleb(-exponent)
    return str(scaled.quantize(quantum, rounding=ROUND_HALF_UP))


def decimal_string_to_minor(amount: str, exponent: int = _DEFAULT_EXPONENT) -> int:
    """Convert a Prava decimal string back to integer minor units.

    Inverse of :func:`minor_to_decimal_string` for any value representable at
    ``exponent`` decimal places. Round-trip is asserted by the plugin test suite.

    Example::

        assert decimal_string_to_minor("40.00") == 4_000

    Raises:
        PravaConfigError: If ``amount`` is not a valid decimal or ``exponent``
            is negative.
    """
    if exponent < 0:
        msg = f"exponent must be >= 0: {exponent}"
        raise PravaConfigError(msg)
    try:
        value = Decimal(amount)
    except InvalidOperation as exc:
        msg = f"not a decimal amount: {amount!r}"
        raise PravaConfigError(msg) from exc
    scaled = value.scaleb(exponent)
    return int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_UP))


@runtime_checkable
class PravaTransport(Protocol):
    """Structural interface for anything that can carry a Prava request.

    Implemented by :class:`HttpxPravaTransport` (real network) and
    :class:`SimulatedPravaTransport` (deterministic test double). Injecting
    the transport is what lets the scenario run credential-free in CI while
    the identical plugin code path runs live against the sandbox.

    Example::

        async def run(transport: PravaTransport) -> dict[str, object]:
            return await transport.request("GET", "/v1/mandates/mdt_1")
    """

    async def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Issue one request and return the decoded JSON object.

        Example::

            data = await transport.request("GET", "/v1/mandates/mdt_1")
        """
        ...

    async def aclose(self) -> None:
        """Release any underlying resources.

        Example::

            await transport.aclose()
        """
        ...


class HttpxPravaTransport:
    """Real HTTP transport for Prava, backed by ``httpx.AsyncClient``.

    Retries only idempotent-safe failures (``429`` and ``5xx``) with capped
    exponential backoff. A ``4xx`` other than ``429`` is never retried: Prava
    charge semantics are idempotent on ``reference``, but a client-side
    rejection will not become valid by repetition.

    Refuses a ``sk_live_`` key pointed at the sandbox host and vice versa,
    because that mistake is silent and expensive.

    Example::

        transport = HttpxPravaTransport(
            api_key="sk_test_abc",
            base_url=SANDBOX_BASE_URL,
            timeout_s=30.0,
            max_retries=3,
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = SANDBOX_BASE_URL,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.25,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            msg = "api_key is required for the live Prava transport"
            raise PravaConfigError(msg)
        is_live_key = api_key.startswith("sk_live_")
        is_sandbox_host = "sandbox" in base_url
        if is_live_key and is_sandbox_host:
            msg = "live key (sk_live_) against the sandbox host - refusing"
            raise PravaConfigError(msg)
        if not is_live_key and not is_sandbox_host:
            msg = "test key (sk_test_) against the production host - refusing"
            raise PravaConfigError(msg)
        if max_retries < 0:
            msg = f"max_retries must be >= 0: {max_retries}"
            raise PravaConfigError(msg)

        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Issue one request, retrying transient failures.

        Example::

            data = await transport.request("GET", "/v1/mandates/mdt_1")

        Raises:
            PravaError: On a non-2xx response that is not retryable, or after
                the retry budget is exhausted.
        """
        url = f"{self._base_url}{path}"
        last: PravaError | None = None
        for attempt in range(self._max_retries + 1):
            response = await self._client.request(method, url, json=body, params=params)
            decoded = self._decode(response)
            if response.status_code < 400:
                return decoded
            error = PravaError(response.status_code, decoded)
            if response.status_code not in _RETRYABLE_STATUS:
                raise error
            last = error
            if attempt < self._max_retries:
                await asyncio.sleep(self._backoff_base_s * (2**attempt))
        raise last if last is not None else PravaError(0, {})

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` if this transport owns it.

        Example::

            await transport.aclose()
        """
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            decoded: Any = response.json()
        except ValueError:
            return {"raw": response.text[:500]}
        as_object = _as_dict(decoded)
        if as_object is not None:
            return as_object
        return {"raw": decoded}


class SimulatedPravaTransport:
    """Deterministic in-process test double for the Prava API. NOT Prava.

    **This moves no money.** It reproduces four documented behaviours the
    plugin must handle, so that CI (which has no ``PRAVA_API_KEY``) still
    exercises the real plugin code path:

    #. ``reference`` is an idempotency key -- replaying one returns the
       original charge with ``deduplicated: true``.
    #. Exceeding ``cap_minor`` returns ``status: "failed"`` with
       ``errorCode: "THRESHOLD_EXCEEDED"``.
    #. ``GET /v1/mandates/{id}`` reports ``spent``, ``chargeCount`` and the
       ``charges[]`` array that ``verify_payment`` reads back.
    #. An unknown mandate id raises a 404 :class:`PravaError`.

    Identifiers are derived from a monotonic counter, never from ``uuid4`` or
    the clock, so a replayed simulation yields a byte-identical trace.

    Example::

        transport = SimulatedPravaTransport(mandate_id="mdt_demo", cap_minor=10_000)
        result = await transport.request(
            "POST",
            "/v1/mandates/mdt_demo/charge",
            body={"amount": "40.00", "reference": "r-1"},
        )
        assert result["status"] == CHARGE_STATUS_AWAITING
    """

    def __init__(
        self,
        mandate_id: str,
        cap_minor: int,
        currency: str = "USD",
        exponent: int = _DEFAULT_EXPONENT,
        fail_references: frozenset[str] = frozenset(),
    ) -> None:
        self._mandate_id = mandate_id
        self._cap_minor = cap_minor
        self._currency = currency
        self._exponent = exponent
        self._fail_references = fail_references
        self._spent_minor = 0
        self._seq = 0
        self._by_reference: dict[str, dict[str, Any]] = {}

    async def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Route a simulated request. See the class docstring for coverage.

        Example::

            detail = await transport.request("GET", "/v1/mandates/mdt_demo")

        Raises:
            PravaError: 404 for an unknown mandate or transaction id.
            NotImplementedError: For a path this double does not simulate.
        """
        del params
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "v1" and parts[1] == "mandates":
            mandate_id = parts[2]
            if mandate_id != self._mandate_id:
                raise PravaError(404, {"error": {"code": "MANDATE_NOT_FOUND"}})
            if method == "GET" and len(parts) == 3:
                return self._mandate_detail()
            if method == "POST" and len(parts) == 4 and parts[3] == "charge":
                return self._charge(body or {})
            if method == "POST" and len(parts) == 6 and parts[5] == "report":
                return self._report(parts[4], body or {})
        msg = f"SimulatedPravaTransport does not simulate {method} {path}"
        raise NotImplementedError(msg)

    async def aclose(self) -> None:
        """No-op; the double owns no resources.

        Example::

            await transport.aclose()
        """
        return None

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_sim{self._seq:06d}"

    def _charge(self, body: dict[str, Any]) -> dict[str, Any]:
        reference = str(body.get("reference", ""))
        amount = str(body.get("amount", "0"))
        existing = self._by_reference.get(reference)
        if existing is not None:
            return {**existing, "deduplicated": True}

        if reference in self._fail_references:
            return {
                "mandateId": self._mandate_id,
                "status": CHARGE_STATUS_FAILED,
                "errorCode": "CHARGE_DECLINED",
                "errorMessage": "simulated decline for reference in fail_references",
                "deduplicated": False,
            }

        amount_minor = decimal_string_to_minor(amount, self._exponent)
        if self._spent_minor + amount_minor > self._cap_minor:
            return {
                "mandateId": self._mandate_id,
                "status": CHARGE_STATUS_FAILED,
                "errorCode": THRESHOLD_EXCEEDED,
                "errorMessage": "charge would exceed the mandate threshold",
                "deduplicated": False,
            }

        self._spent_minor += amount_minor
        result: dict[str, Any] = {
            "mandateId": self._mandate_id,
            "instructionId": self._next_id("ins"),
            "transactionId": self._next_id("txn"),
            "orderId": self._next_id("ord"),
            "status": CHARGE_STATUS_AWAITING,
            "fetchStatus": "ready",
            "deduplicated": False,
            "_amount": amount,
            "_reference": reference,
        }
        self._by_reference[reference] = result
        return result

    def _report(self, txn_id: str, body: dict[str, Any]) -> dict[str, Any]:
        record = next(
            (r for r in self._by_reference.values() if r.get("transactionId") == txn_id),
            None,
        )
        if record is None:
            raise PravaError(404, {"error": {"code": "TRANSACTION_NOT_FOUND"}})
        approved = str(body.get("txn_status")) == "APPROVED"
        record["_reported"] = "APPROVED" if approved else "DECLINED"
        return {
            "status": "completed" if approved else "failed",
            "mandateStatus": "active",
            "visaConfirmation": self._next_id("vis"),
        }

    def _mandate_detail(self) -> dict[str, Any]:
        charges: list[dict[str, Any]] = [
            {
                "transactionId": r["transactionId"],
                "amount": r["_amount"],
                "currency": self._currency,
                "status": r.get("_reported", "awaiting_result"),
                "reference": r["_reference"],
                "createdAt": f"sim-tick-{i}",
            }
            for i, r in enumerate(self._by_reference.values())
        ]
        return {
            "id": self._mandate_id,
            "status": "active",
            "currency": self._currency,
            "spent": minor_to_decimal_string(self._spent_minor, self._exponent),
            "remaining": minor_to_decimal_string(
                self._cap_minor - self._spent_minor, self._exponent
            ),
            "chargeCount": len(charges),
            "charges": charges,
        }


class ChargeResult:
    """Normalized outcome of ``POST /v1/mandates/{id}/charge``.

    ``accepted`` collapses Prava's ``status`` plus ``errorCode`` into the one
    question the payments layer actually asks. ``over_cap`` distinguishes the
    mandate-threshold refusal, which is terminal for the run, from a decline,
    which is not.

    Example::

        result = await client.charge("mdt_1", amount_minor=4_000, reference="r-1")
        if result.over_cap:
            ...
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.status: str = str(payload.get("status", ""))
        self.transaction_id: str | None = _opt_str(payload.get("transactionId"))
        self.order_id: str | None = _opt_str(payload.get("orderId"))
        self.instruction_id: str | None = _opt_str(payload.get("instructionId"))
        self.error_code: str | None = _opt_str(payload.get("errorCode"))
        self.error_message: str | None = _opt_str(payload.get("errorMessage"))
        self.deduplicated: bool = bool(payload.get("deduplicated", False))

    @property
    def accepted(self) -> bool:
        """True when Prava took the charge (including an idempotent replay).

        Example::

            assert result.accepted
        """
        return self.status != CHARGE_STATUS_FAILED and self.error_code is None

    @property
    def over_cap(self) -> bool:
        """True when the refusal was the mandate threshold, not a decline.

        Example::

            if result.over_cap:
                stop()
        """
        return self.error_code == THRESHOLD_EXCEEDED


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


class PravaClient:
    """Thin async wrapper over the three Prava endpoints the plugin needs.

    Owns the minor-unit conversion so callers never build decimal strings by
    hand. Deliberately exposes **no** ``refund``: the Prava OpenAPI document
    defines no refund, credit or reversal path.

    Example::

        client = PravaClient(
            transport=SimulatedPravaTransport("mdt_demo", cap_minor=10_000),
            minor_unit_exponent=2,
        )
        result = await client.charge("mdt_demo", amount_minor=4_000, reference="r-1")
    """

    def __init__(
        self,
        transport: PravaTransport,
        minor_unit_exponent: int = _DEFAULT_EXPONENT,
    ) -> None:
        self._transport = transport
        self._exponent = minor_unit_exponent

    @property
    def minor_unit_exponent(self) -> int:
        """Minor-unit exponent this client formats amounts with.

        Example::

            assert client.minor_unit_exponent == 2
        """
        return self._exponent

    async def charge(
        self,
        mandate_id: str,
        amount_minor: int,
        reference: str,
        purchase_context: list[dict[str, Any]] | None = None,
    ) -> ChargeResult:
        """Draw ``amount_minor`` against ``mandate_id``, keyed on ``reference``.

        ``reference`` is Prava's idempotency key (max 255 chars): the same
        mandate id plus reference returns the original charge with
        ``deduplicated: true`` instead of double-charging.

        Example::

            result = await client.charge("mdt_1", amount_minor=4_000, reference="r-1")

        Raises:
            PravaConfigError: If ``amount_minor`` is not positive or
                ``reference`` is empty or longer than 255 characters.
            PravaError: On a non-2xx response.
        """
        if amount_minor <= 0:
            msg = f"amount_minor must be positive: {amount_minor}"
            raise PravaConfigError(msg)
        if not reference or len(reference) > 255:
            msg = f"reference must be 1..255 chars, got {len(reference)}"
            raise PravaConfigError(msg)
        body: dict[str, Any] = {
            "amount": minor_to_decimal_string(amount_minor, self._exponent),
            "reference": reference,
        }
        if purchase_context:
            body["purchase_context"] = purchase_context
        payload = await self._transport.request(
            "POST", f"/v1/mandates/{mandate_id}/charge", body=body
        )
        return ChargeResult(payload)

    async def report_charge(
        self,
        mandate_id: str,
        transaction_id: str,
        approved: bool,
        amount_minor: int | None = None,
        authorization_code: str | None = None,
        response_code: str | None = None,
    ) -> dict[str, Any]:
        """Report the merchant-side outcome so the network settles the charge.

        Example::

            await client.report_charge("mdt_1", "txn_1", approved=True)

        Raises:
            PravaError: On a non-2xx response.
        """
        body: dict[str, Any] = {
            "txn_status": "APPROVED" if approved else "DECLINED",
            "txn_type": "PURCHASE",
        }
        if amount_minor is not None:
            body["amount_paid"] = minor_to_decimal_string(amount_minor, self._exponent)
        if authorization_code:
            body["authorization_code"] = authorization_code
        if response_code:
            body["response_code"] = response_code
        return await self._transport.request(
            "POST",
            f"/v1/mandates/{mandate_id}/charges/{transaction_id}/report",
            body=body,
        )

    async def get_mandate(self, mandate_id: str) -> dict[str, Any]:
        """Read the mandate back, including its ``charges[]`` array.

        Example::

            detail = await client.get_mandate("mdt_1")

        Raises:
            PravaError: On a non-2xx response.
        """
        return await self._transport.request("GET", f"/v1/mandates/{mandate_id}")

    async def find_charge(self, mandate_id: str, reference: str) -> dict[str, Any] | None:
        """Locate a charge in ``charges[]`` by its idempotency ``reference``.

        This is the read-back that makes ``verify_payment`` authoritative:
        the answer comes from Prava's own ledger, not from local state.

        Example::

            charge = await client.find_charge("mdt_1", "r-1")

        Raises:
            PravaError: On a non-2xx response.
        """
        detail = await self.get_mandate(mandate_id)
        raw = detail.get("charges")
        if not isinstance(raw, list):
            return None
        for item in cast("list[object]", raw):
            entry = _as_dict(item)
            if entry is not None and str(entry.get("reference")) == reference:
                return entry
        return None

    async def aclose(self) -> None:
        """Close the underlying transport.

        Example::

            await client.aclose()
        """
        await self._transport.aclose()
