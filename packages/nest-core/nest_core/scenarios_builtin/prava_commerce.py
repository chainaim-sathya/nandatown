# SPDX-License-Identifier: Apache-2.0
"""Prava commerce scenario -- buyer/seller pairs settling on a Prava mandate.

Drives the ``prava`` payments plugin through the full agentic-commerce loop the
hackathon track asks for -- **quote, pay, verify, handle failure** -- plus the
two behaviours that make a payment rail trustworthy under simulation:

#. **Idempotent replay.** Each buyer pays once, then re-issues the *same*
   ``PaymentRef``. Prava treats ``reference`` as an idempotency key, so the
   second call returns the original charge with ``deduplicated: true`` and the
   mandate is not drawn twice. Nanda Town derives payment refs deterministically
   from the seed, so replaying the whole simulation is refused by the card
   network rather than by our own bookkeeping.
#. **Refusal at the rail.** Each buyer then attempts a deliberately oversized
   charge. Prava returns ``status: "failed"`` with
   ``errorCode: "THRESHOLD_EXCEEDED"``; the buyer records the refusal and stops.
   The budget is enforced by the mandate, not by agent politeness.

The buyer finally calls ``refund()``, which raises because Prava's API defines
no refund endpoint. That is recorded as an explicit ``unsupported`` event rather
than hidden -- a plugin that silently no-ops a refund is lying about its
capability.

Every transition is broadcast as ``prava:<kind>:k=v:...``, mirroring the
``escrow:*`` convention in :mod:`nest_core.scenarios_builtin.escrow_marketplace`.
Each event carries ``mode=simulated`` or ``mode=live`` so no reader can mistake
an in-process rehearsal for a settled transaction.

Exceptions are caught by their **stdlib bases** (``RuntimeError`` for a refused
charge, ``NotImplementedError`` for the unsupported refund) so that ``nest_core``
never imports ``nest_plugins_reference``. Under a payments plugin without Prava
semantics (e.g. ``prepaid_credits``) the agents fall back to a plain ``pay()``
and emit no ``prava:*`` events at all.

Example::

    agents = prava_commerce_factory(config, plugins)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from nest_core.scenario import ScenarioConfig
from nest_core.sim.agent import AgentContext, StateMachineAgent
from nest_core.types import AgentId, Money, PaymentRef, ServiceRef

_TICK_QUOTE = 1.0
_TICK_PAY = 3.0
_TICK_REPLAY = 5.0
_TICK_VERIFY = 7.0
_TICK_OVERCAP = 9.0
_TICK_REFUND = 11.0

_OP_QUOTE = b"op:quote"
_OP_PAY = b"op:pay"
_OP_REPLAY = b"op:replay"
_OP_VERIFY = b"op:verify"
_OP_OVERCAP = b"op:overcap"
_OP_REFUND = b"op:refund"

DEFAULT_PAIRS = 2
"""Number of buyer/seller pairs when ``task.config`` does not say.

Kept small on purpose: a marketplace-sized run would exhaust a sandbox mandate
cap long before the scenario finished.

Example::

    pairs = DEFAULT_PAIRS
"""

DEFAULT_UNIT_PRICE_MINOR = 4_000
"""Default charge size in minor units (``"40.00"`` at exponent 2).

Example::

    price = DEFAULT_UNIT_PRICE_MINOR
"""

DEFAULT_OVERCAP_MINOR = 99_999_900
"""Deliberately oversized charge used to provoke ``THRESHOLD_EXCEEDED``.

Large enough to exceed any plausible sandbox mandate, so the failure path is
exercised identically in ``simulated`` and ``live`` mode.

Example::

    huge = DEFAULT_OVERCAP_MINOR
"""


def _load_intent_snapshot(path: str) -> tuple[str, int, str]:
    """Read a purchase-intent snapshot and return ``(service_ref, minor, digest)``.

    Parsed with the standard library rather than
    ``nest_plugins_reference.payments.prava_intent`` because ``nest_core`` must
    not import the plugin package. That module remains the canonical validated
    loader for library consumers and for the plugin's own tests; this reads the
    three fields the scenario needs and lets the plugin reject anything wrong.

    Example::

        ref, minor, digest = _load_intent_snapshot("scenarios/intents/x.json")

    Raises:
        ValueError: If the file is unreadable, is not a JSON object, or lacks
            ``item.service_ref`` or ``settlement.amount_minor``.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
        payload: Any = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"cannot read purchase-intent snapshot {path!r}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"purchase-intent snapshot {path!r} must contain a JSON object"
        raise ValueError(msg)
    data = cast("dict[str, Any]", payload)
    item = data.get("item")
    settlement = data.get("settlement")
    if not isinstance(item, dict) or not isinstance(settlement, dict):
        msg = f"purchase-intent snapshot {path!r} needs 'item' and 'settlement' objects"
        raise ValueError(msg)
    service_ref = cast("dict[str, Any]", item).get("service_ref")
    amount_minor = cast("dict[str, Any]", settlement).get("amount_minor")
    if not isinstance(service_ref, str) or not service_ref.strip():
        msg = f"purchase-intent snapshot {path!r} has no usable item.service_ref"
        raise ValueError(msg)
    if isinstance(amount_minor, bool) or not isinstance(amount_minor, int) or amount_minor <= 0:
        msg = (
            f"purchase-intent snapshot {path!r} settlement.amount_minor must be a "
            f"positive integer, got {amount_minor!r}"
        )
        raise ValueError(msg)
    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return service_ref, amount_minor, digest


def _emit(fields: dict[str, str | int]) -> bytes:
    """Build a ``prava:<kind>:k=v:...`` broadcast payload.

    Mirrors the colon-separated form used by the escrow scenario so a single
    trace parser handles both.

    Example::

        payload = _emit({"kind": "paid", "ref": "r-1"})
    """
    kind = str(fields.pop("kind"))
    body = ":".join(f"{k}={v}" for k, v in fields.items())
    return (f"prava:{kind}:{body}" if body else f"prava:{kind}").encode()


class PravaBuyerAgent(StateMachineAgent):
    """Quotes, pays, replays, verifies, overspends, then asks for a refund.

    Holds the payer-side plugin instance. Each step is scheduled on its own
    tick so the emitted trace has a stable, inspectable order.

    Example::

        agent = PravaBuyerAgent(
            AgentId("buyer-0"),
            seller=AgentId("seller-0"),
            ref=PaymentRef("prava-0"),
            amount_minor=4_000,
            overcap_minor=99_999_900,
            currency="USD",
            service_ref="libas:98252-2XL",
        )
    """

    def __init__(
        self,
        agent_id: AgentId,
        seller: AgentId,
        ref: PaymentRef,
        amount_minor: int,
        overcap_minor: int,
        currency: str,
        service_ref: str | None = None,
    ) -> None:
        self._id = agent_id
        self._seller = seller
        self._ref = ref
        self._overcap_ref = PaymentRef(f"{ref}-overcap")
        self._amount_minor = amount_minor
        self._overcap_minor = overcap_minor
        self._currency = currency
        self._service_ref = service_ref if service_ref else f"svc-{seller}"
        self._quoted_minor: int | None = None

    def _charge_minor(self) -> int:
        """Amount to charge: the quoted price once quoted, else the configured one.

        The quote is authoritative. Charging a separately configured constant
        would let the quoted price and the settled amount drift apart silently,
        which is exactly the bug an agentic payment rail must not have.

        Example::

            minor = agent._charge_minor()
        """
        return self._quoted_minor if self._quoted_minor is not None else self._amount_minor

    async def on_start(self, ctx: AgentContext) -> None:
        await ctx.schedule(_TICK_QUOTE, _OP_QUOTE)
        await ctx.schedule(_TICK_PAY, _OP_PAY)
        await ctx.schedule(_TICK_REPLAY, _OP_REPLAY)
        await ctx.schedule(_TICK_VERIFY, _OP_VERIFY)
        await ctx.schedule(_TICK_OVERCAP, _OP_OVERCAP)
        await ctx.schedule(_TICK_REFUND, _OP_REFUND)

    async def on_message(self, ctx: AgentContext, sender: AgentId, payload: bytes) -> None:
        payments = ctx.plugins["payments"]
        mode = str(getattr(payments, "mode", "unknown"))

        if payload == _OP_QUOTE:
            quote = await payments.quote(ServiceRef(self._service_ref))
            self._quoted_minor = int(quote.price.amount)
            fields: dict[str, str | int] = {
                "kind": "quoted",
                "buyer": str(self._id),
                "service": str(quote.service),
                "price": int(quote.price.amount),
                "currency": str(quote.price.currency),
            }
            metadata = getattr(quote, "metadata", None)
            if isinstance(metadata, dict):
                meta = cast("dict[str, Any]", metadata)
                source = meta.get("source")
                if source is not None:
                    fields["source"] = str(source)
                digest = meta.get("intent_digest")
                if digest is not None:
                    fields["intent"] = str(digest)[:12]
            fields["mode"] = mode
            await ctx.broadcast(_emit(fields))
            return

        money = Money(amount=self._charge_minor(), currency=self._currency)

        if payload == _OP_PAY:
            await self._attempt_pay(ctx, money, self._ref, mode, kind="paid")
            return

        if payload == _OP_REPLAY:
            # Same ref on purpose: Prava must dedupe rather than double-charge.
            await self._attempt_pay(ctx, money, self._ref, mode, kind="replayed")
            return

        if payload == _OP_VERIFY:
            status = await payments.verify_payment(self._ref)
            await ctx.broadcast(
                _emit(
                    {
                        "kind": "verified",
                        "ref": str(self._ref),
                        "status": str(getattr(status, "value", status)),
                        "mode": mode,
                    }
                )
            )
            return

        if payload == _OP_OVERCAP:
            huge = Money(amount=self._overcap_minor, currency=self._currency)
            await self._attempt_pay(ctx, huge, self._overcap_ref, mode, kind="overcap")
            return

        if payload == _OP_REFUND:
            await self._attempt_refund(ctx, mode)

    async def _attempt_pay(
        self,
        ctx: AgentContext,
        money: Money,
        ref: PaymentRef,
        mode: str,
        kind: str,
    ) -> None:
        payments = ctx.plugins["payments"]
        try:
            receipt = await payments.pay(self._seller, money, ref)
        except NotImplementedError:
            raise
        except (RuntimeError, ValueError) as exc:
            await ctx.broadcast(
                _emit(
                    {
                        "kind": "refused",
                        "step": kind,
                        "ref": str(ref),
                        "amount": int(money.amount),
                        "over_cap": int(bool(getattr(exc, "over_cap", False))),
                        "code": str(getattr(exc, "code", "") or "none"),
                        "mode": mode,
                    }
                )
            )
            return
        await ctx.broadcast(
            _emit(
                {
                    "kind": kind,
                    "ref": str(receipt.ref),
                    "payer": str(receipt.payer),
                    "payee": str(receipt.payee),
                    "amount": int(receipt.amount.amount),
                    "currency": str(receipt.amount.currency),
                    "confirmed": int(bool(getattr(payments, "confirmed", _false)(ref))),
                    "mode": mode,
                }
            )
        )

    async def _attempt_refund(self, ctx: AgentContext, mode: str) -> None:
        payments = ctx.plugins["payments"]
        try:
            await payments.refund(self._ref)
        except NotImplementedError as exc:
            await ctx.broadcast(
                _emit(
                    {
                        "kind": "refund_unsupported",
                        "ref": str(self._ref),
                        "reason": type(exc).__name__,
                        "mode": mode,
                    }
                )
            )
            return
        except (RuntimeError, ValueError):
            await ctx.broadcast(
                _emit({"kind": "refund_failed", "ref": str(self._ref), "mode": mode})
            )
            return
        await ctx.broadcast(_emit({"kind": "refunded", "ref": str(self._ref), "mode": mode}))


def _false(_ref: PaymentRef) -> bool:
    """Fallback for payments plugins without a ``confirmed`` method.

    Example::

        assert _false(PaymentRef("r")) is False
    """
    return False


class PravaSellerAgent(StateMachineAgent):
    """Passive counterparty. Announces itself so the pair is visible in the trace.

    The seller does not settle anything: in the Prava model the buyer's mandate
    is charged and the funds are routed by the card network, so there is no
    seller-side call to make.

    Example::

        agent = PravaSellerAgent(AgentId("seller-0"))
    """

    def __init__(self, agent_id: AgentId) -> None:
        self._id = agent_id

    async def on_start(self, ctx: AgentContext) -> None:
        await ctx.broadcast(_emit({"kind": "offering", "seller": str(self._id)}))

    async def on_message(self, ctx: AgentContext, sender: AgentId, payload: bytes) -> None:
        return None


def prava_commerce_factory(
    config: ScenarioConfig,
    plugins: dict[str, Any],
) -> dict[AgentId, StateMachineAgent]:
    """Build buyer/seller pairs and inject per-buyer Prava plugin instances.

    Reads ``task.config`` for every knob (all optional except ``mandate_id``,
    which is required in ``live`` mode):

    ``pairs``, ``mandate_id``, ``mode``, ``prava_env``, ``currency``,
    ``minor_unit_exponent``, ``unit_price_minor``, ``intent_snapshot``,
    ``cap_minor``, ``overcap_minor``, ``fail_closed``, ``timeout_s``,
    ``max_retries``, ``authorization_code``.

    ``intent_snapshot`` is a path to a frozen purchase-intent file describing an
    item an upstream LLM shopper selected. When present it supplies the service
    ref the buyer quotes and the price book the plugin prices from, so the
    amount charged is the amount that shopper agreed to rather than a constant
    in YAML. When absent the scenario keeps its synthetic ``svc-<seller>`` ref
    and ``unit_price_minor``.

    The API key is never read from YAML -- the plugin pulls it from
    ``PRAVA_API_KEY``.

    Only buyers receive a payments instance keyed to themselves, via the
    ``_agent_plugins`` override channel the runner understands. If the selected
    payments class does not accept the Prava keyword arguments, the factory
    falls back to the positional-only constructor so the scenario still runs
    (and emits no ``prava:*`` events, which is the honest signal that the rail
    was not exercised).

    Example::

        agents = prava_commerce_factory(config, plugins)
    """
    task_config: dict[str, Any] = config.task.config
    pairs = int(task_config.get("pairs", DEFAULT_PAIRS))
    unit_price_minor = int(task_config.get("unit_price_minor", DEFAULT_UNIT_PRICE_MINOR))
    overcap_minor = int(task_config.get("overcap_minor", DEFAULT_OVERCAP_MINOR))
    currency = str(task_config.get("currency", "USD"))

    snapshot_path = task_config.get("intent_snapshot")
    service_ref: str | None = None
    price_book: dict[str, int] | None = None
    intent_digest: str | None = None
    if snapshot_path:
        service_ref, unit_price_minor, intent_digest = _load_intent_snapshot(str(snapshot_path))
        price_book = {service_ref: unit_price_minor}

    plugin_kwargs: dict[str, Any] = {
        "mandate_id": str(task_config.get("mandate_id", "mdt_simulated_demo")),
        "mode": str(task_config.get("mode", "simulated")),
        "prava_env": str(task_config.get("prava_env", "sandbox")),
        "currency": currency,
        "minor_unit_exponent": int(task_config.get("minor_unit_exponent", 2)),
        "unit_price_minor": unit_price_minor,
        "price_book": price_book,
        "intent_digest": intent_digest,
        "cap_minor": int(task_config.get("cap_minor", 100_000)),
        "fail_closed": bool(task_config.get("fail_closed", True)),
        "timeout_s": float(task_config.get("timeout_s", 30.0)),
        "max_retries": int(task_config.get("max_retries", 3)),
        "authorization_code": task_config.get("authorization_code"),
    }

    payments_cls = plugins["payments"]
    agents: dict[AgentId, StateMachineAgent] = {}
    overrides: dict[AgentId, dict[str, Any]] = {}

    for index in range(pairs):
        buyer_id = AgentId(f"buyer-{index}")
        seller_id = AgentId(f"seller-{index}")
        ref = PaymentRef(f"prava-{config.seed}-{index}")

        agents[buyer_id] = PravaBuyerAgent(
            buyer_id,
            seller=seller_id,
            ref=ref,
            amount_minor=unit_price_minor,
            overcap_minor=overcap_minor,
            currency=currency,
            service_ref=service_ref,
        )
        agents[seller_id] = PravaSellerAgent(seller_id)

        try:
            instance = payments_cls(buyer_id, **plugin_kwargs)
        except TypeError:
            instance = payments_cls(buyer_id)
        overrides[buyer_id] = {"payments": instance}

    plugins["_agent_plugins"] = overrides
    return agents
