# SPDX-License-Identifier: Apache-2.0
"""Drives the real Prava payments plugin and renders the result as plain English.

This module contains NO payment logic of its own. It imports
``nest_plugins_reference.payments.prava`` and calls the same methods the
``prava_commerce`` scenario calls, in the same order:

    quote -> record_delivery -> check_delivery -> pay -> verify_payment
          -> refund -> judge_return

Nothing here is mocked. The default ``mode="simulated"`` selects
``SimulatedPravaTransport``, which is an in-process test double: it moves no
money, mints no credentials and contacts no network. That is a real code path
in the plugin, not a stand-in written for this service. Every result returned
carries ``mode`` so a caller can never mistake a rehearsal for a settlement.

Live mode is reachable only when all three of these hold:

    * ``ALLOW_LIVE=1`` in the environment
    * ``PRAVA_API_KEY`` is set
    * the caller passes ``mode="live"`` explicitly

A public URL that could charge a real card from a GET request would be a bad
idea, so the demo routes never pass ``mode``.

Example::

    result = await run_purchase(case="drift")
    assert result["verdict"] == "refused"
    assert result["reason"] == "colour_not_in_intent"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- package resolution -----------------------------------------------------
# The plugin lives in the repo's uv workspace under packages/. Both packages
# are pure Python with trivial __init__ files, so putting their source roots on
# sys.path is enough -- no editable install, no change to uv.lock, no new
# workspace member. REPO_ROOT is <repo>/ because this file is
# <repo>/apps/skill-service/runner.py.

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the Nanda Town repo root.

Example::

    snapshot = REPO_ROOT / "scenarios" / "intents" / "libas-98252-2XL.json"
"""

_PACKAGE_DIRS = ("nest-core", "nest-plugins-reference")


def _ensure_packages_on_path() -> None:
    """Put the workspace package source roots on ``sys.path`` if they are there.

    Silent when a directory is absent: if the packages were installed properly
    (``uv sync``) the imports below resolve anyway, and if neither is true the
    ImportError that follows is a clearer error than anything raised here.

    Example::

        _ensure_packages_on_path()
    """
    for name in _PACKAGE_DIRS:
        candidate = REPO_ROOT / "packages" / name
        if candidate.is_dir():
            resolved = str(candidate)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)


_ensure_packages_on_path()

from nest_core.types import (  # noqa: E402
    AgentId,
    Money,
    PaymentRef,
    ServiceRef,
)
from nest_plugins_reference.payments.prava import (  # noqa: E402
    MODE_LIVE,
    MODE_SIMULATED,
    PravaConformanceError,
    PravaPaymentError,
    PravaPayments,
    PravaRefundUnsupportedError,
)
from nest_plugins_reference.payments.prava_intent import (  # noqa: E402
    PurchaseIntent,
    load_intent,
)

DEFAULT_INTENTS_DIR = REPO_ROOT / "scenarios" / "intents"
"""Where the frozen purchase-intent snapshots live in this repo.

Override with the ``NEST_INTENTS_DIR`` environment variable.

Example::

    intents = DEFAULT_INTENTS_DIR
"""

GOOD_SNAPSHOT = "libas-98252-2XL.json"
"""Navy kurti, INR 499 listed, settles at 523 USD minor. Conforms."""

DRIFT_SNAPSHOT = "libas-98115-2XL-drift.json"
"""Maroon kurti, INR 450 listed, settles at 472 USD minor. Cheaper. Does not conform."""

OVERCAP_MINOR = 99_999_900
"""Deliberately oversized charge, used to provoke the mandate cap refusal."""


class CaseError(ValueError):
    """Raised for an unknown case name.

    Example::

        raise CaseError("no such case: banana")
    """


@dataclass(frozen=True)
class Case:
    """One demo scenario: which snapshot, what was delivered, what to charge.

    ``charge_minor_override`` exists only for the ``overcap`` case, which runs
    with no certified intent so the refusal comes from the mandate cap rather
    than from the conformance gate. See the note in ``CASES`` below.

    Example::

        case = CASES["drift"]
    """

    name: str
    summary: str
    snapshot: str | None
    delivery_product_id: str | None
    delivery_merchant: str | None
    require_conformance: bool
    charge_minor_override: int | None = None


CASES: dict[str, Case] = {
    "good": Case(
        name="good",
        summary=(
            "The agent bought the navy kurti it was asked for. Every certified "
            "term matches, so the charge goes through."
        ),
        snapshot=GOOD_SNAPSHOT,
        delivery_product_id="98252-2XL",
        delivery_merchant="Libas",
        require_conformance=True,
    ),
    "drift": Case(
        name="drift",
        summary=(
            "The agent could not find navy, so it bought a maroon kurti instead. "
            "It is CHEAPER than the one that was quoted and sits under the Visa "
            "decline threshold, so no amount-based control objects. Only the "
            "certified colour list catches it."
        ),
        snapshot=DRIFT_SNAPSHOT,
        delivery_product_id="98115-2XL",
        delivery_merchant="Libas",
        require_conformance=True,
    ),
    # No snapshot on purpose. With a certified intent loaded, an oversized
    # charge fails the `amount_mismatch` check inside the plugin BEFORE any
    # network call, so it is refused as INTENT_NONCONFORMANT and the mandate cap
    # is never reached. Running this case without the intent gate is the only
    # way to show the real THRESHOLD_EXCEEDED refusal coming back from the rail.
    "overcap": Case(
        name="overcap",
        summary=(
            "No certified intent. The agent tries to charge far more than the "
            "mandate allows. The refusal comes from the payment rail, not from "
            "the agent deciding to behave."
        ),
        snapshot=None,
        delivery_product_id=None,
        delivery_merchant=None,
        require_conformance=False,
        charge_minor_override=OVERCAP_MINOR,
    ),
}


def _step(name: str, ok: bool, detail: str, **extra: Any) -> dict[str, Any]:
    """Build one entry in the returned ``steps`` list.

    Example::

        _step("pay", True, "charged 523")
    """
    entry: dict[str, Any] = {"step": name, "ok": ok, "detail": detail}
    entry.update(extra)
    return entry


def _intent_summary(intent: PurchaseIntent) -> dict[str, Any]:
    """Flatten the parts of a snapshot a caller needs to see.

    Example::

        summary = _intent_summary(intent)
    """
    certificate = intent.certificate
    reference = certificate.reference if certificate is not None else None
    return {
        "intent_id": intent.intent_id,
        "sku": intent.sku,
        "title": intent.title,
        "colour": intent.colour,
        "merchant": intent.merchant_name,
        "service_ref": intent.service_ref,
        "listed": f"{intent.listed_amount_minor} {intent.listed_currency} minor",
        "settlement_amount_minor": intent.settlement_amount_minor,
        "settlement_currency": intent.settlement_currency,
        "converted": intent.converted,
        "fx_rate": intent.fx_rate,
        "fx_source": intent.fx_source,
        "constructed": intent.constructed,
        "digest": intent.digest[:12],
        "mandate_digest": (intent.mandate_digest[:12] if intent.mandate_digest else None),
        "original_request": (certificate.original_request if certificate else None),
        "allowed_colours": list(reference.allowed_colours) if reference else [],
        "return_window_days": reference.return_window_days if reference else None,
        "decline_threshold_minor": (
            intent.visa.decline_threshold_minor if intent.visa is not None else None
        ),
    }


def _resolve_mode(requested: str) -> str:
    """Return the mode to actually use, refusing live unless all gates are open.

    Example::

        assert _resolve_mode("live") == "simulated"   # when ALLOW_LIVE is unset
    """
    if requested != MODE_LIVE:
        return MODE_SIMULATED
    if os.environ.get("ALLOW_LIVE") != "1":
        msg = "live mode is disabled on this deployment; set ALLOW_LIVE=1 to enable it"
        raise PravaPaymentError(msg, code="LIVE_DISABLED")
    if not os.environ.get("PRAVA_API_KEY"):
        msg = "live mode needs PRAVA_API_KEY in the environment"
        raise PravaPaymentError(msg, code="NO_API_KEY")
    return MODE_LIVE


async def run_purchase(
    *,
    case: str = "good",
    mode: str = MODE_SIMULATED,
    mandate_id: str = "mdt_simulated_demo",
    prava_env: str = "sandbox",
    buyer: str = "buyer-0",
    seller: str = "seller-0",
    payment_ref: str = "prava-demo-0",
    currency: str = "USD",
    minor_unit_exponent: int = 2,
    cap_minor: int = 100_000,
    fail_closed: bool = True,
    days_since_delivery: int = 5,
    intents_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one full purchase cycle against the real plugin and report what happened.

    Every knob is an explicit keyword argument with a default, so behaviour can
    be changed by a caller without editing this file.

    Example::

        result = await run_purchase(case="good", days_since_delivery=40)

    Raises:
        CaseError: If ``case`` is not one of ``good``, ``drift``, ``overcap``.
        PravaPaymentError: If live mode was asked for but is not permitted.
    """
    if case not in CASES:
        known = ", ".join(sorted(CASES))
        msg = f"unknown case {case!r}; known cases are: {known}"
        raise CaseError(msg)

    spec = CASES[case]
    effective_mode = _resolve_mode(mode)
    directory = intents_dir or Path(os.environ.get("NEST_INTENTS_DIR", DEFAULT_INTENTS_DIR))

    steps: list[dict[str, Any]] = []
    intent: PurchaseIntent | None = None
    snapshot_path: Path | None = None
    price_book: dict[str, int] | None = None
    service_ref = f"svc-{seller}"
    charge_minor = 4_000

    # --- load the frozen snapshot -------------------------------------------
    if spec.snapshot is not None:
        snapshot_path = directory / spec.snapshot
        intent = load_intent(snapshot_path, expected_currency=currency)
        service_ref = intent.service_ref
        charge_minor = intent.settlement_amount_minor
        price_book = {service_ref: charge_minor}
        steps.append(
            _step(
                "load_intent",
                True,
                (
                    f"loaded the agreed purchase: {intent.title} "
                    f"({intent.colour}) from {intent.merchant_name}"
                ),
                snapshot=spec.snapshot,
                digest=intent.digest[:12],
            )
        )

    payments = PravaPayments(
        AgentId(buyer),
        mandate_id=mandate_id,
        mode=effective_mode,
        prava_env=prava_env,
        currency=currency,
        minor_unit_exponent=minor_unit_exponent,
        unit_price_minor=charge_minor,
        price_book=price_book,
        intent_digest=intent.digest if intent is not None else None,
        intent=intent if spec.require_conformance else None,
        cap_minor=cap_minor,
        fail_closed=fail_closed,
    )

    verdict = "unknown"
    reason = "unknown"
    code: str | None = None
    settled_minor = 0

    try:
        # --- quote ----------------------------------------------------------
        quote = await payments.quote(ServiceRef(service_ref))
        quoted_minor = int(quote.price.amount)
        steps.append(
            _step(
                "quote",
                True,
                f"price is {quoted_minor} {quote.price.currency} minor units",
                service=str(quote.service),
                source=str(quote.metadata.get("source", "unknown")),
            )
        )

        charge_minor = spec.charge_minor_override or quoted_minor

        # --- delivery + conformance check -----------------------------------
        if spec.delivery_product_id is not None:
            payments.record_delivery(
                product_id=spec.delivery_product_id,
                merchant=spec.delivery_merchant,
            )
            result = payments.check_delivery(amount_minor=charge_minor)
            failed = [
                {
                    "check": check.name,
                    "expected": check.expected,
                    "actual": check.actual,
                }
                for check in result.checks
                if not check.passed
            ]
            steps.append(
                _step(
                    "check_delivery",
                    result.passed,
                    (
                        "the delivery matches everything that was agreed"
                        if result.passed
                        else f"the delivery does not match: {result.reason}"
                    ),
                    delivered=spec.delivery_product_id,
                    merchant=spec.delivery_merchant,
                    reason=result.reason,
                    failed_checks=failed,
                )
            )

        # --- pay ------------------------------------------------------------
        money = Money(amount=charge_minor, currency=currency)
        ref = PaymentRef(payment_ref)
        try:
            receipt = await payments.pay(AgentId(seller), money, ref)
        except PravaConformanceError as exc:
            verdict, reason, code = "refused", exc.result.reason, exc.code
            steps.append(
                _step(
                    "pay",
                    False,
                    "refused before any network call: the item does not match what was agreed",
                    code=exc.code,
                    reason=exc.result.reason,
                )
            )
        except PravaPaymentError as exc:
            verdict = "refused"
            reason = "over_mandate_cap" if exc.over_cap else "declined"
            code = exc.code
            steps.append(
                _step(
                    "pay",
                    False,
                    (
                        "the payment rail refused it: the charge is over the mandate limit"
                        if exc.over_cap
                        else "the payment rail refused it"
                    ),
                    code=exc.code,
                    over_cap=exc.over_cap,
                )
            )
        else:
            settled_minor = int(receipt.amount.amount)
            verdict, reason = "settled", "conforms"
            steps.append(
                _step(
                    "pay",
                    True,
                    f"paid {settled_minor} {receipt.amount.currency} minor units to {seller}",
                    ref=str(receipt.ref),
                    confirmed=payments.confirmed(ref),
                )
            )

            # --- replay: same reference on purpose ---------------------------
            replay = await payments.pay(AgentId(seller), money, ref)
            steps.append(
                _step(
                    "replay",
                    True,
                    (
                        "sent the same payment reference again; the rail treated it as "
                        "the original charge instead of charging twice"
                    ),
                    ref=str(replay.ref),
                )
            )

        # --- verify against the rail's own ledger ---------------------------
        status = await payments.verify_payment(ref)
        steps.append(
            _step(
                "verify",
                True,
                f"the payment rail says this charge is: {status.value}",
                status=status.value,
            )
        )

        # --- refund: not supported by Prava ---------------------------------
        try:
            await payments.refund(ref)
        except PravaRefundUnsupportedError:
            steps.append(
                _step(
                    "refund",
                    False,
                    "no refund was possible: the Prava API has no refund endpoint at all",
                    error="PravaRefundUnsupportedError",
                )
            )

        # --- return window --------------------------------------------------
        if intent is not None and spec.require_conformance:
            if verdict == "settled":
                returned = payments.judge_return(days_since_delivery=days_since_delivery)
                steps.append(
                    _step(
                        "return",
                        returned.approved,
                        (
                            f"a return asked for on day {days_since_delivery} is "
                            f"{'inside' if returned.approved else 'outside'} the agreed "
                            f"{returned.window_days}-day window. Approved means it matches "
                            "the agreed terms; it does not mean money moved."
                        ),
                        approved=returned.approved,
                        reason=returned.reason,
                        window_days=returned.window_days,
                        settlement=("unavailable_on_rail" if returned.approved else None),
                    )
                )
            else:
                steps.append(
                    _step(
                        "return",
                        False,
                        "no return to judge: nothing was ever paid for",
                        reason="no_settled_purchase",
                    )
                )
    finally:
        await payments.aclose()

    payload: dict[str, Any] = {
        "case": spec.name,
        "summary": spec.summary,
        "mode": effective_mode,
        "mandate_id": mandate_id,
        "verdict": verdict,
        "reason": reason,
        "code": code,
        "settled_amount_minor": settled_minor,
        "currency": currency,
        "steps": steps,
        "intent": _intent_summary(intent) if intent is not None else None,
        "snapshot": str(snapshot_path.name) if snapshot_path is not None else None,
    }
    payload["display"] = render_display(payload)
    return payload


def render_display(payload: dict[str, Any]) -> str:
    """Turn a result into a transcript a person can read without a manual.

    Example::

        print(render_display(result))
    """
    lines: list[str] = []
    intent = payload.get("intent")
    if isinstance(intent, dict):
        request = intent.get("original_request")
        if request:
            lines.append(f'The user asked: "{request}"')
        lines.append(
            f"The agent chose: {intent.get('title')} ({intent.get('colour')}) "
            f"from {intent.get('merchant')}, {intent.get('listed')}."
        )
        colours = intent.get("allowed_colours") or []
        if colours:
            lines.append(f"Colours that were agreed: {', '.join(colours)}.")
    else:
        lines.append("No agreed purchase was loaded for this run.")

    lines.append("")
    for entry in payload.get("steps", []):
        mark = "OK  " if entry.get("ok") else "STOP"
        lines.append(f"  {mark} {entry.get('step')}: {entry.get('detail')}")

    lines.append("")
    if payload.get("verdict") == "settled":
        lines.append(
            f"Result: PAID {payload.get('settled_amount_minor')} "
            f"{payload.get('currency')} minor units."
        )
    else:
        lines.append(
            f"Result: NOT PAID. Nothing reached the card rail. "
            f"Reason: {payload.get('reason')}."
        )
    lines.append(f"Mode: {payload.get('mode')} (simulated means no real money moved).")
    return "\n".join(lines)
