# SPDX-License-Identifier: Apache-2.0
"""Frozen purchase-intent snapshots -- the handoff from an LLM shopper to Nanda Town.

An LLM agent picks a product out of a live merchant catalogue. That decision is
non-deterministic and involves network I/O, so it cannot happen inside a Nanda
Town scenario: the simulator guarantees that the same seed produces a
byte-identical trace. The decision is therefore made *once, out of band* and
written to a snapshot file. The scenario reads the snapshot.

The snapshot carries three things. The **item** the agent selected, with the
price the merchant listed and -- when the listing currency differs from the
settlement currency -- the converted amount plus the exact FX rate, its source
and its capture time. The **certificate**: the user's original request turned
into a machine-checkable committed reference (allowed colours, a price ceiling,
the merchant, the quantity, the return window) by an out-of-band certifier. And
the **visa_intent**: the same commitment expressed in Visa Intelligent Commerce
purchase-intent vocabulary, recorded for conformance and audit -- it is not sent
to Visa.

Nothing is recomputed at run time. A run is reproducible because every number it
needs is already written down.

The certificate is what makes a *verdict* possible. A request like "one kurti
under 500 rupees in green, pink or navy on Libas" cannot be checked against a
delivery; ``allowed_colours``, ``max_price`` and ``merchant_name`` can.

This module holds no Nanda Town types and performs no network I/O. It reads a
file, validates it, and judges selections and returns against it.

Example::

    intent = load_intent(Path("scenarios/intents/libas-98252-2XL.json"))
    assert intent.settlement_amount_minor == 523
    result = check_conformance(intent)
    assert result.passed
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

SCHEMA_V1 = "nest.prava.intent/v1"
"""The only schema identifier this module accepts.

Example::

    assert intent.schema == SCHEMA_V1
"""

_REQUIRED_TOP = ("schema", "captured_at", "source", "merchant", "item", "settlement")
_REQUIRED_MERCHANT = ("name",)
_REQUIRED_ITEM = ("sku", "title", "service_ref", "listed_price")
_REQUIRED_PRICE = ("amount_minor", "currency", "minor_unit_exponent")
_REQUIRED_FX = ("rate", "source", "captured_at")


class PravaIntentError(ValueError):
    """Raised when a snapshot is missing, malformed or internally inconsistent.

    Deliberately strict: a snapshot is the record of what a human or an LLM
    agreed to buy, and a silently-defaulted field here becomes a wrong amount
    charged to a real card.

    Example::

        try:
            load_intent(path)
        except PravaIntentError as exc:
            print(f"bad snapshot: {exc}")
    """


@dataclass(frozen=True)
class CommittedReference:
    """The user's request as terms a delivery can be judged against.

    Produced out of band by a certifier. Every field is optional: a snapshot may
    commit to colours only, or to nothing at all, and each check is simply
    skipped when its term is absent.

    Example::

        ref = intent.certificate.reference
        assert "navy" in ref.allowed_colours
    """

    allowed_colours: tuple[str, ...] = ()
    max_price_minor: int | None = None
    max_price_currency: str | None = None
    merchant_name: str | None = None
    quantity: int | None = None
    return_window_days: int | None = None


@dataclass(frozen=True)
class Certificate:
    """An out-of-band certifier's anchored statement of what was agreed.

    ``digest`` is computed over the committed reference, so editing a term after
    certification changes the digest and the edit is detectable.

    Example::

        print(intent.certificate.certifier, intent.certificate.digest[:12])
    """

    certifier: str
    certified_at: str | None
    original_request: str | None
    reference: CommittedReference
    digest: str


@dataclass(frozen=True)
class VisaIntent:
    """The commitment in Visa Intelligent Commerce purchase-intent vocabulary.

    Field names mirror ``POST /acp/v1/instructions``. Recorded for conformance
    and audit only -- this artifact is not sent to Visa.

    Example::

        print(intent.visa.consumer_prompt, intent.visa.decline_threshold_minor)
    """

    consumer_prompt: str | None = None
    mandate_id: str | None = None
    preferred_merchant_name: str | None = None
    decline_threshold_minor: int | None = None
    decline_threshold_currency: str | None = None
    effective_until_time: str | None = None
    description: str | None = None
    product_id: str | None = None
    product_name: str | None = None
    refund_policy: str | None = None


@dataclass(frozen=True)
class ConformanceCheck:
    """One named term, whether it held, and the two values compared.

    Example::

        for check in result.checks:
            print(check.name, check.passed, check.expected, check.actual)
    """

    name: str
    passed: bool
    expected: str
    actual: str


@dataclass(frozen=True)
class ConformanceResult:
    """The verdict, plus every check that produced it.

    Example::

        if not result.passed:
            print(result.reason)
    """

    passed: bool
    checks: tuple[ConformanceCheck, ...] = ()

    @property
    def reason(self) -> str:
        """Name of the first failed check, or ``"conforms"``.

        Example::

            assert result.reason == "colour_not_in_intent"
        """
        for check in self.checks:
            if not check.passed:
                return check.name
        return "conforms"

    @property
    def failed(self) -> tuple[str, ...]:
        """Names of every failed check.

        Example::

            assert result.failed == ("colour_not_in_intent",)
        """
        return tuple(c.name for c in self.checks if not c.passed)


@dataclass(frozen=True)
class ReturnVerdict:
    """Whether a return request falls inside the certified window.

    ``approved`` means the request conforms to the terms agreed up front. It
    does **not** mean money moved: the Prava API defines no reversal endpoint,
    so settlement of an approved return has to happen off this rail.

    Example::

        verdict = judge_return(intent, days_since_delivery=5)
        assert verdict.approved and verdict.reason == "within_window"
    """

    approved: bool
    reason: str
    days_since_delivery: int
    window_days: int | None = None
    policy: str | None = None


@dataclass(frozen=True)
class PurchaseIntent:
    """One item an upstream agent selected, frozen with everything needed to charge it.

    ``settlement_amount_minor`` is what ``quote`` returns and what ``pay``
    charges. ``listed_amount_minor`` and the ``fx_*`` fields exist for
    provenance. ``certificate`` and ``visa`` carry the terms the selection and
    any delivery are judged against; both are optional, and a snapshot without
    them behaves exactly as it did before they existed.

    Example::

        intent = load_intent(path)
        print(intent.sku, intent.settlement_amount_minor)
    """

    schema: str
    captured_at: str
    source: str
    merchant_name: str
    merchant_endpoint: str | None
    sku: str
    title: str
    service_ref: str
    listed_amount_minor: int
    listed_currency: str
    listed_exponent: int
    settlement_amount_minor: int
    settlement_currency: str
    settlement_exponent: int
    fx_rate: str | None
    fx_source: str | None
    fx_captured_at: str | None
    fx_rounding: str | None
    note: str | None
    digest: str
    colour: str | None = None
    size: str | None = None
    constructed: bool = False
    certificate: Certificate | None = None
    visa: VisaIntent | None = field(default=None)

    @property
    def converted(self) -> bool:
        """Whether settlement currency differs from the merchant's listing currency.

        Example::

            if intent.converted:
                print(intent.fx_rate, intent.fx_source)
        """
        return self.listed_currency != self.settlement_currency


def canonical_digest(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest of ``payload``.

    Keys are sorted and separators fixed, so reformatting or reordering the
    source file does not change the digest -- only a change of *content* does.

    Example::

        digest = canonical_digest({"b": 2, "a": 1})
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require(mapping: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        msg = f"{where} is missing required field(s): {', '.join(sorted(missing))}"
        raise PravaIntentError(msg)


def _object(value: Any, where: str) -> dict[str, Any]:
    """Narrow a decoded JSON value to a string-keyed mapping.

    ``json.loads`` is typed ``Any`` and ``isinstance(x, dict)`` narrows only to
    ``dict[Unknown, Unknown]``, so the cast is what makes the rest of this
    module checkable under strict type checking. JSON object keys are strings by
    construction, so the cast is sound.

    Example::

        item = _object(payload["item"], "snapshot.item")
    """
    if not isinstance(value, dict):
        msg = f"{where} must be a JSON object, got {type(value).__name__}"
        raise PravaIntentError(msg)
    return cast("dict[str, Any]", value)


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        msg = f"{where} must be a JSON array, got {type(value).__name__}"
        raise PravaIntentError(msg)
    return cast("list[Any]", value)


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{where} must be an integer number of minor units, got {value!r}"
        raise PravaIntentError(msg)
    if value <= 0:
        msg = f"{where} must be positive, got {value}"
        raise PravaIntentError(msg)
    return value


def _exponent(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
        msg = f"{where} must be an integer between 0 and 4, got {value!r}"
        raise PravaIntentError(msg)
    return value


def _currency(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 3 or not value.isalpha():
        msg = f"{where} must be a three-letter ISO 4217 code, got {value!r}"
        raise PravaIntentError(msg)
    return value.upper()


def _optional_str(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    return str(value) if value is not None else None


def _optional_int(mapping: dict[str, Any], key: str, where: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{where}.{key} must be a non-negative integer, got {value!r}"
        raise PravaIntentError(msg)
    return value


def _amount_to_minor(value: str, exponent: int, where: str) -> int:
    """Convert a decimal amount string to integer minor units, half up.

    Example::

        assert _amount_to_minor("5.30", 2, "x") == 530
    """
    try:
        scaled = Decimal(value).scaleb(exponent)
        return int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError) as exc:
        msg = f"{where} must be a decimal amount string, got {value!r}"
        raise PravaIntentError(msg) from exc


def _parse_certificate(payload: dict[str, Any]) -> Certificate | None:
    raw = payload.get("certificate")
    if raw is None:
        return None
    cert = _object(raw, "snapshot.certificate")
    _require(cert, ("certifier", "committed_reference"), "snapshot.certificate")
    committed = _object(cert["committed_reference"], "snapshot.certificate.committed_reference")

    colours: tuple[str, ...] = ()
    if "allowed_colours" in committed:
        entries = _array(
            committed["allowed_colours"], "snapshot.certificate.committed_reference.allowed_colours"
        )
        colours = tuple(str(c).strip().lower() for c in entries if str(c).strip())

    max_minor: int | None = None
    max_currency: str | None = None
    if "max_price" in committed:
        price = _object(
            committed["max_price"], "snapshot.certificate.committed_reference.max_price"
        )
        _require(price, _REQUIRED_PRICE, "snapshot.certificate.committed_reference.max_price")
        max_minor = _positive_int(
            price["amount_minor"], "snapshot.certificate.committed_reference.max_price.amount_minor"
        )
        max_currency = _currency(
            price["currency"], "snapshot.certificate.committed_reference.max_price.currency"
        )

    reference = CommittedReference(
        allowed_colours=colours,
        max_price_minor=max_minor,
        max_price_currency=max_currency,
        merchant_name=_optional_str(committed, "merchant_name"),
        quantity=_optional_int(committed, "quantity", "snapshot.certificate.committed_reference"),
        return_window_days=_optional_int(
            committed, "return_window_days", "snapshot.certificate.committed_reference"
        ),
    )
    return Certificate(
        certifier=str(cert["certifier"]),
        certified_at=_optional_str(cert, "certified_at"),
        original_request=_optional_str(cert, "original_request"),
        reference=reference,
        digest=canonical_digest(committed),
    )


def _parse_visa_intent(payload: dict[str, Any], exponent: int) -> VisaIntent | None:
    raw = payload.get("visa_intent")
    if raw is None:
        return None
    visa = _object(raw, "snapshot.visa_intent")

    mandate: dict[str, Any] = {}
    if "mandates" in visa:
        mandates = _array(visa["mandates"], "snapshot.visa_intent.mandates")
        if mandates:
            mandate = _object(mandates[0], "snapshot.visa_intent.mandates[0]")

    threshold_minor: int | None = None
    threshold_currency: str | None = None
    if "declineThreshold" in mandate:
        threshold = _object(
            mandate["declineThreshold"], "snapshot.visa_intent.mandates[0].declineThreshold"
        )
        _require(
            threshold,
            ("amount", "currencyCode"),
            "snapshot.visa_intent.mandates[0].declineThreshold",
        )
        threshold_minor = _amount_to_minor(
            str(threshold["amount"]),
            exponent,
            "snapshot.visa_intent.mandates[0].declineThreshold.amount",
        )
        threshold_currency = _currency(
            threshold["currencyCode"],
            "snapshot.visa_intent.mandates[0].declineThreshold.currencyCode",
        )

    product: dict[str, Any] = {}
    if "products" in visa:
        products = _array(visa["products"], "snapshot.visa_intent.products")
        if products:
            product = _object(products[0], "snapshot.visa_intent.products[0]")

    refund_policy: str | None = None
    if "policies" in product:
        policies = _object(product["policies"], "snapshot.visa_intent.products[0].policies")
        refund_policy = _optional_str(policies, "refundPolicy")

    return VisaIntent(
        consumer_prompt=_optional_str(visa, "consumerPrompt"),
        mandate_id=_optional_str(mandate, "mandateId"),
        preferred_merchant_name=_optional_str(mandate, "preferredMerchantName"),
        decline_threshold_minor=threshold_minor,
        decline_threshold_currency=threshold_currency,
        effective_until_time=_optional_str(mandate, "effectiveUntilTime"),
        description=_optional_str(mandate, "description"),
        product_id=_optional_str(product, "productId"),
        product_name=_optional_str(product, "productName"),
        refund_policy=refund_policy,
    )


def parse_intent(payload: dict[str, Any], *, expected_schema: str = SCHEMA_V1) -> PurchaseIntent:
    """Validate a decoded snapshot mapping and build a :class:`PurchaseIntent`.

    Separated from :func:`load_intent` so callers holding a snapshot in memory
    (a test, an upstream generator) can validate it without touching disk.

    Example::

        intent = parse_intent(json.loads(text))

    Raises:
        PravaIntentError: If the schema is unknown, a required field is absent,
            an amount is not a positive integer, or a converted snapshot omits
            its FX provenance.
    """
    _require(payload, _REQUIRED_TOP, "snapshot")
    schema = str(payload["schema"])
    if schema != expected_schema:
        msg = f"unsupported snapshot schema {schema!r}; this build accepts {expected_schema!r}"
        raise PravaIntentError(msg)

    merchant = _object(payload["merchant"], "snapshot.merchant")
    item = _object(payload["item"], "snapshot.item")
    settlement = _object(payload["settlement"], "snapshot.settlement")
    _require(merchant, _REQUIRED_MERCHANT, "snapshot.merchant")
    _require(item, _REQUIRED_ITEM, "snapshot.item")
    _require(settlement, _REQUIRED_PRICE, "snapshot.settlement")

    listed = _object(item["listed_price"], "snapshot.item.listed_price")
    _require(listed, _REQUIRED_PRICE, "snapshot.item.listed_price")

    service_ref = item["service_ref"]
    if not isinstance(service_ref, str) or not service_ref.strip():
        msg = f"snapshot.item.service_ref must be a non-empty string, got {service_ref!r}"
        raise PravaIntentError(msg)

    listed_currency = _currency(listed["currency"], "snapshot.item.listed_price.currency")
    settlement_currency = _currency(settlement["currency"], "snapshot.settlement.currency")
    settlement_exponent = _exponent(
        settlement["minor_unit_exponent"], "snapshot.settlement.minor_unit_exponent"
    )

    fx: dict[str, Any] = {}
    if "fx" in payload:
        fx = _object(payload["fx"], "snapshot.fx")
    if listed_currency != settlement_currency:
        if not fx:
            msg = (
                f"snapshot converts {listed_currency} to {settlement_currency} but carries no "
                "'fx' block; the rate, its source and its capture time must be recorded so "
                "the conversion can be audited and reproduced"
            )
            raise PravaIntentError(msg)
        _require(fx, _REQUIRED_FX, "snapshot.fx")

    return PurchaseIntent(
        schema=schema,
        captured_at=str(payload["captured_at"]),
        source=str(payload["source"]),
        merchant_name=str(merchant["name"]),
        merchant_endpoint=_optional_str(merchant, "endpoint"),
        sku=str(item["sku"]),
        title=str(item["title"]),
        service_ref=service_ref,
        listed_amount_minor=_positive_int(
            listed["amount_minor"], "snapshot.item.listed_price.amount_minor"
        ),
        listed_currency=listed_currency,
        listed_exponent=_exponent(
            listed["minor_unit_exponent"], "snapshot.item.listed_price.minor_unit_exponent"
        ),
        settlement_amount_minor=_positive_int(
            settlement["amount_minor"], "snapshot.settlement.amount_minor"
        ),
        settlement_currency=settlement_currency,
        settlement_exponent=settlement_exponent,
        fx_rate=_optional_str(fx, "rate"),
        fx_source=_optional_str(fx, "source"),
        fx_captured_at=_optional_str(fx, "captured_at"),
        fx_rounding=_optional_str(fx, "rounding"),
        note=_optional_str(payload, "note"),
        digest=canonical_digest(payload),
        colour=_optional_str(item, "colour"),
        size=_optional_str(item, "size"),
        constructed=bool(payload.get("constructed", False)),
        certificate=_parse_certificate(payload),
        visa=_parse_visa_intent(payload, settlement_exponent),
    )


def load_intent(
    path: str | Path,
    *,
    expected_schema: str = SCHEMA_V1,
    expected_currency: str | None = None,
) -> PurchaseIntent:
    """Read and validate a snapshot file.

    ``expected_currency`` is an optional guard for the caller: pass the plugin's
    configured settlement currency and a mismatched snapshot is rejected here,
    at load time, rather than several ticks later inside ``pay``.

    Example::

        intent = load_intent(path, expected_currency="USD")

    Raises:
        PravaIntentError: If the file is absent, is not an object, fails
            validation, or settles in a currency other than
            ``expected_currency``.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read purchase-intent snapshot {p}: {exc}"
        raise PravaIntentError(msg) from exc
    try:
        decoded: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"purchase-intent snapshot {p} is not valid JSON: {exc}"
        raise PravaIntentError(msg) from exc
    if not isinstance(decoded, dict):
        msg = f"purchase-intent snapshot {p} must contain a JSON object at the top level"
        raise PravaIntentError(msg)

    intent = parse_intent(cast("dict[str, Any]", decoded), expected_schema=expected_schema)
    if expected_currency is not None and intent.settlement_currency != expected_currency.upper():
        msg = (
            f"snapshot {p} settles in {intent.settlement_currency!r} but the caller expects "
            f"{expected_currency.upper()!r}; convert upstream and record the rate, or "
            "reconfigure the plugin"
        )
        raise PravaIntentError(msg)
    return intent


def price_book(intents: list[PurchaseIntent]) -> dict[str, int]:
    """Build the ``service_ref -> settlement minor units`` mapping ``quote`` looks up.

    Duplicate service refs are rejected rather than last-write-wins: two
    snapshots claiming different prices for one item is an upstream bug, and
    silently picking one would charge an amount nobody agreed to.

    Example::

        book = price_book([intent])
        assert book["libas:98252-2XL"] == 523

    Raises:
        PravaIntentError: If two intents share a ``service_ref``.
    """
    book: dict[str, int] = {}
    for intent in intents:
        if intent.service_ref in book:
            msg = (
                f"duplicate service_ref {intent.service_ref!r} across purchase-intent "
                "snapshots; each item must appear exactly once"
            )
            raise PravaIntentError(msg)
        book[intent.service_ref] = intent.settlement_amount_minor
    return book


def check_conformance(
    intent: PurchaseIntent,
    *,
    delivered_product_id: str | None = None,
    delivered_amount_minor: int | None = None,
    delivered_merchant: str | None = None,
) -> ConformanceResult:
    """Judge a selection, and optionally a delivery, against the certified terms.

    Two independent axes. **Selection against the committed reference** catches
    an agent that drifted from what the user asked for -- a colour outside the
    allowed set, a price over the ceiling, the wrong merchant. **Delivery
    against the selection** catches a substitution at fulfilment -- a different
    product id, or an amount that is not the one quoted.

    Every check is skipped when the term it needs is absent, so a snapshot
    without a certificate yields an empty, passing result and the caller
    behaves exactly as it did before certificates existed. Delivery arguments
    default to the intent's own selection.

    Note that a cheaper substitution passes every amount-based control -- a
    decline threshold, a mandate cap, the network's own ceiling -- because all
    of those ask only *how much*. The colour and identity terms are what catch
    it.

    Example::

        result = check_conformance(intent, delivered_product_id="98115-2XL")
        assert not result.passed
        assert result.reason == "identity_mismatch"
    """
    checks: list[ConformanceCheck] = []
    product_id = delivered_product_id if delivered_product_id is not None else intent.sku
    amount_minor = (
        delivered_amount_minor
        if delivered_amount_minor is not None
        else intent.settlement_amount_minor
    )
    merchant = delivered_merchant if delivered_merchant is not None else intent.merchant_name

    reference = intent.certificate.reference if intent.certificate is not None else None

    if reference is not None and reference.allowed_colours:
        actual = (intent.colour or "").strip().lower()
        checks.append(
            ConformanceCheck(
                name="colour_not_in_intent",
                passed=actual in reference.allowed_colours,
                expected="|".join(reference.allowed_colours),
                actual=actual or "unspecified",
            )
        )

    if (
        reference is not None
        and reference.max_price_minor is not None
        and reference.max_price_currency == intent.listed_currency
    ):
        checks.append(
            ConformanceCheck(
                name="price_over_ceiling",
                passed=intent.listed_amount_minor <= reference.max_price_minor,
                expected=f"<={reference.max_price_minor}",
                actual=str(intent.listed_amount_minor),
            )
        )

    if reference is not None and reference.merchant_name is not None:
        checks.append(
            ConformanceCheck(
                name="merchant_mismatch",
                passed=merchant.strip().lower() == reference.merchant_name.strip().lower(),
                expected=reference.merchant_name,
                actual=merchant,
            )
        )

    checks.append(
        ConformanceCheck(
            name="identity_mismatch",
            passed=product_id == intent.sku,
            expected=intent.sku,
            actual=product_id,
        )
    )

    checks.append(
        ConformanceCheck(
            name="amount_mismatch",
            passed=amount_minor == intent.settlement_amount_minor,
            expected=str(intent.settlement_amount_minor),
            actual=str(amount_minor),
        )
    )

    if intent.visa is not None and intent.visa.decline_threshold_minor is not None:
        checks.append(
            ConformanceCheck(
                name="over_decline_threshold",
                passed=amount_minor <= intent.visa.decline_threshold_minor,
                expected=f"<={intent.visa.decline_threshold_minor}",
                actual=str(amount_minor),
            )
        )

    return ConformanceResult(passed=all(c.passed for c in checks), checks=tuple(checks))


def judge_return(intent: PurchaseIntent, *, days_since_delivery: int) -> ReturnVerdict:
    """Decide whether a return request falls inside the certified window.

    ``days_since_delivery`` is supplied by the caller rather than derived from
    the clock: a wall-clock read would make the verdict, and therefore the
    trace, non-reproducible.

    An approved verdict means the request conforms to terms agreed before the
    purchase. It does not mean money moved -- the Prava API defines no reversal
    endpoint, so settling an approved return happens off this rail.

    Example::

        verdict = judge_return(intent, days_since_delivery=5)
        assert verdict.approved

    Raises:
        PravaIntentError: If ``days_since_delivery`` is negative.
    """
    if days_since_delivery < 0:
        msg = f"days_since_delivery must not be negative, got {days_since_delivery}"
        raise PravaIntentError(msg)

    policy = intent.visa.refund_policy if intent.visa is not None else None
    window = (
        intent.certificate.reference.return_window_days if intent.certificate is not None else None
    )
    if window is None:
        return ReturnVerdict(
            approved=False,
            reason="no_return_window_certified",
            days_since_delivery=days_since_delivery,
            policy=policy,
        )
    inside = days_since_delivery <= window
    return ReturnVerdict(
        approved=inside,
        reason="within_window" if inside else "outside_window",
        days_since_delivery=days_since_delivery,
        window_days=window,
        policy=policy,
    )
