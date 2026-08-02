# SPDX-License-Identifier: Apache-2.0
"""Frozen purchase-intent snapshots -- the handoff from an LLM shopper to Nanda Town.

An LLM agent picks a product out of a live merchant catalogue. That decision is
non-deterministic and involves network I/O, so it cannot happen inside a Nanda
Town scenario: the simulator guarantees that the same seed produces a
byte-identical trace. The decision is therefore made *once, out of band* and
written to a snapshot file. The scenario reads the snapshot.

The snapshot carries the item identity, the price the merchant listed, and --
when the listing currency differs from the settlement currency -- the converted
amount together with the exact FX rate, its source and its capture time. Nothing
is recomputed at run time. A run is reproducible because every number it needs is
already written down.

This module holds no Nanda Town types and performs no network I/O. It reads a
file and validates it.

Example::

    intent = load_intent(Path("scenarios/intents/libas-98252-2XL.json"))
    assert intent.service_ref == "libas:98252-2XL"
    assert intent.settlement_amount_minor == 523
    book = price_book([intent])
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
class PurchaseIntent:
    """One item an upstream agent selected, frozen with everything needed to charge it.

    ``settlement_amount_minor`` is what ``quote`` returns and what ``pay``
    charges. ``listed_amount_minor`` and the ``fx_*`` fields exist for
    provenance: they let a reader reconstruct how the settlement figure was
    reached without trusting it blindly.

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
        settlement_exponent=_exponent(
            settlement["minor_unit_exponent"], "snapshot.settlement.minor_unit_exponent"
        ),
        fx_rate=_optional_str(fx, "rate"),
        fx_source=_optional_str(fx, "source"),
        fx_captured_at=_optional_str(fx, "captured_at"),
        fx_rounding=_optional_str(fx, "rounding"),
        note=_optional_str(payload, "note"),
        digest=canonical_digest(payload),
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
