# SPDX-License-Identifier: Apache-2.0
"""Tests for purchase-intent snapshots and the price book they feed ``quote``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nest_core.types import AgentId, Money, PaymentRef, ServiceRef
from nest_plugins_reference.payments.prava import PravaPayments
from nest_plugins_reference.payments.prava_client import PravaConfigError
from nest_plugins_reference.payments.prava_intent import (
    SCHEMA_V1,
    PravaIntentError,
    canonical_digest,
    load_intent,
    parse_intent,
    price_book,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_SNAPSHOT = REPO_ROOT / "scenarios" / "intents" / "libas-98252-2XL.json"


def _snapshot(**overrides: Any) -> dict[str, Any]:
    """Build a valid snapshot mapping, with optional top-level overrides."""
    payload: dict[str, Any] = {
        "schema": SCHEMA_V1,
        "captured_at": "2026-08-02T06:52:43Z",
        "source": "test",
        "merchant": {"name": "Libas", "endpoint": "https://www.libas.in/api/ucp/mcp"},
        "item": {
            "sku": "98252-2XL",
            "title": "Navy kurti",
            "service_ref": "libas:98252-2XL",
            "listed_price": {
                "amount_minor": 49_900,
                "currency": "INR",
                "minor_unit_exponent": 2,
            },
        },
        "settlement": {"amount_minor": 523, "currency": "USD", "minor_unit_exponent": 2},
        "fx": {
            "pair": "USD/INR",
            "rate": "95.4340",
            "source": "xe.com mid-market",
            "captured_at": "2026-08-02T02:20:00Z",
        },
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _payments(**kwargs: Any) -> PravaPayments:
    defaults: dict[str, Any] = {
        "mandate_id": "mdt_test",
        "mode": "simulated",
        "cap_minor": 100_000,
    }
    defaults.update(kwargs)
    return PravaPayments(AgentId("buyer-0"), **defaults)


# -- snapshot parsing ----------------------------------------------------


def test_parse_intent_reads_both_prices() -> None:
    intent = parse_intent(_snapshot())
    assert intent.service_ref == "libas:98252-2XL"
    assert intent.listed_amount_minor == 49_900
    assert intent.listed_currency == "INR"
    assert intent.settlement_amount_minor == 523
    assert intent.settlement_currency == "USD"
    assert intent.converted is True


def test_unconverted_snapshot_needs_no_fx_block() -> None:
    payload = _snapshot()
    payload["item"]["listed_price"]["currency"] = "USD"
    payload["item"]["listed_price"]["amount_minor"] = 523
    del payload["fx"]
    intent = parse_intent(payload)
    assert intent.converted is False
    assert intent.fx_rate is None


def test_converted_snapshot_without_fx_is_rejected() -> None:
    payload = _snapshot()
    del payload["fx"]
    with pytest.raises(PravaIntentError, match="carries no 'fx' block"):
        parse_intent(payload)


def test_unknown_schema_is_rejected() -> None:
    with pytest.raises(PravaIntentError, match="unsupported snapshot schema"):
        parse_intent(_snapshot(schema="nest.prava.intent/v99"))


def test_blank_service_ref_is_rejected() -> None:
    payload = _snapshot()
    payload["item"]["service_ref"] = "   "
    with pytest.raises(PravaIntentError, match="service_ref"):
        parse_intent(payload)


def test_non_positive_settlement_amount_is_rejected() -> None:
    payload = _snapshot()
    payload["settlement"]["amount_minor"] = 0
    with pytest.raises(PravaIntentError, match="must be positive"):
        parse_intent(payload)


def test_float_amount_is_rejected() -> None:
    payload = _snapshot()
    payload["settlement"]["amount_minor"] = 5.23
    with pytest.raises(PravaIntentError, match="minor units"):
        parse_intent(payload)


def test_digest_ignores_key_order() -> None:
    payload = _snapshot()
    reordered = dict(reversed(list(payload.items())))
    assert canonical_digest(payload) == canonical_digest(reordered)


def test_digest_changes_with_content() -> None:
    changed = _snapshot()
    changed["settlement"]["amount_minor"] = 524
    assert canonical_digest(_snapshot()) != canonical_digest(changed)


# -- loading -------------------------------------------------------------


def test_load_intent_round_trips(tmp_path: Path) -> None:
    intent = load_intent(_write(tmp_path, _snapshot()), expected_currency="USD")
    assert intent.settlement_amount_minor == 523


def test_load_intent_rejects_wrong_settlement_currency(tmp_path: Path) -> None:
    with pytest.raises(PravaIntentError, match="expects 'EUR'"):
        load_intent(_write(tmp_path, _snapshot()), expected_currency="EUR")


def test_load_intent_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PravaIntentError, match="cannot read"):
        load_intent(tmp_path / "absent.json")


def test_load_intent_reports_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "intent.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PravaIntentError, match="not valid JSON"):
        load_intent(path)


def test_shipped_snapshot_is_valid() -> None:
    if not SHIPPED_SNAPSHOT.exists():
        pytest.skip("shipped snapshot not present in this checkout")
    intent = load_intent(SHIPPED_SNAPSHOT, expected_currency="USD")
    assert intent.service_ref == "libas:98252-2XL"
    assert intent.settlement_amount_minor == 523
    assert intent.listed_amount_minor == 49_900


# -- price book ----------------------------------------------------------


def test_price_book_maps_service_ref_to_settlement_amount() -> None:
    assert price_book([parse_intent(_snapshot())]) == {"libas:98252-2XL": 523}


def test_price_book_rejects_duplicate_service_refs() -> None:
    intent = parse_intent(_snapshot())
    with pytest.raises(PravaIntentError, match="duplicate service_ref"):
        price_book([intent, intent])


# -- quote uses the book -------------------------------------------------


async def test_quote_prices_from_the_book() -> None:
    payments = _payments(price_book={"libas:98252-2XL": 523})
    quote = await payments.quote(ServiceRef("libas:98252-2XL"))
    assert quote.price.amount == 523
    assert quote.metadata["source"] == "prava_intent_snapshot"


async def test_quote_raises_on_an_item_no_snapshot_priced() -> None:
    payments = _payments(price_book={"libas:98252-2XL": 523})
    with pytest.raises(PravaConfigError, match="no price for service"):
        await payments.quote(ServiceRef("libas:00000-XS"))


async def test_quote_without_a_book_keeps_the_configured_unit_price() -> None:
    payments = _payments(unit_price_minor=4_000)
    quote = await payments.quote(ServiceRef("svc-seller-0"))
    assert quote.price.amount == 4_000
    assert quote.metadata["source"] == "prava_plugin_price_book"


async def test_quote_stamps_the_intent_digest() -> None:
    payments = _payments(price_book={"libas:98252-2XL": 523}, intent_digest="deadbeef")
    quote = await payments.quote(ServiceRef("libas:98252-2XL"))
    assert quote.metadata["intent_digest"] == "deadbeef"


async def test_charged_amount_equals_quoted_amount() -> None:
    payments = _payments(price_book={"libas:98252-2XL": 523})
    quote = await payments.quote(ServiceRef("libas:98252-2XL"))
    receipt = await payments.pay(
        AgentId("seller-0"),
        Money(amount=quote.price.amount, currency=quote.price.currency),
        PaymentRef("prava-42-0"),
    )
    assert receipt.amount.amount == quote.price.amount == 523
    await payments.aclose()
