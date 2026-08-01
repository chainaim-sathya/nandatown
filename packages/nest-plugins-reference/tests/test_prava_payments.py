# SPDX-License-Identifier: Apache-2.0
"""Tests for the Prava payments plugin and its API client.

Every test in the default suite is credential-free: it runs against
``SimulatedPravaTransport``, the in-process test double, so CI needs no
``PRAVA_API_KEY``. The one test that touches the real sandbox carries the
``live`` marker and is deselected by the repo-wide ``-m "not live"``.

Coverage: minor-unit conversion, quote, pay/report, idempotent replay,
authoritative verify_payment read-back, mandate-cap refusal, the deliberate
refund refusal, currency handling, live-transport key/host guards, the
unconfirmed-report policy, and registry resolution.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from nest_core.plugins import PluginRegistry
from nest_core.types import AgentId, Money, PaymentRef, PaymentStatus, ServiceRef
from nest_plugins_reference.payments.prava import (
    ENV_SANDBOX,
    MODE_LIVE,
    MODE_SIMULATED,
    PravaPaymentError,
    PravaPayments,
    PravaRefundUnsupportedError,
)
from nest_plugins_reference.payments.prava_client import (
    CHARGE_STATUS_AWAITING,
    CHARGE_STATUS_FAILED,
    PRODUCTION_BASE_URL,
    SANDBOX_BASE_URL,
    THRESHOLD_EXCEEDED,
    ChargeResult,
    HttpxPravaTransport,
    PravaClient,
    PravaConfigError,
    PravaError,
    PravaTransport,
    SimulatedPravaTransport,
)

MANDATE_ID = "mdt_test_0001"
BUYER = AgentId("buyer-0")
SELLER = AgentId("seller-0")

# Environment variables that drive the single `live` test. Names are constants
# rather than literals so the README and the test cannot drift apart.
ENV_API_KEY = "PRAVA_API_KEY"
ENV_MANDATE_ID = "PRAVA_MANDATE_ID"
ENV_LIVE_AMOUNT_MINOR = "PRAVA_LIVE_AMOUNT_MINOR"
ENV_LIVE_REFERENCE = "PRAVA_LIVE_REFERENCE"


def _transport(
    *,
    cap_minor: int = 100_000,
    currency: str = "USD",
    exponent: int = 2,
    fail_references: frozenset[str] = frozenset(),
) -> SimulatedPravaTransport:
    """Build the in-process test double. Moves no money, contacts no network."""
    return SimulatedPravaTransport(
        mandate_id=MANDATE_ID,
        cap_minor=cap_minor,
        currency=currency,
        exponent=exponent,
        fail_references=fail_references,
    )


def _payments(
    *,
    cap_minor: int = 100_000,
    currency: str = "USD",
    exponent: int = 2,
    unit_price_minor: int = 4_000,
    fail_closed: bool = True,
    fail_references: frozenset[str] = frozenset(),
    agent: AgentId = BUYER,
) -> tuple[PravaPayments, SimulatedPravaTransport]:
    """Build a plugin bound to a fresh double, returning both for inspection."""
    transport = _transport(
        cap_minor=cap_minor,
        currency=currency,
        exponent=exponent,
        fail_references=fail_references,
    )
    payments = PravaPayments(
        agent,
        mandate_id=MANDATE_ID,
        mode=MODE_SIMULATED,
        currency=currency,
        minor_unit_exponent=exponent,
        unit_price_minor=unit_price_minor,
        fail_closed=fail_closed,
        transport=transport,
    )
    return payments, transport


async def _detail(transport: PravaTransport) -> dict[str, Any]:
    """Read the mandate back the way ``verify_payment`` does."""
    return await transport.request("GET", f"/v1/mandates/{MANDATE_ID}")


def _references(detail: dict[str, Any]) -> list[str]:
    charges: list[Any] = detail["charges"]
    return [str(entry["reference"]) for entry in charges]


class _FlakyTransport:
    """Wrapper that injects transport failures. A TEST DOUBLE, not Prava.

    Failures are selected by flag rather than hardcoded so one class covers
    both the failed-merchant-report policy and the unreachable-API policy.
    """

    def __init__(
        self,
        inner: SimulatedPravaTransport,
        *,
        fail_report: bool = False,
        fail_get: bool = False,
        status: int = 503,
        code: str = "UPSTREAM_UNAVAILABLE",
    ) -> None:
        self._inner = inner
        self._fail_report = fail_report
        self._fail_get = fail_get
        self._status = status
        self._code = code

    async def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self._fail_report and path.endswith("/report"):
            raise PravaError(self._status, {"error": {"code": self._code}})
        if self._fail_get and method == "GET":
            raise PravaError(self._status, {"error": {"code": self._code}})
        return await self._inner.request(method, path, body, params)

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---------------------------------------------------------------------------
# Minor-unit <-> decimal-string conversion
# ---------------------------------------------------------------------------


class TestAmountConversion:
    """Nanda Town Money is integer minor units; Prava wants a decimal string."""

    @pytest.mark.parametrize(
        ("minor", "exponent", "expected"),
        [
            (4_000, 2, "40.00"),
            (0, 2, "0.00"),
            (1, 2, "0.01"),
            (100, 2, "1.00"),
            (123_456, 2, "1234.56"),
            (4_000, 0, "4000"),
            (0, 0, "0"),
            (1_234, 3, "1.234"),
        ],
    )
    def test_minor_to_decimal_string(self, minor: int, exponent: int, expected: str) -> None:
        from nest_plugins_reference.payments.prava_client import minor_to_decimal_string

        assert minor_to_decimal_string(minor, exponent) == expected

    @pytest.mark.parametrize(
        ("amount", "exponent", "expected"),
        [
            ("40.00", 2, 4_000),
            ("40", 2, 4_000),
            ("0.01", 2, 1),
            ("1234.56", 2, 123_456),
            ("4000", 0, 4_000),
            ("1.234", 3, 1_234),
        ],
    )
    def test_decimal_string_to_minor(self, amount: str, exponent: int, expected: int) -> None:
        from nest_plugins_reference.payments.prava_client import decimal_string_to_minor

        assert decimal_string_to_minor(amount, exponent) == expected

    @pytest.mark.parametrize("minor", [0, 1, 7, 99, 100, 4_000, 999_999])
    @pytest.mark.parametrize("exponent", [0, 2, 3])
    def test_round_trip(self, minor: int, exponent: int) -> None:
        from nest_plugins_reference.payments.prava_client import (
            decimal_string_to_minor,
            minor_to_decimal_string,
        )

        assert decimal_string_to_minor(minor_to_decimal_string(minor, exponent), exponent) == minor

    def test_half_up_rounding_is_explicit(self) -> None:
        from nest_plugins_reference.payments.prava_client import decimal_string_to_minor

        # 40.005 -> 4000.5 minor units -> ROUND_HALF_UP -> 4001, not banker's 4000.
        assert decimal_string_to_minor("40.005", 2) == 4_001
        assert decimal_string_to_minor("40.004", 2) == 4_000

    def test_zero_decimal_currency_is_not_silently_wrong(self) -> None:
        from nest_plugins_reference.payments.prava_client import minor_to_decimal_string

        # JPY has exponent 0: 4000 yen is "4000", never "40.00".
        assert minor_to_decimal_string(4_000, 0) == "4000"
        assert minor_to_decimal_string(4_000, 2) == "40.00"

    def test_negative_exponent_rejected(self) -> None:
        from nest_plugins_reference.payments.prava_client import (
            decimal_string_to_minor,
            minor_to_decimal_string,
        )

        with pytest.raises(PravaConfigError, match="exponent must be"):
            minor_to_decimal_string(100, -1)
        with pytest.raises(PravaConfigError, match="exponent must be"):
            decimal_string_to_minor("1.00", -1)

    def test_non_decimal_string_rejected(self) -> None:
        from nest_plugins_reference.payments.prava_client import decimal_string_to_minor

        with pytest.raises(PravaConfigError, match="not a decimal amount"):
            decimal_string_to_minor("forty dollars")


# ---------------------------------------------------------------------------
# ChargeResult envelope normalization
# ---------------------------------------------------------------------------


class TestChargeResult:
    """``accepted`` and ``over_cap`` collapse Prava's status + errorCode."""

    def test_accepted_charge(self) -> None:
        result = ChargeResult(
            {
                "status": CHARGE_STATUS_AWAITING,
                "transactionId": "txn_1",
                "orderId": "ord_1",
                "instructionId": "ins_1",
                "deduplicated": False,
            }
        )
        assert result.accepted is True
        assert result.over_cap is False
        assert result.transaction_id == "txn_1"

    def test_over_cap_refusal(self) -> None:
        result = ChargeResult({"status": CHARGE_STATUS_FAILED, "errorCode": THRESHOLD_EXCEEDED})
        assert result.accepted is False
        assert result.over_cap is True

    def test_ordinary_decline_is_not_over_cap(self) -> None:
        result = ChargeResult({"status": CHARGE_STATUS_FAILED, "errorCode": "CHARGE_DECLINED"})
        assert result.accepted is False
        assert result.over_cap is False


# ---------------------------------------------------------------------------
# The simulated transport is a labelled double, and it is honest about limits
# ---------------------------------------------------------------------------


class TestSimulatedTransport:
    """The double must satisfy the Protocol and refuse to invent behaviour."""

    def test_satisfies_transport_protocol(self) -> None:
        assert isinstance(_transport(), PravaTransport)
        assert isinstance(_FlakyTransport(_transport()), PravaTransport)

    @pytest.mark.asyncio
    async def test_unknown_mandate_is_404(self) -> None:
        transport = _transport()
        with pytest.raises(PravaError) as exc_info:
            await transport.request("GET", "/v1/mandates/mdt_someone_else")
        assert exc_info.value.status == 404
        assert exc_info.value.code == "MANDATE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_unsimulated_path_raises_rather_than_faking(self) -> None:
        transport = _transport()
        with pytest.raises(NotImplementedError, match="does not simulate"):
            await transport.request("POST", "/v1/sessions", body={"mandate_setup": {}})

    @pytest.mark.asyncio
    async def test_identifiers_are_deterministic_across_runs(self) -> None:
        seen: list[list[str]] = []
        for _ in range(2):
            payments, transport = _payments()
            await payments.pay(SELLER, Money(amount=4_000, currency="USD"), PaymentRef("r-1"))
            await payments.pay(SELLER, Money(amount=1_500, currency="USD"), PaymentRef("r-2"))
            detail = await _detail(transport)
            charges: list[Any] = detail["charges"]
            seen.append([str(entry["transactionId"]) for entry in charges])
        assert seen[0] == seen[1]


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Every knob is validated at construction, before any money is at risk."""

    def test_mandate_id_is_required(self) -> None:
        with pytest.raises(PravaConfigError, match="mandate_id is required"):
            PravaPayments(BUYER, mandate_id="", mode=MODE_SIMULATED)

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(PravaConfigError, match="mode must be one of"):
            PravaPayments(BUYER, mandate_id=MANDATE_ID, mode="dry_run")

    def test_unknown_env_rejected(self) -> None:
        with pytest.raises(PravaConfigError, match="prava_env must be one of"):
            PravaPayments(BUYER, mandate_id=MANDATE_ID, prava_env="staging")

    def test_live_mode_without_key_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        with pytest.raises(PravaConfigError, match="requires a Prava secret key"):
            PravaPayments(BUYER, mandate_id=MANDATE_ID, mode=MODE_LIVE)

    def test_mode_and_mandate_are_public(self) -> None:
        payments, _ = _payments()
        assert payments.mode == MODE_SIMULATED
        assert payments.mandate_id == MANDATE_ID


class TestHttpxTransportGuards:
    """A key/host mismatch is silent and expensive, so it is refused up front."""

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(PravaConfigError, match="api_key is required"):
            HttpxPravaTransport(api_key="", base_url=SANDBOX_BASE_URL)

    def test_live_key_against_sandbox_rejected(self) -> None:
        with pytest.raises(PravaConfigError, match="live key"):
            HttpxPravaTransport(api_key="sk_live_abc", base_url=SANDBOX_BASE_URL)

    def test_test_key_against_production_rejected(self) -> None:
        with pytest.raises(PravaConfigError, match="test key"):
            HttpxPravaTransport(api_key="sk_test_abc", base_url=PRODUCTION_BASE_URL)

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(PravaConfigError, match="max_retries must be"):
            HttpxPravaTransport(api_key="sk_test_abc", base_url=SANDBOX_BASE_URL, max_retries=-1)


# ---------------------------------------------------------------------------
# quote
# ---------------------------------------------------------------------------


class TestQuote:
    """Prava has no quote endpoint; the price book is local and deterministic."""

    @pytest.mark.asyncio
    async def test_quote_returns_configured_price(self) -> None:
        payments, _ = _payments(unit_price_minor=2_500)
        quote = await payments.quote(ServiceRef("svc-1"))
        assert quote.service == ServiceRef("svc-1")
        assert quote.price.amount == 2_500
        assert quote.price.currency == "USD"

    @pytest.mark.asyncio
    async def test_quote_discloses_mode(self) -> None:
        payments, _ = _payments()
        quote = await payments.quote(ServiceRef("svc-1"))
        assert quote.metadata["mode"] == MODE_SIMULATED

    @pytest.mark.asyncio
    async def test_quote_makes_no_charge(self) -> None:
        payments, transport = _payments()
        await payments.quote(ServiceRef("svc-1"))
        detail = await _detail(transport)
        assert detail["chargeCount"] == 0


# ---------------------------------------------------------------------------
# pay
# ---------------------------------------------------------------------------


class TestPay:
    """``pay`` charges the mandate, then reports the merchant-side outcome."""

    @pytest.mark.asyncio
    async def test_pay_returns_receipt(self) -> None:
        payments, _ = _payments()
        ref = PaymentRef("scenario-42-trade-0")
        receipt = await payments.pay(SELLER, Money(amount=4_000, currency="USD"), ref)
        assert receipt.ref == ref
        assert receipt.payer == BUYER
        assert receipt.payee == SELLER
        assert receipt.amount.amount == 4_000
        assert receipt.amount.currency == "USD"

    @pytest.mark.asyncio
    async def test_pay_appears_in_the_mandate_ledger(self) -> None:
        payments, transport = _payments()
        await payments.pay(SELLER, Money(amount=4_000, currency="USD"), PaymentRef("r-1"))
        detail = await _detail(transport)
        assert detail["chargeCount"] == 1
        assert detail["spent"] == "40.00"
        assert detail["remaining"] == "960.00"
        assert _references(detail) == ["r-1"]

    @pytest.mark.asyncio
    async def test_pay_caches_receipt_and_confirmation(self) -> None:
        payments, _ = _payments()
        ref = PaymentRef("r-1")
        await payments.pay(SELLER, Money(amount=4_000, currency="USD"), ref)
        cached = payments.receipt(ref)
        assert cached is not None
        assert cached.ref == ref
        assert payments.confirmed(ref) is True

    @pytest.mark.asyncio
    async def test_unknown_ref_has_no_cached_receipt(self) -> None:
        payments, _ = _payments()
        assert payments.receipt(PaymentRef("never-paid")) is None
        assert payments.confirmed(PaymentRef("never-paid")) is False

    @pytest.mark.asyncio
    async def test_non_positive_amount_rejected(self) -> None:
        payments, _ = _payments()
        with pytest.raises(PravaConfigError, match="must be positive"):
            await payments.pay(SELLER, Money(amount=0, currency="USD"), PaymentRef("r-1"))

    @pytest.mark.asyncio
    async def test_zero_decimal_currency_charges_whole_units(self) -> None:
        payments, transport = _payments(currency="JPY", exponent=0, cap_minor=10_000)
        await payments.pay(SELLER, Money(amount=4_000, currency="JPY"), PaymentRef("r-jpy"))
        detail = await _detail(transport)
        assert detail["spent"] == "4000"
        assert detail["currency"] == "JPY"


class TestIdempotentReplay:
    """PaymentRef is Prava's idempotency key: replay safety is rail-enforced."""

    @pytest.mark.asyncio
    async def test_replayed_ref_does_not_draw_the_mandate_twice(self) -> None:
        payments, transport = _payments()
        ref = PaymentRef("scenario-42-trade-0")
        money = Money(amount=4_000, currency="USD")

        first = await payments.pay(SELLER, money, ref)
        after_first = await _detail(transport)
        second = await payments.pay(SELLER, money, ref)
        after_second = await _detail(transport)

        assert first.ref == second.ref
        assert after_second["spent"] == after_first["spent"] == "40.00"
        assert after_second["chargeCount"] == 1
        assert _references(after_second) == [str(ref)]

    @pytest.mark.asyncio
    async def test_replay_is_flagged_deduplicated_at_the_client(self) -> None:
        client = PravaClient(_transport(), minor_unit_exponent=2)
        first = await client.charge(MANDATE_ID, amount_minor=4_000, reference="r-1")
        second = await client.charge(MANDATE_ID, amount_minor=4_000, reference="r-1")
        assert first.deduplicated is False
        assert second.deduplicated is True
        assert second.transaction_id == first.transaction_id

    @pytest.mark.asyncio
    async def test_distinct_refs_are_distinct_charges(self) -> None:
        payments, transport = _payments()
        money = Money(amount=1_000, currency="USD")
        await payments.pay(SELLER, money, PaymentRef("r-1"))
        await payments.pay(SELLER, money, PaymentRef("r-2"))
        detail = await _detail(transport)
        assert detail["chargeCount"] == 2
        assert detail["spent"] == "20.00"


class TestClientValidation:
    """The client refuses malformed charges without spending a round trip."""

    @pytest.mark.asyncio
    async def test_non_positive_amount_rejected(self) -> None:
        client = PravaClient(_transport())
        with pytest.raises(PravaConfigError, match="amount_minor must be positive"):
            await client.charge(MANDATE_ID, amount_minor=0, reference="r-1")

    @pytest.mark.asyncio
    async def test_empty_reference_rejected(self) -> None:
        client = PravaClient(_transport())
        with pytest.raises(PravaConfigError, match="reference must be"):
            await client.charge(MANDATE_ID, amount_minor=100, reference="")

    @pytest.mark.asyncio
    async def test_overlong_reference_rejected(self) -> None:
        client = PravaClient(_transport())
        with pytest.raises(PravaConfigError, match="reference must be"):
            await client.charge(MANDATE_ID, amount_minor=100, reference="x" * 256)

    @pytest.mark.asyncio
    async def test_max_length_reference_accepted(self) -> None:
        client = PravaClient(_transport())
        result = await client.charge(MANDATE_ID, amount_minor=100, reference="x" * 255)
        assert result.accepted is True

    @pytest.mark.asyncio
    async def test_find_charge_returns_none_for_unknown_reference(self) -> None:
        client = PravaClient(_transport())
        assert await client.find_charge(MANDATE_ID, "never-charged") is None


# ---------------------------------------------------------------------------
# verify_payment -- authoritative read-back
# ---------------------------------------------------------------------------


class TestVerifyPayment:
    """The answer comes from Prava's ledger, never from local state."""

    @pytest.mark.asyncio
    async def test_confirmed_after_pay(self) -> None:
        payments, _ = _payments()
        ref = PaymentRef("r-1")
        await payments.pay(SELLER, Money(amount=4_000, currency="USD"), ref)
        assert await payments.verify_payment(ref) == PaymentStatus.CONFIRMED

    @pytest.mark.asyncio
    async def test_failed_for_unknown_ref(self) -> None:
        payments, _ = _payments()
        assert await payments.verify_payment(PaymentRef("never-paid")) == PaymentStatus.FAILED

    @pytest.mark.asyncio
    async def test_pending_for_charged_but_unreported(self) -> None:
        payments, transport = _payments()
        client = PravaClient(transport, minor_unit_exponent=2)
        # Charge without the merchant-report leg: Prava's ledger still says
        # awaiting_result, so the honest answer is PENDING, not CONFIRMED.
        await client.charge(MANDATE_ID, amount_minor=1_000, reference="r-unreported")
        status = await payments.verify_payment(PaymentRef("r-unreported"))
        assert status == PaymentStatus.PENDING

    @pytest.mark.asyncio
    async def test_unreachable_api_is_pending_not_failed(self) -> None:
        # An unreachable API is not evidence that a payment did not happen.
        inner = _transport()
        payments = PravaPayments(
            BUYER,
            mandate_id=MANDATE_ID,
            mode=MODE_SIMULATED,
            transport=_FlakyTransport(inner, fail_get=True),
        )
        assert await payments.verify_payment(PaymentRef("r-1")) == PaymentStatus.PENDING

    @pytest.mark.asyncio
    async def test_refused_charge_is_not_in_the_ledger(self) -> None:
        payments, _ = _payments(cap_minor=5_000)
        await payments.pay(SELLER, Money(amount=4_000, currency="USD"), PaymentRef("r-1"))
        with pytest.raises(PravaPaymentError):
            await payments.pay(SELLER, Money(amount=4_000, currency="USD"), PaymentRef("r-2"))
        assert await payments.verify_payment(PaymentRef("r-2")) == PaymentStatus.FAILED


# ---------------------------------------------------------------------------
# Refusals: mandate cap, decline, unconfirmed report
# ---------------------------------------------------------------------------


class TestRefusals:
    """Prava's refusals are surfaced, never swallowed or retried into success."""

    @pytest.mark.asyncio
    async def test_over_cap_raises_with_threshold_code(self) -> None:
        payments, transport = _payments(cap_minor=5_000)
        await payments.pay(SELLER, Money(amount=4_000, currency="USD"), PaymentRef("r-1"))
        with pytest.raises(PravaPaymentError) as exc_info:
            await payments.pay(SELLER, Money(amount=4_000, currency="USD"), PaymentRef("r-2"))
        assert exc_info.value.over_cap is True
        assert exc_info.value.code == THRESHOLD_EXCEEDED
        detail = await _detail(transport)
        assert detail["spent"] == "40.00"
        assert detail["chargeCount"] == 1

    @pytest.mark.asyncio
    async def test_decline_is_not_over_cap(self) -> None:
        payments, _ = _payments(fail_references=frozenset({"r-declined"}))
        with pytest.raises(PravaPaymentError) as exc_info:
            await payments.pay(
                SELLER, Money(amount=1_000, currency="USD"), PaymentRef("r-declined")
            )
        assert exc_info.value.over_cap is False
        assert exc_info.value.code == "CHARGE_DECLINED"

    @pytest.mark.asyncio
    async def test_fail_closed_refuses_a_receipt_when_report_fails(self) -> None:
        payments = PravaPayments(
            BUYER,
            mandate_id=MANDATE_ID,
            mode=MODE_SIMULATED,
            fail_closed=True,
            transport=_FlakyTransport(_transport(), fail_report=True),
        )
        with pytest.raises(PravaPaymentError, match="UNCONFIRMED"):
            await payments.pay(SELLER, Money(amount=1_000, currency="USD"), PaymentRef("r-1"))

    @pytest.mark.asyncio
    async def test_fail_open_returns_an_unconfirmed_receipt(self) -> None:
        inner = _transport()
        payments = PravaPayments(
            BUYER,
            mandate_id=MANDATE_ID,
            mode=MODE_SIMULATED,
            fail_closed=False,
            transport=_FlakyTransport(inner, fail_report=True),
        )
        ref = PaymentRef("r-1")
        receipt = await payments.pay(SELLER, Money(amount=1_000, currency="USD"), ref)
        assert receipt.ref == ref
        # The receipt exists, but the plugin does not claim confirmation...
        assert payments.confirmed(ref) is False
        # ...and the ledger read-back agrees it is still in flight.
        assert await payments.verify_payment(ref) == PaymentStatus.PENDING


class TestRefundIsUnsupported:
    """Prava defines no refund, credit or reversal path. We do not fake one."""

    @pytest.mark.asyncio
    async def test_refund_raises(self) -> None:
        payments, _ = _payments()
        ref = PaymentRef("r-1")
        await payments.pay(SELLER, Money(amount=4_000, currency="USD"), ref)
        with pytest.raises(PravaRefundUnsupportedError, match="no refund"):
            await payments.refund(ref)

    @pytest.mark.asyncio
    async def test_refund_raises_even_for_unknown_ref(self) -> None:
        payments, _ = _payments()
        with pytest.raises(PravaRefundUnsupportedError):
            await payments.refund(PaymentRef("never-paid"))

    def test_error_is_a_notimplementederror(self) -> None:
        # Rooted in a stdlib base on purpose: nest_core catches it in the
        # scenario without importing nest_plugins_reference.
        assert issubclass(PravaRefundUnsupportedError, NotImplementedError)

    def test_payment_error_is_a_runtimeerror(self) -> None:
        assert issubclass(PravaPaymentError, RuntimeError)

    def test_config_error_is_a_valueerror(self) -> None:
        assert issubclass(PravaConfigError, ValueError)

    def test_client_exposes_no_refund_method(self) -> None:
        assert not hasattr(PravaClient, "refund")


# ---------------------------------------------------------------------------
# Currency handling
# ---------------------------------------------------------------------------


class TestCurrency:
    """Configured ISO code, plus Nanda Town's layer-neutral ``credits``."""

    @pytest.mark.asyncio
    async def test_mismatched_currency_rejected(self) -> None:
        payments, _ = _payments(currency="USD")
        with pytest.raises(PravaConfigError, match="currency mismatch"):
            await payments.pay(SELLER, Money(amount=1_000, currency="EUR"), PaymentRef("r-1"))

    @pytest.mark.asyncio
    async def test_credits_accepted_and_settled_in_configured_currency(self) -> None:
        payments, transport = _payments(currency="USD")
        receipt = await payments.pay(
            SELLER, Money(amount=1_000, currency="credits"), PaymentRef("r-1")
        )
        assert receipt.amount.currency == "USD"
        detail = await _detail(transport)
        assert detail["currency"] == "USD"

    @pytest.mark.asyncio
    async def test_mismatch_is_rejected_before_any_charge(self) -> None:
        payments, transport = _payments(currency="USD")
        with pytest.raises(PravaConfigError):
            await payments.pay(SELLER, Money(amount=1_000, currency="GBP"), PaymentRef("r-1"))
        detail = await _detail(transport)
        assert detail["chargeCount"] == 0


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    """The plugin must be reachable by name, which is how scenarios load it."""

    def test_resolves_by_layer_and_name(self) -> None:
        assert PluginRegistry().resolve("payments", "prava") is PravaPayments

    def test_listed_under_the_payments_layer(self) -> None:
        assert ("payments", "prava") in PluginRegistry().list_plugins("payments")


# ---------------------------------------------------------------------------
# Live sandbox test -- deselected by default via the repo-wide -m "not live"
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_sandbox_charge_is_idempotent() -> None:
    """Charge a real sandbox mandate twice and prove the rail deduplicates.

    Requires a mandate provisioned out of band (POST /v1/sessions plus a human
    passkey approval). Run with::

        PRAVA_API_KEY=sk_test_... PRAVA_MANDATE_ID=mdt_... \
            uv run pytest -m live -k live_sandbox
    """
    api_key = os.environ.get(ENV_API_KEY, "")
    mandate_id = os.environ.get(ENV_MANDATE_ID, "")
    if not api_key or not mandate_id:
        pytest.skip(f"set {ENV_API_KEY} and {ENV_MANDATE_ID} to run the live Prava test")
    if not api_key.startswith("sk_test_"):
        pytest.skip("live test runs against the sandbox only; expected an sk_test_ key")

    amount_minor = int(os.environ.get(ENV_LIVE_AMOUNT_MINOR, "100"))
    reference = os.environ.get(ENV_LIVE_REFERENCE, "nest-prava-live-0001")

    payments = PravaPayments(
        BUYER,
        mandate_id=mandate_id,
        mode=MODE_LIVE,
        prava_env=ENV_SANDBOX,
        api_key=api_key,
        unit_price_minor=amount_minor,
    )
    try:
        ref = PaymentRef(reference)
        receipt = await payments.pay(SELLER, Money(amount=amount_minor, currency="USD"), ref)
        assert receipt.amount.amount == amount_minor
        assert await payments.verify_payment(ref) in (
            PaymentStatus.CONFIRMED,
            PaymentStatus.PENDING,
        )

        client = PravaClient(
            HttpxPravaTransport(api_key=api_key, base_url=SANDBOX_BASE_URL),
            minor_unit_exponent=2,
        )
        try:
            replay = await client.charge(mandate_id, amount_minor=amount_minor, reference=reference)
            assert replay.deduplicated is True
            charge = await client.find_charge(mandate_id, reference)
            assert charge is not None
        finally:
            await client.aclose()
    finally:
        await payments.aclose()
