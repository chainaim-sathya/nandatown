# Payments layer

**What it does.** Price a service, pay, verify a payment, refund.

## Interface

```python
class Payments(Protocol):
    async def quote(self, service: ServiceRef) -> Quote: ...
    async def pay(self, to: AgentId, amount: Money, ref: PaymentRef) -> Receipt: ...
    async def verify_payment(self, ref: PaymentRef) -> PaymentStatus: ...
    async def refund(self, ref: PaymentRef) -> None: ...
```

Full definition: [`nest_core/layers/payments.py`](../../packages/nest-core/nest_core/layers/payments.py).

## Default plugin

`prepaid_credits` — in-memory debit/credit ledger. Constant-price
quotes, raises on insufficient balance, supports refund by `PaymentRef`.

Source: [`nest_plugins_reference/payments/prepaid_credits.py`](../../packages/nest-plugins-reference/nest_plugins_reference/payments/prepaid_credits.py).

## Additional reference plugins

`streaming` — bilateral per-tick streams with mid-stream cancellation.

`empic_escrow` — EMPIC-shaped escrow for service providers and
consumers. Pull mode locks one request payment until accepted data is
delivered; pubsub mode pre-funds a maximum stream amount, releases one
tick for each accepted delivery, and refunds unused escrow on close.
The bundled `empic_payments` scenario demonstrates provider service
registration, consumer acceptance policy, pull refunds, and pubsub
overbilling protection. Its adversarial validators also check consumer /
provider / service binding by payment reference and reject traces that leak
private keys, API keys, bearer tokens, wallet secrets, or other live rail
secrets.

`prava` — settles on Visa rails through
[Prava](https://docs.prava.space)'s agentic-payments mandate API. The
only payments plugin here that does not settle against a Python
dictionary. `PaymentRef` is passed through verbatim as Prava's
`reference` idempotency key, so replaying a simulation is refused by the
card network rather than by local bookkeeping. `refund` raises rather
than no-ops, because the Prava API defines no reversal endpoint. Ships
a simulated transport (a loudly labelled test double, used by CI, which
holds no credentials) alongside the live one, selected by a `mode` flag.

Details: [`prava_payments.md`](prava_payments.md).

## Writing your own

See [`writing-a-plugin.md`](../writing-a-plugin.md) — the full
walkthrough on that page builds a custom payments plugin end-to-end.
Register under entry point group `nest.plugins.payments`.

Good fits to test here: escrow, streaming payments, multi-party
settlement, on-chain stubs, x402-style HTTP payments.
