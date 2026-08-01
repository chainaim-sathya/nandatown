# Prava Payments: settling agent commerce on Visa rails

Plugin: `("payments", "prava")` —
[`nest_plugins_reference/payments/prava.py`](../../packages/nest-plugins-reference/nest_plugins_reference/payments/prava.py)
Client: [`prava_client.py`](../../packages/nest-plugins-reference/nest_plugins_reference/payments/prava_client.py)
Scenario: [`scenarios/prava_commerce.yaml`](../../scenarios/prava_commerce.yaml)

## Problem

Every payments plugin in this repo settles against a Python dictionary.
`prepaid_credits` moves integers between balances; `escrow` holds them in a
vault; `streaming` drips them per tick. All three are correct, and none of them
can tell you what happens when an agent's spending is governed by something
outside the simulation.

That gap matters most for **replay safety**. Nanda Town derives payment refs
deterministically from the scenario seed, so re-running a simulation re-issues
the identical `PaymentRef`. Under an in-process ledger, the protection against
double-charging is a `if ref in self._payments: raise` — our own bookkeeping
checking our own bookkeeping. It proves the plugin is internally consistent. It
proves nothing about whether the same agent, pointed at a real rail, would
charge a real card twice.

## Solution

`PravaPayments` maps the four `Payments` protocol methods onto
[Prava](https://docs.prava.space)'s agentic-payments mandate API, and passes
`PaymentRef` through **verbatim** as Prava's `reference` field — which Prava
documents as an idempotency key.

| Protocol method | Prava call |
|---|---|
| `quote` | None. Local price book; Prava authorizes and settles, it does not price. Keeping it local also keeps `quote` deterministic under replay. |
| `pay` | `POST /v1/mandates/{id}/charge`, then `POST /v1/mandates/{id}/charges/{txn}/report` |
| `verify_payment` | `GET /v1/mandates/{id}`, then locate the charge by `reference` |
| `refund` | Raises. See [No refunds exist](#no-refunds-exist-and-we-do-not-pretend-otherwise). |

The consequence is the point: replay safety stops being an assertion in our code
and becomes a property enforced by the card network. Replay the simulation, and
Prava — not `nest_core` — refuses the second charge.

`verify_payment` is authoritative for the same reason. It re-reads the mandate
and finds the charge in Prava's own `charges[]` array, so it reports what the
payment rail believes, not what this process believes. A network failure returns
`PENDING`, never `FAILED`: an unreachable API is not evidence that a payment did
not happen.

## Install

The plugin ships inside `nest-plugins-reference` and needs no extra step:

```bash
uv sync
uv run nest run scenarios/prava_commerce.yaml
```

`httpx>=0.27` is a declared dependency of `nest-plugins-reference`. The plugin is
registered two ways, so it resolves whether or not the package is installed as a
distribution:

```python
from nest_core.plugins import PluginRegistry
PluginRegistry().resolve("payments", "prava")   # -> PravaPayments
```

— via `_BUILTINS` in `nest_core/plugins.py`, and via the
`[project.entry-points."nest.plugins.payments"]` block in
`packages/nest-plugins-reference/pyproject.toml`.

### Reuse outside this repo

`prava_client.py` imports **no Nanda Town types**. It is a standalone async Prava
client — transport protocol, error envelopes, retry policy, and the integer
minor-unit ↔ decimal-string conversion Prava's wire format requires — and can be
lifted into any project:

```python
from nest_plugins_reference.payments.prava_client import (
    HttpxPravaTransport, PravaClient, SANDBOX_BASE_URL,
)

client = PravaClient(HttpxPravaTransport(api_key="sk_test_...", base_url=SANDBOX_BASE_URL))
result = await client.charge("mdt_...", amount_minor=4_000, reference="order-1")
```

The Nanda Town binding lives entirely in `prava.py`.

## Two modes, never confused

```yaml
task:
  config:
    mode: simulated            # simulated | live
    prava_env: sandbox         # sandbox | production
    mandate_id: mdt_simulated_demo
```

`mode: simulated` selects `SimulatedPravaTransport`, an in-process test double.
**It moves no money, mints no credentials, and contacts no network.** It exists
because CI holds no Prava key and because determinism is a graded axis. Its
identifiers come from a monotonic counter rather than `uuid4` or the clock, so a
replayed simulation yields a byte-identical trace.

`mode: live` selects `HttpxPravaTransport` against the sandbox or production
host. The **identical plugin code path** runs in both modes — only the transport
differs.

There is deliberately **no `dry_run` flag**. An earlier draft had one, returning
a synthetic `Receipt` without calling out. That was removed: a `dry_run` receipt
makes a rehearsed trace indistinguishable from a settled one. `PravaPayments.mode`
is public, `PravaPayments.confirmed(ref)` reports whether the merchant-report leg
closed, and every scenario broadcast carries `mode=simulated` or `mode=live`. A
reader can always tell what actually happened.

`HttpxPravaTransport` refuses an `sk_live_` key pointed at the sandbox host and
an `sk_test_` key pointed at production, because that mistake is silent and
expensive.

## Mandate provisioning is out of band

Creating a mandate requires `POST /v1/sessions` with a `mandate_setup` block and
a **human passkey approval**. That cannot happen mid-simulation. So `mandate_id`
is provisioned beforehand and injected as config, and the API key is read from
`PRAVA_API_KEY` — never from scenario YAML.

```bash
export PRAVA_API_KEY=sk_test_...
# then set mode: live and mandate_id: mdt_<yours> in the YAML
uv run nest run scenarios/prava_commerce.yaml
```

This is a real constraint on agentic commerce, not an implementation shortcut:
an autonomous agent cannot mint its own spending authority. A human grants a
capped mandate; the agent spends within it; the rail enforces the cap.

## No refunds exist, and we do not pretend otherwise

The Prava OpenAPI 3.1 document was read in full — all 13 paths. **There is no
refund, credit, or reversal endpoint.**

`refund()` therefore raises `PravaRefundUnsupportedError`. It does not silently
no-op, and it does not return `None` as though something happened. A plugin that
swallows an unsupported refund is lying about its capability, and the lie only
surfaces in production.

Callers needing a reversal must model it as a compensating **forward** payment to
the original payer, with a fresh `PaymentRef`, recording both legs:

```python
try:
    await payments.refund(ref)
except PravaRefundUnsupportedError:
    await seller_payments.pay(buyer, amount, PaymentRef(f"{ref}-reversal"))
```

`PravaRefundUnsupportedError` subclasses `NotImplementedError`, and
`PravaPaymentError` subclasses `RuntimeError`, on purpose: the scenario in
`nest_core` catches them by their stdlib bases, so `nest_core` never imports
`nest_plugins_reference`.

## Configuration

All knobs are explicit keyword arguments with defaults, surfaced through
`task.config`. Nothing is hardcoded.

| Key | Default | Meaning |
|---|---|---|
| `mandate_id` | *required* | Provisioned out of band. |
| `mode` | `simulated` | `simulated` \| `live`. |
| `prava_env` | `sandbox` | `sandbox` \| `production`. |
| `currency` | `USD` | ISO 4217 code sent to Prava. `credits` is accepted as layer-neutral and interpreted as this currency. |
| `minor_unit_exponent` | `2` | **`0` for JPY/KRW.** A flag, not a constant, so zero-decimal currencies are not silently 100× wrong. |
| `unit_price_minor` | `4000` | Price returned by `quote`. |
| `cap_minor` | `100000` | Simulated-mode mandate ceiling. |
| `fail_closed` | `true` | See below. |
| `timeout_s` / `max_retries` | `30.0` / `3` | Live transport only; retries `429` and `5xx` with capped exponential backoff, never other `4xx`. |
| `authorization_code` | `None` | Merchant auth code, if any. |

`fail_closed` governs the report leg. A charge Prava **accepted** but whose
merchant-side report **failed** is an *unconfirmed* payment, and the plugin
refuses to guess which it is. With `fail_closed: true` it raises and issues no
receipt. With `fail_closed: false` it returns a receipt, `confirmed(ref)` is
`False`, and `verify_payment` reports `PENDING` until the report succeeds —
useful when a scenario wants to observe the uncertainty rather than abort on it.
Neither setting ever assumes success.

## Measured behavior

`uv run nest run scenarios/prava_commerce.yaml` — 4 agents, 2 buyer/seller pairs,
seed 42, simulated mode. 14 `prava:*` broadcasts, ticks 0→11, both pairs
identical in shape:

```
t= 0  prava:offering:seller=seller-0
t= 1  prava:quoted:buyer=buyer-0:service=svc-seller-0:price=4000:currency=USD:mode=simulated
t= 3  prava:paid:ref=prava-42-0:payer=buyer-0:payee=seller-0:amount=4000:confirmed=1:mode=simulated
t= 5  prava:replayed:ref=prava-42-0:...:amount=4000:confirmed=1:mode=simulated
t= 7  prava:verified:ref=prava-42-0:status=confirmed:mode=simulated
t= 9  prava:refused:step=overcap:ref=prava-42-0-overcap:amount=99999900:over_cap=1:code=THRESHOLD_EXCEEDED
t=11  prava:refund_unsupported:ref=prava-42-0:reason=PravaRefundUnsupportedError:mode=simulated
```

The `replayed` event re-issues the **same** `PaymentRef` at t=5. Prava returns
the original charge with `deduplicated: true`; the mandate's `spent` stays at
`"40.00"` and `chargeCount` stays at `1`. The oversized charge at t=9 is refused
by the mandate threshold, not by agent politeness — the budget is enforced by the
rail.

Test suite: **1403 passed, 1 skipped, 2 deselected**, up from a 1311-passing
baseline. Every default test is credential-free and runs against the simulated
transport. The one test that touches the real sandbox carries the `live` marker
and is deselected by the repo-wide `-m "not live"`:

```bash
PRAVA_API_KEY=sk_test_... PRAVA_MANDATE_ID=mdt_... uv run pytest -m live
```

Optional overrides for that test: `PRAVA_LIVE_AMOUNT_MINOR` (default `100`),
`PRAVA_LIVE_REFERENCE` (default `nest-prava-live-0001`).

## Limits

- **Determinism holds in simulated mode only.** Same seed → byte-identical
  trace, because the double derives identifiers from a monotonic counter. In
  `mode: live` the trace carries real transaction ids, real timestamps, and real
  network timing; **the byte-identical guarantee does not hold and is not
  claimed**. This is the honest cost of running against a real rail, and it is
  why simulated is the default.
- **The simulated transport is a test double, not a Prava emulator.** It
  reproduces exactly four documented behaviours — idempotent `reference`,
  `THRESHOLD_EXCEEDED` over cap, the `charges[]` read-back, and 404 on an unknown
  mandate. Any other path raises `NotImplementedError` rather than inventing a
  response. Never present its output as a settled transaction.
- **Trace evidence for dedupe is indirect.** The trace shows the replay returned
  the same ref and amount; that the mandate was not drawn twice is asserted in
  the test suite (`test_replayed_ref_does_not_draw_the_mandate_twice`), not
  visible in the trace itself.
- **Buyers do not share a ledger in simulated mode.** Each buyer gets its own
  `PravaPayments` instance and therefore its own double. In `live` mode all
  buyers charge the **same real mandate**, so the shared cap becomes contended —
  size `cap_minor` accordingly.
- **The scenario is deliberately small.** Two pairs, not a marketplace. A
  50×50×10 run would exhaust a real sandbox mandate long before finishing. This
  scenario proves the rail; it does not stress-test it.
- **`quote` is a local price book.** Prava has no quote endpoint. The price is
  configuration, not a network read.
- **No refund path exists upstream.** Covered above. This is a property of the
  Prava API, and if Prava adds a reversal endpoint this plugin will need a real
  implementation rather than a relaxed exception.
