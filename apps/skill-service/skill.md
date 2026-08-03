---
name: chainaim-nanda-prava-visa-payment
description: Pays a merchant on Visa rails through the Prava mandate API, and refuses the charge if the delivered item is not the one that was agreed. Use when an agent spends against a human-granted card mandate and must only pay for the item that was actually authorised.
version: "1.0"
author: ChainAIM
metadata:
  openclaw:
    requires:
      bins: [curl]
---

# ChainAIM Nanda Prava-VISA-Payment

**An agent pays a real card rail — but only for the item a human authorised.**

A buyer agent quotes an item, checks what was delivered against terms agreed
before the purchase, and only then charges a Prava mandate on Visa rails. If the
delivery does not match, the charge is refused **before any network call**, so
nothing reaches the card rail: no charge, no credential, nothing to reverse.

**Use this when** an agent must pay a merchant and you need proof it paid only
for what was authorised.

**The case this exists for is a cheaper swap.** An agent asked for a navy kurti
under INR 500 buys a maroon one at INR 450 instead. It is cheaper than the
quote, under the Visa decline threshold, under the mandate cap, and from the
right merchant — so every amount-based control in the stack lets it through.
They all ask *how much*. None asks *what*.

**Tags:** payments · visa · prava · mandate · agentic-commerce · idempotency · conformance

## Base URL

Set after deployment. Replace `<base>` below with it.

## Quick start — one call, no setup

```
curl "<base>/demo/purchase?case=drift"
```

The `display` field is a transcript anyone can read:

```
The user asked: "find one kurti under 500 rupees in green, pink or navy on libas and buy one"
The agent chose: Libas Maroon Printed Cotton Blend Straight Short Kurti (Maroon) from Libas, 45000 INR minor.
Colours that were agreed: green, pink, navy.

  OK   load_intent: loaded the agreed purchase: Libas Maroon Printed Cotton Blend Straight Short Kurti (Maroon) from Libas
  OK   quote: price is 472 USD minor units
  STOP check_delivery: the delivery does not match: colour_not_in_intent
  STOP pay: refused before any network call: the item does not match what was agreed
  ...

Result: NOT PAID. Nothing reached the card rail. Reason: colour_not_in_intent.
Mode: simulated (simulated means no real money moved).
```

Use `?case=good` for the path that settles.

## How the agent should use this

1. Call `GET /demo/purchase?case=good` or `?case=drift`.
2. Print the `display` field **verbatim**. It is already a complete transcript.
3. **Never call a `simulated` result a real transaction.** Every response carries
   a `mode` field for exactly this reason.

## Judge test

Two calls, no setup.

1. `GET /demo/purchase?case=good` — expect `"verdict":"settled"`,
   `"settled_amount_minor":523`, and a step where `check_delivery` has
   `"ok":true`.
2. `GET /demo/purchase?case=drift` — expect `"verdict":"refused"`,
   `"reason":"colour_not_in_intent"`, `"code":"INTENT_NONCONFORMANT"`,
   `"settled_amount_minor":0`, and a `failed_checks` entry with
   `"expected":"green|pink|navy"` and `"actual":"maroon"`.

Passing both proves the point in under ten seconds: it pays for the item that
was agreed, and refuses a swap that is *cheaper* than the thing it replaced.

Third call, optional: `GET /demo/purchase?case=overcap` runs with no agreed
purchase loaded and tries to charge far more than the mandate allows. The
refusal comes back from the payment rail itself.

## Endpoints

| Route | What it does |
|---|---|
| `GET /health` | liveness |
| `GET /skill.md` | this document |
| `GET /cases` | the demo cases and what each shows |
| `GET /demo/purchase?case=good\|drift\|overcap` | one full cycle, always simulated |
| `POST /purchase` | same cycle with your own settings |
| `GET /docs` | generated OpenAPI page |

Every response from a purchase route has the same shape:

| field | meaning |
|---|---|
| `verdict` | `settled` or `refused` |
| `reason` | which check decided, e.g. `conforms`, `colour_not_in_intent` |
| `code` | the plugin's error code, e.g. `INTENT_NONCONFORMANT` |
| `settled_amount_minor` | what was actually paid, in minor units |
| `steps` | every step in order, each with `ok` and a plain-English `detail` |
| `intent` | the agreed purchase: colours allowed, price, merchant, digests |
| `mode` | `simulated` or `live` |
| `display` | the printable transcript |

## Nothing here is mocked

The service calls the real plugin. `simulated` mode is a real code path in it:
an in-process transport that moves no money, mints no credentials and contacts
no network. It uses a counter instead of random ids, so the same input gives the
same output every time.

Live mode is off unless three things are all true: `ALLOW_LIVE=1` is set,
`PRAVA_API_KEY` is set, and the caller asks for it in a POST body. The demo
routes never pass a mode, so a public URL cannot charge a real card.

## What each method does

| Method | Prava call |
|---|---|
| `quote` | None. Local price list. Prava does not price anything. |
| `pay` | `POST /v1/mandates/{id}/charge`, then `POST /v1/mandates/{id}/charges/{txn}/report` |
| `verify_payment` | `GET /v1/mandates/{id}`, then find the charge by `reference` |
| `refund` | Raises. Prava has no refund endpoint. |

`PaymentRef` is sent to Prava as its `reference` field, unchanged. Prava treats
that as an idempotency key, so a repeat charge with the same reference is
refused by Prava rather than by our own code. The demo sends the same reference
twice on purpose to show this.

`verify_payment` reads Prava's own ledger, not local state. If the network fails
it returns `PENDING`, not `FAILED` — a failed call is not proof the payment did
not happen.

## The checks

Run before any network call. Each one is skipped if the term is missing.

| Check | Compares |
|---|---|
| `colour_not_in_intent` | colour is in the allowed list |
| `price_over_ceiling` | listed price is under the ceiling |
| `merchant_mismatch` | merchant matches |
| `identity_mismatch` | delivered product id matches the sku |
| `amount_mismatch` | charged amount matches the settlement amount |
| `over_decline_threshold` | amount is under the Visa decline threshold |

The first four are what catch a cheaper swap. The last two cannot.

## No refunds exist, and we do not pretend otherwise

The Prava API has no refund, credit or reversal endpoint. `refund()` raises
instead of quietly doing nothing. A reversal has to be modelled as a new forward
payment back to the buyer, with a fresh reference, recording both legs.

An approved return verdict means the request matches the terms agreed up front.
It does **not** mean money moved.

## Running it yourself

```bash
git clone <repo> && cd nandatown
pip install -r apps/skill-service/requirements.txt
python -m uvicorn app:app --app-dir apps/skill-service --port 8000
curl "localhost:8000/demo/purchase?case=drift"
```

The scenarios behind this also run in the simulator:

```bash
uv sync
uv run nest run scenarios/prava_dvp.yaml         # settles
uv run nest run scenarios/prava_dvp_drift.yaml   # refused
```

## Limits

- Same-input determinism holds in simulated mode only. Live runs are not.
- The simulated transport is a test double, not a copy of Prava. It handles four
  cases: repeat reference, over-cap refusal, the charges read-back, and 404 on an
  unknown mandate. Anything else raises rather than inventing a response.
- The drift snapshot is hand-written and marked `"constructed": true`. It is not
  a recording of a real agent run.
- `THRESHOLD_EXCEEDED` does not appear in the `good` or `drift` cases. With an
  agreed purchase loaded, an oversized charge fails the amount check first and
  never reaches the rail. That is why `case=overcap` runs without one.
- In simulated mode each buyer has its own ledger. In live mode all buyers
  charge the same real mandate, so the cap is shared.
- The demo is small on purpose. It proves the rail works; it does not stress-test it.

## The guarantee

No charge is attempted unless the delivery matches what was agreed beforehand,
so a refused purchase leaves nothing at the rail. Every charge that is attempted
carries a fixed reference as Prava's idempotency key, so a repeat is refused by
the card network, not by our own bookkeeping. And the status reported is what
Prava's ledger says, not what this process thinks.
