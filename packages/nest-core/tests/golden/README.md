<!-- SPDX-License-Identifier: Apache-2.0 -->
# Negotiation golden traces (Problem-07, Iteration 0)

Committed JSONL trace(s) that lock *today's* byte-for-byte behaviour of the
Problem-07 negotiation scenarios. Consumed by
`tests/test_negotiation_golden.py::test_scenario_matches_golden`, which runs the
scenario in-process at its YAML seed (42) and asserts the fresh trace matches the
committed file **byte-for-byte**.

| Fixture | Scenario YAML | Plugin | Byte-locked? | Verdict |
|---|---|---|---|---|
| `chainaim_multi_attribute_market.jsonl` | `scenarios/chainaim_multi_attribute_market.yaml` | `chainaim_neg_multi_pareto` | **yes** | PASS (all checks) |
| *(none)* | `scenarios/multi_attribute_baseline.yaml` | `alternating_offers` | **no** (verdict-only) | FAIL (frontier check only) |

## Why only the market scenario is byte-locked

The market plugin (`chainaim_neg_multi_pareto`) assigns session ids from a
per-instance counter, so its trace serialises identically every run (the trace
writer uses `json.dumps(..., sort_keys=True)` over a deterministic `VirtualClock`,
with no wall-clock timestamps).

The baseline uses the vanilla `alternating_offers` plugin, whose `open()` mints
session ids with `uuid.uuid4()`. Those ids appear as `<sid8>` in every wire token,
so the baseline trace is **not** byte-reproducible across runs. That file is kept
byte-identical to upstream (the "don't break `alternating_offers`" constraint), so
the non-determinism is inherited and out of scope to change here. The
spec-relevant invariant for the baseline — the feasible-frontier **FAIL** — is
seed-deterministic (it depends on prices/deadlines/weights, and the validator keys
sessions by the agent pair, not by session id) and is locked by
`test_scenario_verdict` instead.

## Regenerate

Run from anywhere inside the repo (writes the byte-stable fixture(s) here):

```bash
uv run python packages/nest-core/tests/test_negotiation_golden.py --regen
```

Regenerate **only** when a trace change is intentional. Review the diff before
committing — an unexpected diff here means some earlier iteration changed behaviour
it should have preserved, which is exactly what this lock exists to catch.
