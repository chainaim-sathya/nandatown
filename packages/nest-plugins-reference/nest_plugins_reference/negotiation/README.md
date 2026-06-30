# ChainAim Multi-Attribute Negotiation — How to Run

A negotiation plugin that bargains over **two** attributes (price + deadline) with a
private weighted utility, plus an **adversarial Pareto validator**. It ships alongside
the single-attribute `alternating_offers` baseline.

- Plugin: `chainaim_neg_multi_pareto.py` (`ChainAimMultiAttributeNegotiation`)
- Validators: `negotiation_pareto`, `negotiation_disclosure`, `negotiation_frontier`
  (in `nest_core/validators.py`, registered under `VALIDATORS["negotiation"]`)
- Scenarios: `scenarios/chainaim_multi_attribute_market.yaml` (PASS), `scenarios/multi_attribute_baseline.yaml` (FAIL)

> Run all commands from the repo root. `uv run nest …` uses the uv-managed env;
> bare `nest …` works if the package is installed / your venv is active.

## Setup

```bash
uv sync
uv run nest doctor          # verifies Python >=3.12 and that all 12 plugins resolve
uv run nest plugins list negotiation
#   -> should list BOTH `alternating_offers` and `chainaim_neg_multi_pareto`
```

## Run the scenarios (generates traces under ./traces/)

```bash
# Multi-attribute market — ChainAim plugin (PASS demo)
uv run nest run scenarios/chainaim_multi_attribute_market.yaml
#   -> Trace written to: ./traces/chainaim_multi_attribute_market.jsonl

# Single-attribute baseline — alternating_offers (adversarial FAIL demo)
uv run nest run scenarios/multi_attribute_baseline.yaml
#   -> Trace written to: ./traces/multi_attribute_baseline.jsonl
```

Both are deterministic (seed 42). Optional flags: `--seed N`, `--ticks N`, `-o/--output PATH`.

## Validate the traces

`nest validate <trace> -s <scenario>` runs that scenario's validators and **exits 0 if
all pass, 1 if any fail** (CI-gatable). Flags: `-s/--scenario` (required), `--explain`
(per-session Pareto breakdown), `--json`.

```bash
# Market: all THREE checks PASS  (exit 0)
uv run nest validate traces/chainaim_multi_attribute_market.jsonl --scenario negotiation

# Baseline: pareto PASS, disclosure PASS, frontier FAIL  (exit 1) — the intended adversarial FAIL
uv run nest validate traces/multi_attribute_baseline.jsonl --scenario negotiation
```

Expected baseline output:
```
PASS: negotiation_pareto_efficient - ...        # observed-bids check can't catch a price-only antichain
PASS: negotiation_profile_disclosed - ...
FAIL: negotiation_frontier_efficient - ... dominated by feasible p30/d1 ...
```

Add `--explain` for disclosed weights / per-offer utilities / the dominated-by line:
```bash
uv run nest validate traces/multi_attribute_baseline.jsonl --scenario negotiation --explain
uv run nest validate traces/chainaim_multi_attribute_market.jsonl   --scenario negotiation --json
```

## CI gates (must be green before tagging / PR)

`make` is not available in Git Bash — run the five directly:

```bash
uv sync
ruff check
ruff format --check
pyright
pytest -v
#   expected: ~370 passed, 1 skipped, 1 deselected, 0 failed
```

## Tests for this feature

- `packages/nest-core/tests/test_validators.py` — `TestNegotiationPareto`,
  `TestNegotiationDisclosure`, `TestNegotiationFrontier`, plus two `hypothesis`
  property tests (`TestNegotiationFrontierProperty`, `TestNegotiationParetoProperty`).

```bash
pytest packages/nest-core/tests/test_validators.py -v -k Negotiation
```

## Design docs (deeper detail)

See `DESIGN/` in the repo root:
- `negotiation-chainaim-baseline-vs-multiattr-patience-*.md` — baseline vs multi-attribute
  walkthrough, the patience analysis, and Problem-07 spec compliance.
- `negotiation-chainaim-RUNBOOK-*.md` — the full runbook (this README is the short version).

---

*Commands verified against `nest_core/cli.py` and the scenario YAMLs. The "~370 passed"
figure is from prior CI runs — confirm by running.*
