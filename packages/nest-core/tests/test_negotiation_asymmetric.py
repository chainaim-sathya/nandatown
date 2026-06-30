# SPDX-License-Identifier: Apache-2.0
"""Verdict locks for the per-role (cross-brain) negotiation scenario -- Problem-07.

Headline (verified): a single multi-attribute ChainAim agent is sufficient to reach an
efficient, feasible-frontier agreement even when paired against a myopic price-only
``alternating_offers`` opponent. The smart buyer infers the seller is deadline-heavy and
logrolls -- pinning the deadline at the seller's ideal (``d=1``) and conceding only on
price -- so every cross-brain agreement lands ON the feasible Pareto frontier
(``pareto_dist=0.000``) and all four negotiation validators pass.

The three scenarios isolate the strategy from the referee:

* ``baseline`` (stock vs stock) -> FAILS the feasible-frontier check: both ignore the
  deadline attribute, so a mutually-better deadline trade is left untaken and the
  settlement sits below the frontier. (The observed-bids check still passes: no
  *exchanged* offer dominates the settlement.)
* ``asymmetric`` (smart buyer vs stock seller) -> PASSES the frontier check: one smart
  agent recovers efficiency on its own.
* ``market`` (smart vs smart) -> passes everything.

So the differentiator for frontier-efficiency is the presence of *one* multi-attribute
brain -- exactly what the per-role ``task.config.role_plugins`` assignment makes
expressible. The observed-bids check (``negotiation_pareto_efficient``) passes in all
three because each agreement is itself on the exchanged-offer Pareto frontier.

Verdict-locked (seed-deterministic), not byte-golden: the stock seller's
``alternating_offers`` plugin mints ``uuid.uuid4()`` session ids, so the trace is not
byte-reproducible (like the baseline).

Example::

    pytest packages/nest-core/tests/test_negotiation_asymmetric.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nest_core.runner import ScenarioRunner
from nest_core.scenario import ScenarioConfig
from nest_core.validators import validate_trace

# tests/ -> nest-core/ -> packages/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_DIR = REPO_ROOT / "scenarios"

#: The exchanged-offers ("observed-bids") Pareto check.
_OBSERVED_BIDS = "negotiation_pareto_efficient"
#: The feasible-frontier check (reconstructs the outcome space from disclosed profiles).
_FRONTIER = "negotiation_frontier_efficient"


async def _verdict(scenario_filename: str, out_path: Path) -> dict[str, Any]:
    """Run a scenario YAML in-process and return ``{validator_name: result}``.

    The trace path is redirected to ``out_path`` so a run never touches the YAML's
    default ``./traces`` directory.

    Args:
        scenario_filename: File name under ``<repo>/scenarios``.
        out_path: Where to write the JSONL trace for this run.

    Returns:
        Mapping of validator name -> its result object (with ``.passed`` / ``.detail``).

    Example::

        by_name = await _verdict("multi_attribute_asymmetric.yaml", tmp_path / "a.jsonl")
    """
    yaml_path = SCENARIOS_DIR / scenario_filename
    assert yaml_path.exists(), (
        f"required scenario {scenario_filename!r} not found at {yaml_path} -- this is a "
        f"mandatory Problem-07 artifact and must not be moved or deleted"
    )
    config = ScenarioConfig.from_yaml(yaml_path)
    config.output.trace = str(out_path)
    runner = ScenarioRunner(config)
    trace = await runner.run()
    return {r.name: r for r in validate_trace(trace, "negotiation")}


@pytest.mark.asyncio
async def test_asymmetric_reaches_feasible_frontier(tmp_path: Path) -> None:
    """One smart agent is enough: the cross-brain run lands on the feasible frontier.

    The ChainAim buyer logrolls (pins the deadline at the seller's ideal, concedes only on
    price), so even against a price-only ``alternating_offers`` seller every agreement is
    feasible-frontier efficient and all four negotiation validators pass -- including the
    frontier check that the symmetric baseline fails.

    Example::

        await test_asymmetric_reaches_feasible_frontier(tmp_path)
    """
    by_name = await _verdict("multi_attribute_asymmetric.yaml", tmp_path / "asym.jsonl")
    assert by_name[_FRONTIER].passed is True, (
        f"asymmetric: {_FRONTIER} unexpectedly FAILED -- the smart buyer's logrolling should "
        f"pull every agreement onto the feasible frontier. detail: {by_name[_FRONTIER].detail}"
    )
    assert by_name[_OBSERVED_BIDS].passed is True, (
        f"asymmetric: {_OBSERVED_BIDS} unexpectedly FAILED. "
        f"detail: {by_name[_OBSERVED_BIDS].detail}"
    )
    assert all(r.passed for r in by_name.values()), (
        "asymmetric: overall verdict FAILED, expected PASS "
        f"({[(n, r.passed) for n, r in by_name.items()]})"
    )


@pytest.mark.asyncio
async def test_baseline_fails_only_the_feasible_frontier(tmp_path: Path) -> None:
    """The symmetric stock-vs-stock baseline settles BELOW the feasible frontier.

    Both agents run price-only ``alternating_offers`` and ignore the deadline attribute, so a
    mutually-better deadline trade is left untaken and the feasible-frontier check fails --
    while the observed-bids check still passes (no *exchanged* offer dominates the
    settlement). This is the contrast that isolates the smart agent's contribution in the
    asymmetric run.

    Example::

        await test_baseline_fails_only_the_feasible_frontier(tmp_path)
    """
    by_name = await _verdict("multi_attribute_baseline.yaml", tmp_path / "baseline.jsonl")
    assert by_name[_FRONTIER].passed is False, (
        f"baseline: {_FRONTIER} unexpectedly PASSED -- a price-only symmetric run must settle "
        f"below the feasible frontier. detail: {by_name[_FRONTIER].detail}"
    )
    assert by_name[_OBSERVED_BIDS].passed is True, (
        f"baseline: {_OBSERVED_BIDS} unexpectedly FAILED -- no *exchanged* offer dominates the "
        f"settlement. detail: {by_name[_OBSERVED_BIDS].detail}"
    )


@pytest.mark.asyncio
async def test_market_is_fully_efficient(tmp_path: Path) -> None:
    """Control: the symmetric smart-vs-smart market passes every negotiation validator.

    Example::

        await test_market_is_fully_efficient(tmp_path)
    """
    by_name = await _verdict("chainaim_multi_attribute_market.yaml", tmp_path / "market.jsonl")
    assert all(r.passed for r in by_name.values()), (
        "market: expected every negotiation validator to PASS "
        f"({[(n, r.passed) for n, r in by_name.items()]})"
    )
