# SPDX-License-Identifier: Apache-2.0
"""Verdict + A/B locks for the shipped 3-attribute price+deadline+quantity scenario.

This pins the *committed* contract of the quantity logrolling demo -- the third,
genuinely-opposed attribute that distinguishes this submission from a price+deadline
or price+deadline+quality entry:

1. ``test_quantity_spec_gate_passes_with_honest_frontier_residual`` -- the shipped
   heuristic scenario satisfies every *spec-literal* negotiation gate
   (exchanged-offers Pareto, profile disclosure, individual rationality) over its 10
   agreements, while the stricter *beyond-spec* feasible-frontier audit FLAGS a small
   honest residual. The FLAG is the EXPECTED ~0.005 epsilon-frontier asymptote at
   three opposed attributes, not a broken run: the spec gate is
   ``negotiation_pareto_efficient`` (PASS), and the frontier sweep is an additional,
   stricter audit shipped on top of it.

2. ``test_bayesian_opponent_model_flags_fewer_pairs_than_heuristic`` -- the central
   novelty claim, locked as an executable invariant. Swapping ONLY the opponent
   model (concession heuristic -> deterministic Bayesian grid-posterior, Mechanism B)
   leaves the spec gate passing and drives strictly MORE pairs onto the exact feasible
   frontier, so the frontier audit flags strictly FEWER of them. Same seed, agents and
   weights -- the only independent variable is opponent inference, so the reduction is
   attributable to the Bayesian estimator alone.

Both tests run the REAL scenarios in-process (no mocks/stubs) through the same public
``validate_trace`` path the hackathon validator uses. Unlike ``chainaim_multi_attribute_market``
these scenarios are NOT byte-golden-locked: a 3-attr frontier FLAG is an expected,
non-byte-stable residual, so these are *verdict* locks, not trace-byte locks.

NOTE (runtime): the Bayesian 3-attr run is compute-heavy (a grid posterior over the
price x deadline x quantity space, per hypothesis, per offer). The A/B test runs it
once, so this file is slower than the byte-stable market locks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from nest_core.runner import ScenarioRunner
from nest_core.scenario import ScenarioConfig
from nest_core.validators import ValidationResult, validate_trace

# tests/ -> nest-core/ -> packages/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_DIR = REPO_ROOT / "scenarios"

HEURISTIC_YAML = SCENARIOS_DIR / "multi_attribute_market_3attr_price_deadline_quantity.yaml"
BAYES_YAML = SCENARIOS_DIR / "multi_attribute_market_3attr_price_deadline_quantity_bayes.yaml"

#: Per-session dominated-pair token in the frontier validator's human-readable detail,
#: e.g. "buyer-5<->seller-5: agreed ... dominated by ...". One match == one flagged pair.
_FLAGGED_PAIR_RE = re.compile(r"buyer-\d+<->seller-\d+")


async def _run_trace(yaml_path: Path, out_path: Path) -> Path:
    """Run a scenario YAML in-process and write its JSONL trace to ``out_path``.

    Mirrors ``test_negotiation_golden.run_scenario_trace`` (kept local so this file is
    self-contained and independent of cross-test-module import order). The trace path
    is overridden so a run never touches the YAML's default ``./traces`` directory.

    Example::

        trace = await _run_trace(HEURISTIC_YAML, tmp_path / "quantity.jsonl")
    """
    assert yaml_path.exists(), (
        f"shipped scenario {yaml_path.name!r} not found at {yaml_path} -- the "
        f"price+deadline+quantity demo is a committed Problem-07 artifact"
    )
    config = ScenarioConfig.from_yaml(yaml_path)
    config.output.trace = str(out_path)
    runner = ScenarioRunner(config)
    return await runner.run()


def _results_by_name(trace: Path) -> dict[str, ValidationResult]:
    """Validate a negotiation trace and index the results by validator name.

    Example::

        by_name = _results_by_name(trace)
        assert by_name["negotiation_pareto_efficient"].passed
    """
    return {r.name: r for r in validate_trace(trace, "negotiation")}


def _count_flagged_pairs(detail: str) -> int:
    """Count distinct buyer<->seller pairs the frontier audit flagged in ``detail``.

    Example::

        n = _count_flagged_pairs("buyer-5<->seller-5: ...; buyer-6<->seller-6: ...")
    """
    return len(set(_FLAGGED_PAIR_RE.findall(detail)))


@pytest.mark.asyncio
async def test_quantity_spec_gate_passes_with_honest_frontier_residual(tmp_path: Path) -> None:
    """Shipped quantity scenario: spec gates PASS, frontier audit FLAGS the residual.

    ``negotiation_pareto_efficient`` (the spec-literal gate), profile disclosure and
    individual rationality all PASS. The stricter beyond-spec
    ``negotiation_frontier_efficient`` sweep FLAGS the expected ~0.005 epsilon-frontier
    residual at three opposed attributes -- an honest, quantified asymptote, not a
    broken run.

    Example::

        await test_quantity_spec_gate_passes_with_honest_frontier_residual(tmp_path)
    """
    trace = await _run_trace(HEURISTIC_YAML, tmp_path / "quantity.jsonl")
    by_name = _results_by_name(trace)

    assert by_name["negotiation_pareto_efficient"].passed, (
        f"spec gate regressed -- exchanged-offers Pareto must PASS: "
        f"{by_name['negotiation_pareto_efficient'].detail}"
    )
    assert by_name["negotiation_profile_disclosed"].passed
    assert by_name["negotiation_individually_rational"].passed

    assert not by_name["negotiation_frontier_efficient"].passed, (
        "expected the beyond-spec frontier audit to FLAG the honest 3-attr residual; "
        "if it now PASSES the residual has closed -- update this lock intentionally"
    )


@pytest.mark.asyncio
async def test_bayesian_opponent_model_flags_fewer_pairs_than_heuristic(tmp_path: Path) -> None:
    """A/B: the Bayesian opponent model keeps the spec gate and flags strictly fewer pairs.

    The ONLY difference between the two scenarios is ``opponent_model`` (concession
    heuristic vs Bayesian grid-posterior). Both keep ``negotiation_pareto_efficient``
    passing; the Bayesian estimator drives more pairs onto the exact feasible frontier,
    so the frontier audit flags STRICTLY FEWER of them -- the submission's core novelty
    claim, locked as an invariant rather than a run-log assertion.

    Example::

        await test_bayesian_opponent_model_flags_fewer_pairs_than_heuristic(tmp_path)
    """
    heuristic_trace = await _run_trace(HEURISTIC_YAML, tmp_path / "quantity.jsonl")
    bayes_trace = await _run_trace(BAYES_YAML, tmp_path / "quantity_bayes.jsonl")

    heuristic = _results_by_name(heuristic_trace)
    bayes = _results_by_name(bayes_trace)

    # The spec-literal gate holds under both opponent models.
    assert heuristic["negotiation_pareto_efficient"].passed
    assert bayes["negotiation_pareto_efficient"].passed

    heuristic_flagged = _count_flagged_pairs(heuristic["negotiation_frontier_efficient"].detail)
    bayes_flagged = _count_flagged_pairs(bayes["negotiation_frontier_efficient"].detail)

    assert heuristic_flagged > 0, (
        "expected the heuristic run to flag a non-zero residual so the A/B is "
        "meaningful; it flagged none -- re-baseline this lock intentionally"
    )
    assert bayes_flagged < heuristic_flagged, (
        f"Bayesian opponent model did not tighten the frontier: heuristic flagged "
        f"{heuristic_flagged} pair(s), bayesian flagged {bayes_flagged} -- "
        f"expected strictly fewer"
    )
