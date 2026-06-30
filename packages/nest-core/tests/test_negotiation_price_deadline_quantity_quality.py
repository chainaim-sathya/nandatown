# SPDX-License-Identifier: Apache-2.0
"""Verdict + superset locks for the 4-attribute price+deadline+quantity+quality scenario.

This pins the committed contract of the Option-D 4-attribute demo -- a coarse-tier extension
of the shipped 3-attribute price+deadline+quantity logrolling scenario, adding a fourth
genuinely opposed attribute (``quality``) with no production-code change (the
``negotiation_multi`` factory, the wire codec, and the exhaustive feasible-frontier validator
are all already N-generic):

1. ``test_four_attr_profiles_recovered_end_to_end`` -- proves the fourth axis genuinely travels
   and that quantity and quality do NOT collide on the wire. If the 4-attr schema broke,
   ``_neg_attr_schema`` / ``_NegProfile.parse_n`` would fail to recover a pair's profiles and
   every agreed pair would be reported as ``skipped (missing profile)``. Asserting disclosure
   PASSES and no pair is skipped proves the 4-attr profile+offer format (quantity on ``n``,
   quality on ``q`` -- distinct single-char keys) round-trips end-to-end. Encoding-agnostic: it
   reads validator verdicts, not raw trace bytes.

2. ``test_four_attr_spec_gates_pass_and_frontier_audit_runs`` -- the Problem-07 spec contract
   holds at four attributes (exchanged-offers Pareto, profile disclosure, individual
   rationality all PASS), and the stricter beyond-spec feasible-frontier audit runs the
   exhaustive 4-D sweep over the agreed pairs and returns a coherent verdict (every agreement
   sits on the feasible frontier, or a dominated one is flagged with a concrete 4-axis feasible
   point). The frontier PASS/FLAG direction is tuning-dependent (coarse tiers can land the
   agreement exactly on the frontier), so it is NOT asserted rigidly -- only that the sweep
   executed over parsed 4-attr profiles.

Both tests run the REAL scenario in-process (no mocks/stubs) through the same public
``validate_trace`` path the hackathon validator uses. Like the 3-attr scenario -- and unlike
the byte-golden ``chainaim_multi_attribute_market`` -- this is a verdict lock, not a byte lock.

Runtime: every axis is a coarse ordinal tier (price/quantity <= 11 values, deadline/quality
<= 6), so the exhaustive feasible-frontier lattice is ~4356 points/pair -- exact and instant.
Widening any axis range trades runtime for resolution without changing the verdict contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nest_core.runner import ScenarioRunner
from nest_core.scenario import ScenarioConfig
from nest_core.validators import ValidationResult, validate_trace

# tests/ -> nest-core/ -> packages/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_DIR = REPO_ROOT / "scenarios"

QUALITY_YAML = SCENARIOS_DIR / "multi_attribute_market_4attr_price_deadline_quantity_quality.yaml"


async def _run_trace(yaml_path: Path, out_path: Path) -> Path:
    """Run a scenario YAML in-process and write its JSONL trace to ``out_path``.

    The trace path is overridden so a run never touches the YAML's default ``./traces``
    directory. Mirrors ``test_negotiation_price_deadline_quantity._run_trace`` (kept local so
    this file is self-contained and independent of cross-test-module import order).

    Example::

        trace = await _run_trace(QUALITY_YAML, tmp_path / "quality4.jsonl")
    """
    assert yaml_path.exists(), (
        f"shipped scenario {yaml_path.name!r} not found at {yaml_path} -- the "
        f"price+deadline+quantity+quality demo is a committed Problem-07 artifact"
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


@pytest.mark.asyncio
async def test_four_attr_profiles_recovered_end_to_end(tmp_path: Path) -> None:
    """4-attr profiles recover through the validator (quantity ``n`` / quality ``q`` distinct).

    If the fourth axis broke the wire schema, ``_neg_attr_schema`` / ``_NegProfile.parse_n``
    would fail to recover a pair's profiles and every agreed pair would be reported as
    ``skipped (missing profile)``. Asserting disclosure PASSES and that no pair is skipped for
    a missing profile proves the 4-attr profile+offer format parses end-to-end with quantity
    and quality on distinct single-char keys. Encoding-agnostic (reads verdicts, not bytes).

    Example::

        await test_four_attr_profiles_recovered_end_to_end(tmp_path)
    """
    trace = await _run_trace(QUALITY_YAML, tmp_path / "quality4.jsonl")
    by_name = _results_by_name(trace)

    assert by_name["negotiation_profile_disclosed"].passed, (
        f"4-attr profiles failed to parse/disclose: "
        f"{by_name['negotiation_profile_disclosed'].detail}"
    )
    pareto_detail = by_name["negotiation_pareto_efficient"].detail.lower()
    assert "missing profile" not in pareto_detail, (
        f"a pair was skipped for a missing profile -- the 4-attr wire schema did not round-trip "
        f"(quantity/quality may have collided): {by_name['negotiation_pareto_efficient'].detail}"
    )


@pytest.mark.asyncio
async def test_four_attr_spec_gates_pass_and_frontier_audit_runs(tmp_path: Path) -> None:
    """Spec gates PASS at 4 attributes and the exhaustive 4-D frontier audit runs coherently.

    ``negotiation_pareto_efficient`` (the Problem-07 spec-literal gate), profile disclosure,
    and individual rationality all PASS. The beyond-spec ``negotiation_frontier_efficient``
    sweep runs the exhaustive 4-D feasible-point lattice over the agreed pairs and returns a
    coherent verdict: either every agreement is certified on the feasible frontier, or a
    dominated agreement is flagged with a concrete feasible point. The PASS/FLAG direction is
    tuning-dependent (coarse tiers can land the agreement exactly on the frontier), so only the
    coherence of the sweep is asserted here.

    Example::

        await test_four_attr_spec_gates_pass_and_frontier_audit_runs(tmp_path)
    """
    trace = await _run_trace(QUALITY_YAML, tmp_path / "quality4.jsonl")
    by_name = _results_by_name(trace)

    assert by_name["negotiation_pareto_efficient"].passed, (
        f"spec gate regressed -- exchanged-offers Pareto must PASS at 4 attributes: "
        f"{by_name['negotiation_pareto_efficient'].detail}"
    )
    assert by_name["negotiation_profile_disclosed"].passed, (
        f"every bargaining agent must disclose a 4-attr profile: "
        f"{by_name['negotiation_profile_disclosed'].detail}"
    )
    assert by_name["negotiation_individually_rational"].passed, (
        f"IR must PASS (no reservation disclosed -> no-op PASS): "
        f"{by_name['negotiation_individually_rational'].detail}"
    )

    frontier = by_name["negotiation_frontier_efficient"]
    assert "missing profile" not in frontier.detail.lower(), (
        f"the 4-D frontier sweep skipped a pair for a missing profile -- 4-attr schema broke: "
        f"{frontier.detail}"
    )
    # Coherent either way: certified on the frontier (PASS), or a dominated agreement flagged
    # with a concrete feasible 4-axis point. Proves the exhaustive N-generic sweep ran over 4 axes.
    assert frontier.passed or ("dominated by feasible" in frontier.detail), (
        f"frontier audit returned an incoherent verdict: {frontier.detail}"
    )
