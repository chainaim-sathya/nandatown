# SPDX-License-Identifier: Apache-2.0
"""Tests for the best-deal certificate emitter at 2, 3, and 4 attributes.

Runs the shipped negotiation scenarios in-process (no mocks), builds the certificate from
each trace, and checks it carries the right attribute count, verdicts, per-pair deals, and
frontier flags -- plus that it is deterministic (byte-identical when rebuilt). The emitter
imports only the public validator API and modifies nothing in ``validators.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nest_core.negotiation_certificate import certificate_json, negotiation_certificate
from nest_core.runner import ScenarioRunner
from nest_core.scenario import ScenarioConfig

# tests/ -> nest-core/ -> packages/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_DIR = REPO_ROOT / "scenarios"

_FOUR_ATTR = "multi_attribute_market_4attr_price_deadline_quantity_quality.yaml"

# scenario filename -> expected number of attributes
_CASES: dict[str, int] = {
    "chainaim_multi_attribute_market.yaml": 2,
    "multi_attribute_market_3attr_price_deadline_quantity.yaml": 3,
    _FOUR_ATTR: 4,
}

# Fine-grained (real-unit) domains coarsen ONLY the verdict-neutral reporting metrics
# (Nash/Pareto/social-welfare) so the certificate builds fast; the four pass/fail verdicts
# are always exhaustive and unaffected. Coarse/tiered domains stay exhaustive (grid_step=1).
_METRICS_GRID_STEP: dict[str, int] = {
    "multi_attribute_market_3attr_price_deadline_quantity.yaml": 10,
}


async def _run_trace(yaml_name: str, out_path: Path) -> Path:
    """Run a shipped scenario YAML in-process and return its trace path."""
    yaml_path = SCENARIOS_DIR / yaml_name
    assert yaml_path.exists(), f"scenario {yaml_name!r} not found at {yaml_path}"
    config = ScenarioConfig.from_yaml(yaml_path)
    config.output.trace = str(out_path)
    return await ScenarioRunner(config).run()


@pytest.mark.parametrize(("yaml_name", "n_attrs"), list(_CASES.items()))
@pytest.mark.asyncio
async def test_certificate_at_n_attributes(yaml_name: str, n_attrs: int, tmp_path: Path) -> None:
    """The certificate builds correctly at 2, 3, and 4 attributes.

    Example::

        await test_certificate_at_n_attributes("...4attr...yaml", 4, tmp_path)
    """
    trace = await _run_trace(yaml_name, tmp_path / "trace.jsonl")
    grid_step = _METRICS_GRID_STEP.get(yaml_name, 1)
    cert = negotiation_certificate(trace, grid_step=grid_step)

    assert cert["kind"] == "negotiation_best_deal_certificate"
    assert len(cert["attributes"]) == n_attrs, (
        f"expected {n_attrs} attributes, got {cert['attributes']}"
    )

    # The spec-literal gate and disclosure must pass at every attribute count.
    assert cert["verdicts"]["negotiation_pareto_efficient"]["passed"], cert["verdicts"][
        "negotiation_pareto_efficient"
    ]["detail"]
    assert cert["verdicts"]["negotiation_profile_disclosed"]["passed"], cert["verdicts"][
        "negotiation_profile_disclosed"
    ]["detail"]

    # Every scored pair carries a deal over exactly n_attrs axes and a frontier flag.
    assert cert["pairs"], "no agreed pairs in the certificate"
    for pair in cert["pairs"]:
        assert len(pair["deal"]) == n_attrs
        assert isinstance(pair["on_frontier"], bool)
        assert set(pair["utility"].keys()) == set(pair["parties"])
    assert cert["summary"]["pairs_scored"] == len(cert["pairs"])


@pytest.mark.asyncio
async def test_certificate_is_deterministic(tmp_path: Path) -> None:
    """Rebuilding the certificate from the same trace yields byte-identical JSON.

    Covers the additive provenance/Nash fields implicitly: they are part of the serialized
    certificate, so a byte-identical rebuild proves they are deterministic too.

    Example::

        await test_certificate_is_deterministic(tmp_path)
    """
    trace = await _run_trace(_FOUR_ATTR, tmp_path / "trace.jsonl")
    first = certificate_json(negotiation_certificate(trace))
    second = certificate_json(negotiation_certificate(trace))
    assert first == second
