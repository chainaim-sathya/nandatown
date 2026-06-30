# SPDX-License-Identifier: Apache-2.0
"""Golden-trace regression lock for the Problem-07 negotiation scenarios.

This is **Iteration 0** of the IR / multi-attribute work: a pure-test safety net
that pins *today's* behaviour as an executable contract, so every later iteration
(base IR, unjustified-breakdown IR, ANAC metrics) can prove it changed nothing it
was not meant to. It adds tests only -- no production code is touched.

The locks are scoped to what is actually invariant:

1. **Verdict lock** (`test_scenario_verdict`, **both** scenarios) -- runs each
   scenario in-process at its YAML seed and asserts the *semantic* contract off the
   validator registry:

   * ``chainaim_multi_attribute_market``   -> every negotiation validator PASSES.
   * ``multi_attribute_baseline`` -> overall FAIL, and the failure is isolated to
     ``negotiation_frontier_efficient`` while ``negotiation_pareto_efficient`` and
     ``negotiation_profile_disclosed`` still PASS.

   That last assertion is the heart of Problem-07 (charter criterion 4): a
   symmetric price-only baseline only ever exchanges a Pareto *antichain*, so the
   exchanged-offers check is blind to it; only the feasible-frontier check, which
   reconstructs the outcome space from the disclosed profiles, catches the
   dominated settlement. The verdict depends only on prices/deadlines/weights (all
   seeded) and the validator keys sessions by the agent pair, so it is fully
   deterministic for **both** scenarios.

2. **Determinism + byte-golden locks** (`test_scenario_deterministic`,
   `test_scenario_matches_golden`) -- applied **only to byte-stable scenarios**
   (see ``_BYTE_STABLE``). They assert two same-seed runs are byte-identical, and
   that a fresh run reproduces a committed fixture under ``tests/golden/``
   byte-for-byte (skipping with a regen command when the fixture is absent).

   **Why only the market scenario is byte-locked.** The market plugin
   (``chainaim_neg_multi_pareto``) generates session ids from a per-instance
   counter, so its trace is byte-reproducible. The baseline uses the vanilla
   ``alternating_offers`` plugin, whose ``open()`` mints session ids with
   ``uuid.uuid4()``; those ids appear as ``<sid8>`` in every wire token, so the
   baseline trace is **not** byte-reproducible across runs. That file is kept
   byte-identical to upstream (the "don't break ``alternating_offers``"
   constraint), so the non-determinism is inherited and out of scope to change
   here. The spec-relevant invariant for the baseline -- the feasible-frontier
   FAIL -- is deterministic and is locked by ``test_scenario_verdict`` above.

The same in-process helper (:func:`run_scenario_trace`) powers both the tests and
the ``--regen`` entry point, so a regenerated fixture is byte-identical to what the
tests run -- they cannot drift apart by construction.

Regenerate the committed fixtures (run from anywhere inside the repo)::

    uv run python packages/nest-core/tests/test_negotiation_golden.py --regen
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from nest_core.runner import ScenarioRunner
from nest_core.scenario import ScenarioConfig
from nest_core.validators import validate_trace

# ---------------------------------------------------------------------------
# Locations (all derived from this file, so they are CWD-independent)
# ---------------------------------------------------------------------------

# tests/ -> nest-core/ -> packages/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_DIR = REPO_ROOT / "scenarios"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

#: Scenario name -> (YAML filename, expected overall PASS, expected frontier PASS).
#: The market run passes every check; the baseline fails *only* the frontier check.
SCENARIOS: dict[str, tuple[str, bool, bool]] = {
    "market": ("chainaim_multi_attribute_market.yaml", True, True),
    "baseline": ("multi_attribute_baseline.yaml", False, False),
}

#: Scenarios whose full trace is byte-reproducible across runs, and therefore
#: eligible for the determinism + byte-golden locks. The baseline is excluded
#: because the vanilla ``alternating_offers`` plugin assigns ``uuid.uuid4()``
#: session ids (surfaced as ``<sid8>`` on the wire) -- inherited, kept
#: byte-identical to upstream, and out of scope to change. The baseline is locked
#: by verdict instead (see :func:`test_scenario_verdict`).
_BYTE_STABLE: frozenset[str] = frozenset({"market"})

#: Validators that must PASS for *both* scenarios -- the baseline's settlement is
#: caught by the frontier check alone, never by these, which is the whole point.
#: IR is included here too: it no-ops PASS when no ``rmin`` reservation is disclosed,
#: which is the case for both required 12-field scenarios -- this locks the no-op
#: contract so a later IR change cannot silently start failing a required scenario.
_ALWAYS_PASS = (
    "negotiation_pareto_efficient",
    "negotiation_profile_disclosed",
    "negotiation_individually_rational",
)


# ---------------------------------------------------------------------------
# Shared in-process run helper (single source of truth for tests + --regen)
# ---------------------------------------------------------------------------


async def run_scenario_trace(
    yaml_path: Path,
    out_path: Path,
    *,
    seed: int | None = None,
) -> Path:
    """Run a scenario YAML in-process and write its JSONL trace to ``out_path``.

    The trace output path is overridden to ``out_path`` so a run never touches the
    YAML's default ``./traces`` directory. ``seed`` overrides the YAML seed when
    given (defaults to the scenario's own seed, ``42`` for Problem-07). Metrics, if
    configured, are computed after the trace is written and do not affect its bytes.

    Args:
        yaml_path: Path to the scenario YAML.
        out_path: Where to write the JSONL trace.
        seed: Optional seed override; ``None`` keeps the YAML's seed.

    Returns:
        The path the trace was written to (``out_path``).

    Example::

        path = await run_scenario_trace(
            SCENARIOS_DIR / "chainaim_multi_attribute_market.yaml", tmp_path / "m.jsonl"
        )
    """
    config = ScenarioConfig.from_yaml(yaml_path)
    if seed is not None:
        config.seed = seed
    config.output.trace = str(out_path)
    runner = ScenarioRunner(config)
    return await runner.run()


def _yaml_for(name: str) -> Path:
    """Return the scenario YAML path for ``name``, asserting it exists.

    Example::

        path = _yaml_for("market")
    """
    filename = SCENARIOS[name][0]
    path = SCENARIOS_DIR / filename
    assert path.exists(), (
        f"required scenario {filename!r} not found at {path} -- this is a mandatory "
        f"Problem-07 artifact and must not be moved or deleted"
    )
    return path


# ---------------------------------------------------------------------------
# 1. Verdict lock -- the semantic contract (both scenarios; active on a clean checkout)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.asyncio
async def test_scenario_verdict(name: str, tmp_path: Path) -> None:
    """Each scenario yields its expected negotiation verdict from the registry.

    Market passes every validator; baseline fails overall but *only* on the
    feasible-frontier check, with the exchanged-offers and disclosure checks still
    passing -- the Problem-07 thesis, locked. This holds for both scenarios because
    the verdict is seed-deterministic regardless of session-id reproducibility.

    Example::

        await test_scenario_verdict("baseline", tmp_path)
    """
    _filename, expect_overall, expect_frontier = SCENARIOS[name]
    trace = await run_scenario_trace(_yaml_for(name), tmp_path / f"{name}.jsonl")

    results = validate_trace(trace, "negotiation")
    by_name = {r.name: r for r in results}
    overall = all(r.passed for r in results)

    assert overall is expect_overall, (
        f"{name}: overall verdict {overall}, expected {expect_overall} "
        f"({[(r.name, r.passed) for r in results]})"
    )
    assert by_name["negotiation_frontier_efficient"].passed is expect_frontier, (
        f"{name}: frontier check {by_name['negotiation_frontier_efficient'].passed}, "
        f"expected {expect_frontier} -- detail: "
        f"{by_name['negotiation_frontier_efficient'].detail}"
    )
    for validator_name in _ALWAYS_PASS:
        assert by_name[validator_name].passed, (
            f"{name}: {validator_name} unexpectedly FAILED -- the baseline must be "
            f"caught by the frontier check alone. detail: {by_name[validator_name].detail}"
        )


# ---------------------------------------------------------------------------
# 2. Determinism lock -- same seed, byte-identical trace (byte-stable scenarios only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_BYTE_STABLE))
@pytest.mark.asyncio
async def test_scenario_deterministic(name: str, tmp_path: Path) -> None:
    """A byte-stable scenario produces byte-identical traces across two seeded runs.

    Only scenarios in ``_BYTE_STABLE`` are checked; the baseline is excluded because
    vanilla ``alternating_offers`` uses ``uuid.uuid4()`` session ids (see module
    docstring) and is locked by verdict instead.

    Example::

        await test_scenario_deterministic("market", tmp_path)
    """
    yaml_path = _yaml_for(name)
    first = (await run_scenario_trace(yaml_path, tmp_path / f"{name}_a.jsonl")).read_text(
        encoding="utf-8"
    )
    second = (await run_scenario_trace(yaml_path, tmp_path / f"{name}_b.jsonl")).read_text(
        encoding="utf-8"
    )
    assert first == second, f"{name}: traces diverged across two identical-seed runs"
    assert first, f"{name}: trace is empty"


# ---------------------------------------------------------------------------
# 3. Byte-golden lock -- reproduces today's committed trace exactly (byte-stable only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_BYTE_STABLE))
@pytest.mark.asyncio
async def test_scenario_matches_golden(name: str, tmp_path: Path) -> None:
    """A fresh run reproduces the committed golden trace byte-for-byte.

    Skips (rather than fails) when the fixture has not been generated yet, so a
    clean checkout is green; the skip message carries the exact regen command.

    Example::

        await test_scenario_matches_golden("market", tmp_path)
    """
    golden = GOLDEN_DIR / SCENARIOS[name][0].replace(".yaml", ".jsonl")
    if not golden.exists():
        pytest.skip(
            f"golden fixture missing: {golden}\n"
            f"generate it with: uv run python "
            f"packages/nest-core/tests/{Path(__file__).name} --regen --scenario {name}"
        )
    produced = (await run_scenario_trace(_yaml_for(name), tmp_path / f"{name}.jsonl")).read_text(
        encoding="utf-8"
    )
    assert produced == golden.read_text(encoding="utf-8"), (
        f"{name}: trace drifted from committed golden {golden.name}. If this change is "
        f"intentional, regenerate with --regen and review the diff before committing."
    )


# ---------------------------------------------------------------------------
# --regen entry point: write the committed fixtures using the SAME run path
# ---------------------------------------------------------------------------


def _regen(scenario: str, seed: int | None, out_dir: Path, scenarios_dir: Path) -> None:
    """Regenerate byte-stable golden fixtures into ``out_dir``.

    Verdict-only scenarios (not in ``_BYTE_STABLE``) have no byte-golden and are
    skipped with an explanatory note.

    Example::

        _regen("all", None, GOLDEN_DIR, SCENARIOS_DIR)
    """
    requested = sorted(_BYTE_STABLE) if scenario == "all" else [scenario]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in requested:
        if name not in _BYTE_STABLE:
            print(
                f"skip {name}: verdict-only scenario, no byte-golden -- its trace is not "
                f"byte-reproducible (inherited vanilla uuid4 session ids); locked by verdict"
            )
            continue
        filename = SCENARIOS[name][0]
        yaml_path = scenarios_dir / filename
        out_path = out_dir / SCENARIOS[name][0].replace(".yaml", ".jsonl")
        trace = asyncio.run(run_scenario_trace(yaml_path, out_path, seed=seed))
        line_count = sum(1 for _ in trace.read_text(encoding="utf-8").splitlines())
        print(f"wrote {out_path}  ({line_count} events)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate Problem-07 negotiation golden trace fixtures.",
    )
    parser.add_argument("--regen", action="store_true", help="Write the golden fixtures.")
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        default="all",
        help="Which fixture(s) to regenerate (default: all byte-stable scenarios).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the scenario seed (default: use the YAML's seed).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=GOLDEN_DIR,
        help="Directory to write fixtures into (default: tests/golden).",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=SCENARIOS_DIR,
        help="Directory holding the scenario YAMLs (default: <repo>/scenarios).",
    )
    args = parser.parse_args()
    if not args.regen:
        parser.error("nothing to do: pass --regen to write fixtures")
    _regen(args.scenario, args.seed, args.out_dir, args.scenarios_dir)
