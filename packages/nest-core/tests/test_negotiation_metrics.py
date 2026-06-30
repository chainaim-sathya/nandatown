# SPDX-License-Identifier: Apache-2.0
"""Iteration 3 tests for the ANAC negotiation outcome-quality metrics.

Covers :func:`nest_core.validators.negotiation_metrics` -- social welfare, Pareto
distance, and Nash distance -- which is **reporting only** and must never change a
validator verdict. Traces are hand-built event lists with bounds price[30,100] /
deadline[1,30] (matching the rest of the negotiation suite), so:

* buyer  (dir price -1, deadline +1): u_price=(100-p)/70, u_deadline=(d-1)/29
* seller (dir price +1, deadline -1): u_price=(p-30)/70, u_deadline=(30-d)/29

With buyer 0.70/0.30 and seller 0.30/0.70 weights, the agreement p30/d1 is the
symmetric logrolling optimum U=(0.700, 0.700): on the Pareto frontier AND the Nash
point, so both distances are 0. The price-only-style settlement p50/d15 is interior
(dominated), so both distances are strictly positive.

Example::

    pytest packages/nest-core/tests/test_negotiation_metrics.py -v
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from nest_core.validators import VALIDATORS, negotiation_metrics, validate_events


def _send(agent: str, to: str, msg: str) -> dict[str, Any]:
    return {"kind": "send", "agent": agent, "to": to, "msg": msg}


def _profile(
    agent: str,
    to: str,
    wp: float,
    wd: float,
    *,
    dir_p: int,
    dir_d: int,
    rmin: float | None = None,
) -> dict[str, Any]:
    body = (
        f"nego:profile:sess0:{agent}:wp{wp:.2f}:wd{wd:.2f}:"
        f"pmin30:pmax100:dmin1:dmax30:dir_p{dir_p:+d}:dir_d{dir_d:+d}"
    )
    if rmin is not None:
        body += f":rmin{rmin}"
    return _send(agent, to, body)


def _offer(agent: str, to: str, kind: str, rnd: int, price: int, deadline: int) -> dict[str, Any]:
    return _send(agent, to, f"nego:{kind}:sess0:r{rnd}:p{price}:d{deadline}")


def _profiles() -> list[dict[str, Any]]:
    return [
        _profile("buyer-0", "seller-0", 0.70, 0.30, dir_p=-1, dir_d=1),
        _profile("seller-0", "buyer-0", 0.30, 0.70, dir_p=1, dir_d=-1),
    ]


def test_metrics_frontier_optimal_agreement_zero_distance() -> None:
    """The logrolling optimum p30/d1 is on the frontier AND the Nash point.

    U=(0.700, 0.700): social welfare 1.40, Pareto distance ~0, Nash distance ~0,
    Nash product 0.49.
    """
    events = [
        *_profiles(),
        _offer("buyer-0", "seller-0", "offer", 0, 30, 30),
        _offer("buyer-0", "seller-0", "accept", 1, 30, 1),
    ]
    m = negotiation_metrics(events)["pairs"]["buyer-0<->seller-0"]
    assert m["agreement"] == {"price": 30, "deadline": 1}
    assert abs(m["social_welfare"] - 1.4) < 1e-3
    assert m["pareto_distance"] < 1e-6
    assert m["nash_distance"] < 1e-6
    assert abs(m["nash_product"] - 0.49) < 1e-3


def test_metrics_dominated_agreement_positive_distance() -> None:
    """The interior settlement p50/d15 sits below the frontier and off the Nash point.

    U=(0.645, 0.448): social welfare ~1.093, with strictly positive Pareto and Nash
    distances -- the numeric version of "a better deal existed".
    """
    events = [
        *_profiles(),
        _offer("buyer-0", "seller-0", "offer", 0, 30, 30),
        _offer("buyer-0", "seller-0", "accept", 1, 50, 15),
    ]
    m = negotiation_metrics(events)["pairs"]["buyer-0<->seller-0"]
    assert abs(m["social_welfare"] - 1.0926) < 1e-3
    assert m["pareto_distance"] > 0.05
    assert m["nash_distance"] > 0.05


def test_metrics_excludes_breakdowns() -> None:
    """A breakdown (offers, no accept) is counted in aggregate, not scored per-pair."""
    events = [
        *_profiles(),
        _offer("buyer-0", "seller-0", "offer", 0, 30, 30),
        _offer("seller-0", "buyer-0", "counter", 1, 90, 1),
        # no accept -> breakdown
    ]
    data = negotiation_metrics(events)
    assert data["pairs"] == {}
    assert data["aggregate"]["pairs_scored"] == 0
    assert data["aggregate"]["breakdowns"] == 1


def test_metrics_aggregate_over_two_pairs() -> None:
    """Two agreed pairs produce a populated aggregate with non-null means."""
    events = [
        *_profiles(),
        _profile("buyer-1", "seller-1", 0.70, 0.30, dir_p=-1, dir_d=1),
        _profile("seller-1", "buyer-1", 0.30, 0.70, dir_p=1, dir_d=-1),
        _offer("buyer-0", "seller-0", "offer", 0, 30, 30),
        _offer("buyer-0", "seller-0", "accept", 1, 30, 1),
        _offer("buyer-1", "seller-1", "offer", 0, 30, 30),
        _offer("buyer-1", "seller-1", "accept", 1, 50, 15),
    ]
    agg = negotiation_metrics(events)["aggregate"]
    assert agg["pairs_scored"] == 2
    assert agg["breakdowns"] == 0
    assert agg["mean_social_welfare"] is not None
    assert agg["mean_pareto_distance"] is not None
    assert agg["mean_nash_distance"] is not None


def test_metrics_is_reporting_only_no_verdict_change() -> None:
    """Computing metrics has no effect on validator verdicts, and the registry is 4."""
    events = [
        *_profiles(),
        _offer("buyer-0", "seller-0", "offer", 0, 30, 30),
        _offer("buyer-0", "seller-0", "accept", 1, 30, 1),
    ]
    before = [(r.name, r.passed) for r in validate_events(events, "negotiation")]
    _ = negotiation_metrics(events)
    after = [(r.name, r.passed) for r in validate_events(events, "negotiation")]
    assert before == after
    assert len(VALIDATORS["negotiation"]) == 4


@settings(max_examples=20, deadline=None)
@given(
    wp_b=st.integers(min_value=10, max_value=90),
    wp_s=st.integers(min_value=10, max_value=90),
    price=st.integers(min_value=30, max_value=100),
    deadline=st.integers(min_value=1, max_value=30),
)
def test_metrics_pareto_distance_nonneg_and_welfare_bounded(
    wp_b: int, wp_s: int, price: int, deadline: int
) -> None:
    """For any agreed trace: Pareto distance >= 0 and social welfare in [0, 2].

    Locks the metric ranges across the input space (utilities are bounded in [0, 1]).
    """
    wpb = round(wp_b / 100, 2)
    wdb = round(1.0 - wpb, 2)
    wps = round(wp_s / 100, 2)
    wds = round(1.0 - wps, 2)
    events = [
        _profile("buyer-0", "seller-0", wpb, wdb, dir_p=-1, dir_d=1),
        _profile("seller-0", "buyer-0", wps, wds, dir_p=1, dir_d=-1),
        _offer("buyer-0", "seller-0", "offer", 0, 30, 30),
        _offer("buyer-0", "seller-0", "accept", 1, price, deadline),
    ]
    m = negotiation_metrics(events)["pairs"]["buyer-0<->seller-0"]
    assert m["pareto_distance"] >= -1e-9
    assert -1e-9 <= m["social_welfare"] <= 2.0 + 1e-9
    assert m["nash_product"] >= -1e-9
    if m["nash_distance"] is not None:
        assert m["nash_distance"] >= -1e-9
