# SPDX-License-Identifier: Apache-2.0
"""Iteration 1 tests for the negotiation individual-rationality (IR) validator.

Covers the base **dominated-acceptance** assertion of
:func:`nest_core.validators.validate_negotiation_individual_rationality` plus the
version-tolerant ``nego:profile`` parser. Traces are hand-built event lists (no
scenario run needed), so each test pins one invariant in isolation:

* parser accepts 12- and 13-field tokens; rejects a malformed 13th field;
* IR FAILs a settlement below a disclosing party's reservation;
* IR PASSes a settlement at/above the reservation;
* IR **no-ops PASS** on a 12-field trace (the required-scenario contract), pinned
  both by example and by a Hypothesis property.

Example::

    pytest packages/nest-core/tests/test_negotiation_ir.py -v
"""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st
from nest_core.validators import validate_negotiation_individual_rationality


def _send(agent: str, to: str, msg: str) -> dict[str, Any]:
    """Build a minimal ``send`` trace event carrying ``msg``."""
    return {"kind": "send", "agent": agent, "to": to, "msg": msg}


def _profile_token(
    agent: str,
    *,
    wp: float,
    wd: float,
    pmin: int = 0,
    pmax: int = 100,
    dmin: int = 1,
    dmax: int = 30,
    dir_p: int = -1,
    dir_d: int = 1,
    rmin: float | None = None,
    sid: str = "s1",
) -> str:
    """Build a ``nego:profile`` token (13-field iff ``rmin`` is given)."""
    tok = (
        f"nego:profile:{sid}:{agent}:wp{wp}:wd{wd}:"
        f"pmin{pmin}:pmax{pmax}:dmin{dmin}:dmax{dmax}:dir_p{dir_p}:dir_d{dir_d}"
    )
    if rmin is not None:
        tok += f":rmin{rmin}"
    return tok


def _accept(sender: str, to: str, price: int, deadline: int) -> dict[str, Any]:
    """Build an accept event sealing an agreement at ``price``/``deadline``."""
    return _send(sender, to, f"nego:accept:s1:r0:p{price}:d{deadline}")


# ---------------------------------------------------------------------------
# Parser version tolerance, observed through IR behaviour (no private imports)
# ---------------------------------------------------------------------------


def test_malformed_13th_field_is_rejected_not_parsed_as_reservation() -> None:
    """A 13th field that is not an ``rmin`` token is rejected (profile dropped).

    If the malformed field were silently read as ``rmin=0.50``, the buyer's
    settlement utility (``0.20``) would fall below it and IR would FAIL. Instead the
    profile fails to parse, the pair is skipped for a missing profile, and IR PASSes
    -- proving the malformed 13th field is not mistaken for a reservation. The
    12-field (no-op) and 13-field (enforced) paths are covered by the IR tests below.
    """
    buyer = _profile_token("buyer-0", wp=1.0, wd=0.0) + ":xmin0.50"
    seller = _profile_token("seller-0", wp=0.5, wd=0.5, dir_p=1, dir_d=-1)
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _accept("buyer-0", "seller-0", price=80, deadline=10),
    ]
    (result,) = validate_negotiation_individual_rationality(events)
    assert result.passed is True
    assert "missing profile" in result.detail


# ---------------------------------------------------------------------------
# IR: dominated-acceptance assertion
# ---------------------------------------------------------------------------


def test_ir_fails_dominated_acceptance() -> None:
    """A settlement below a disclosing party's reservation FAILs IR.

    Buyer: ``wp=1.0`` over price bounds ``[0, 100]`` with ``dir_p=-1`` (cheap is
    ideal), so ``U_buyer(price=80) = (80-100)/(0-100) = 0.20`` -- below the disclosed
    floor ``rmin=0.50``.
    """
    buyer = _profile_token("buyer-0", wp=1.0, wd=0.0, rmin=0.50)
    seller = _profile_token("seller-0", wp=0.5, wd=0.5, dir_p=1, dir_d=-1)
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _accept("buyer-0", "seller-0", price=80, deadline=10),
    ]
    (result,) = validate_negotiation_individual_rationality(events)
    assert result.name == "negotiation_individually_rational"
    assert result.passed is False
    assert "below its reservation" in result.detail


def test_ir_passes_when_above_reservation() -> None:
    """The same buyer accepting a cheaper price stays above its floor and PASSes.

    ``U_buyer(price=20) = (20-100)/(0-100) = 0.80 >= rmin=0.50``.
    """
    buyer = _profile_token("buyer-0", wp=1.0, wd=0.0, rmin=0.50)
    seller = _profile_token("seller-0", wp=0.5, wd=0.5, dir_p=1, dir_d=-1)
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _accept("buyer-0", "seller-0", price=20, deadline=10),
    ]
    (result,) = validate_negotiation_individual_rationality(events)
    assert result.passed is True
    assert "1 agreement(s) individually rational" in result.detail


def test_ir_noops_on_12_field_trace() -> None:
    """With no reservation disclosed, IR PASSes regardless of the settlement.

    This is the required-scenario contract: the 12-field ``chainaim_multi_attribute_market``
    token carries no ``rmin``, so IR must never fail it.
    """
    buyer = _profile_token("buyer-0", wp=1.0, wd=0.0)  # no rmin -> 12-field
    seller = _profile_token("seller-0", wp=0.5, wd=0.5, dir_p=1, dir_d=-1)
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _accept("buyer-0", "seller-0", price=80, deadline=10),
    ]
    (result,) = validate_negotiation_individual_rationality(events)
    assert result.passed is True
    assert "without disclosed reservation" in result.detail


def test_ir_skips_pair_missing_a_profile() -> None:
    """An agreement whose pair is missing a profile is skipped (and still PASSes)."""
    buyer = _profile_token("buyer-0", wp=1.0, wd=0.0, rmin=0.50)
    events = [
        _send("buyer-0", "seller-0", buyer),  # seller never discloses
        _accept("buyer-0", "seller-0", price=80, deadline=10),
    ]
    (result,) = validate_negotiation_individual_rationality(events)
    assert result.passed is True
    assert "missing profile" in result.detail


# ---------------------------------------------------------------------------
# IR: property -- never fails a 12-field trace (no-op invariant)
# ---------------------------------------------------------------------------


@given(
    wp=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).map(
        lambda x: round(x, 2)
    ),
    price=st.integers(min_value=0, max_value=200),
    deadline=st.integers(min_value=1, max_value=60),
)
def test_ir_never_fails_a_12_field_trace(wp: float, price: int, deadline: int) -> None:
    """For any 12-field trace (no reservation disclosed), IR never FAILs.

    Locks the no-op contract across the input space, not just the example above.
    """
    wd = round(1.0 - wp, 2)
    buyer = _profile_token("buyer-0", wp=wp, wd=wd)
    seller = _profile_token("seller-0", wp=wd, wd=wp, dir_p=1, dir_d=-1)
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _accept("buyer-0", "seller-0", price=price, deadline=deadline),
    ]
    results = validate_negotiation_individual_rationality(events)
    assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# IR: unjustified-breakdown assertion (Iteration 2, above-spec)
# ---------------------------------------------------------------------------
#
# A *breakdown* is a pair that exchanged offers but never accepted. With both
# parties on symmetric 0.5/0.5 weights over the full [0,100]x[1,30] box and opposing
# directions, utilities sum to exactly 1.0 at every feasible point -- so a zone of
# possible agreement (ZOPA) exists iff the two reservations sum to <= 1.0. That
# gives clean, exact thresholds for the cases below.


def _offer(sender: str, to: str, price: int, deadline: int, *, rnd: int = 0) -> dict[str, Any]:
    """Build an offer event for ``price``/``deadline`` (no agreement sealed)."""
    return _send(sender, to, f"nego:offer:s1:r{rnd}:p{price}:d{deadline}")


def _counter(sender: str, to: str, price: int, deadline: int, *, rnd: int = 1) -> dict[str, Any]:
    """Build a counter-offer event for ``price``/``deadline`` (no agreement sealed)."""
    return _send(sender, to, f"nego:counter:s1:r{rnd}:p{price}:d{deadline}")


def _breakdown_events(
    *,
    buyer_rmin: float | None,
    seller_rmin: float | None,
) -> list[dict[str, Any]]:
    """Two disclosed profiles + an offer/counter with NO accept (a breakdown).

    Symmetric 0.5/0.5 weights over the full ``[0,100] x [1,30]`` box with opposing
    directions, so both parties' utilities sum to 1.0 at every feasible point.
    """
    buyer = _profile_token("buyer-0", wp=0.5, wd=0.5, rmin=buyer_rmin)
    seller = _profile_token("seller-0", wp=0.5, wd=0.5, dir_p=1, dir_d=-1, rmin=seller_rmin)
    return [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _offer("buyer-0", "seller-0", price=10, deadline=25),
        _counter("seller-0", "buyer-0", price=90, deadline=5),
        # no accept -> the pair breaks down
    ]


def test_ir_fails_unjustified_breakdown() -> None:
    """A breakdown FAILs when a feasible deal cleared both reservations.

    With both floors at 0.30 (sum 0.60 <= 1.0) a ZOPA exists, so walking away was
    irrational -- the above-spec breakdown assertion fires.
    """
    events = _breakdown_events(buyer_rmin=0.30, seller_rmin=0.30)
    (result,) = validate_negotiation_individual_rationality(events)
    assert result.name == "negotiation_individually_rational"
    assert result.passed is False
    assert "broke down" in result.detail
    assert "cleared both reservations" in result.detail


def test_ir_passes_legitimate_breakdown_no_zopa() -> None:
    """A breakdown PASSes when no feasible deal cleared both reservations.

    With both floors at 0.90 (sum 1.80 > 1.0) no feasible point clears both, so the
    breakdown is individually rational and the assertion PASSes.
    """
    events = _breakdown_events(buyer_rmin=0.90, seller_rmin=0.90)
    (result,) = validate_negotiation_individual_rationality(events)
    assert result.passed is True
    assert "breakdown(s) individually rational" in result.detail


def test_ir_breakdown_noops_without_both_reservations() -> None:
    """A breakdown is not judged unless BOTH parties disclosed a reservation.

    Only the buyer discloses a (high, normally-failing) floor; the seller's profile
    is 12-field, so the breakdown assertion no-ops to PASS and never flags it.
    """
    events = _breakdown_events(buyer_rmin=0.90, seller_rmin=None)
    (result,) = validate_negotiation_individual_rationality(events)
    assert result.passed is True
    assert "broke down" not in result.detail


def test_ir_breakdown_requires_surplus_flag() -> None:
    """The ``breakdown_requires_surplus`` flag tightens the clearing predicate.

    With both floors at exactly 0.50 (sum 1.0) every feasible point ties on the
    reservation line (utilities sum to 1.0). The weak default counts that tie as
    clearing -> FAIL; requiring a strict bilateral surplus finds no point strictly
    above both floors -> PASS.
    """
    events = _breakdown_events(buyer_rmin=0.50, seller_rmin=0.50)

    (weak,) = validate_negotiation_individual_rationality(events)
    assert weak.passed is False
    assert "cleared both reservations" in weak.detail

    (surplus,) = validate_negotiation_individual_rationality(
        events, breakdown_requires_surplus=True
    )
    assert surplus.passed is True
    assert "breakdown(s) individually rational" in surplus.detail


@given(
    wp=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).map(
        lambda x: round(x, 2)
    ),
    price=st.integers(min_value=0, max_value=200),
    deadline=st.integers(min_value=1, max_value=60),
)
def test_ir_never_fails_a_breakdown_without_reservations(
    wp: float, price: int, deadline: int
) -> None:
    """For any breakdown with no reservation disclosed, IR never FAILs.

    The breakdown assertion requires BOTH parties to disclose a reservation; on a
    12-field trace it must no-op to PASS across the input space.
    """
    wd = round(1.0 - wp, 2)
    buyer = _profile_token("buyer-0", wp=wp, wd=wd)  # 12-field, no rmin
    seller = _profile_token("seller-0", wp=wd, wd=wp, dir_p=1, dir_d=-1)  # 12-field
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _offer("buyer-0", "seller-0", price, deadline),
        _counter("seller-0", "buyer-0", price, deadline),
        # no accept -> breakdown
    ]
    results = validate_negotiation_individual_rationality(events)
    assert all(r.passed for r in results)
