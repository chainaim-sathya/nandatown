# SPDX-License-Identifier: Apache-2.0
"""Tests for the Mechanism-A decision seam on the ChainAim negotiation plugin.

The plugin separates protocol mechanics from the decision *policy* via a
constructor-injected ``strategy`` (Mechanism A). These tests prove the seam is
**real** -- an injected strategy overrides the decision and its counter reaches the
wire -- and unit-test the shipped :class:`FrontierWalkStrategy` default in isolation,
which the inline policy could not be. End-to-end behaviour preservation (the default
reproduces the original policy byte-for-byte) is covered separately by the golden and
seed-range property tests.

Example::

    pytest packages/nest-plugins-reference/tests/test_negotiation_strategy_seam.py -v
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from nest_core.types import AgentId, Money, Terms
from nest_plugins_reference.negotiation.chainaim_neg_multi_pareto import (
    ChainAimMultiAttributeNegotiation,
    Decision,
    DecisionContext,
    FrontierWalkStrategy,
)


class _AcceptAllStrategy:
    """A trivial injected brain that accepts every offer."""

    def decide(self, ctx: DecisionContext) -> Decision:
        return Decision(accept=True)


class _FixedCounterStrategy:
    """A trivial injected brain that always counters with a fixed point."""

    def __init__(self, price: int, deadline: int) -> None:
        self._price = price
        self._deadline = deadline

    def decide(self, ctx: DecisionContext) -> Decision:
        return Decision(
            accept=False,
            counter_price=self._price,
            counter_deadline=self._deadline,
        )


def _u_clears_only_at_p30(p: int, d: int) -> float:
    """Own-utility stub: clears only at price 30, regardless of deadline."""
    return 1.0 if p == 30 else 0.0


def _u_opp_prefers_long_deadline(p: int, d: int) -> float:
    """Inferred-opponent stub: strictly increasing in deadline."""
    return float(d)


# --- the seam is real: an injected strategy overrides the decision --------------


async def test_default_counters_where_custom_strategy_accepts() -> None:
    """The same poor offer is countered by the default but accepted by an injected brain.

    Example::

        await test_default_counters_where_custom_strategy_accepts()
    """
    poor = Terms(price=Money(amount=100), conditions={"deadline": 30})

    default = ChainAimMultiAttributeNegotiation(AgentId("buyer-0"))
    s1 = await default.open(AgentId("seller-0"), poor)
    r1 = await default.respond(s1)
    assert r1.accepted is False  # the default frontier-walker counters a dominated deal

    injected = ChainAimMultiAttributeNegotiation(AgentId("buyer-0"), strategy=_AcceptAllStrategy())
    s2 = await injected.open(AgentId("seller-0"), poor)
    r2 = await injected.respond(s2)
    assert r2.accepted is True  # the injected brain overrides the decision


async def test_injected_counter_is_emitted_verbatim() -> None:
    """A strategy's counter point is exactly what the plugin puts on the wire.

    Example::

        await test_injected_counter_is_emitted_verbatim()
    """
    neg = ChainAimMultiAttributeNegotiation(
        AgentId("buyer-0"), strategy=_FixedCounterStrategy(42, 7)
    )
    session = await neg.open(
        AgentId("seller-0"), Terms(price=Money(amount=100), conditions={"deadline": 30})
    )
    resp = await neg.respond(session)

    assert resp.accepted is False
    assert resp.counter_terms is not None
    assert resp.counter_terms.price is not None
    assert int(resp.counter_terms.price.amount) == 42
    assert int(resp.counter_terms.conditions["deadline"]) == 7


# --- the default policy is now unit-testable in isolation -----------------------


def test_frontier_walk_accepts_when_offer_clears_aspiration() -> None:
    """The default accepts when own utility clears the aspiration level.

    Example::

        test_frontier_walk_accepts_when_offer_clears_aspiration()
    """
    ctx = DecisionContext(
        current_price=30,
        current_deadline=1,
        round=1,
        aspiration=0.5,
        u_me=_u_clears_only_at_p30,
        u_opp=_u_opp_prefers_long_deadline,
        price_grid=(30, 100),
        deadline_grid=(1, 5, 30),
        opp_offers=((30, 1),),
    )
    assert FrontierWalkStrategy().decide(ctx).accept is True


def test_frontier_walk_counters_toward_opponent_below_aspiration() -> None:
    """Below aspiration, the default counters the opponent-best aspiration-clearing point.

    Example::

        test_frontier_walk_counters_toward_opponent_below_aspiration()
    """
    ctx = DecisionContext(
        current_price=100,
        current_deadline=1,
        round=1,
        aspiration=0.5,
        u_me=_u_clears_only_at_p30,
        u_opp=_u_opp_prefers_long_deadline,
        price_grid=(30, 100),
        deadline_grid=(1, 5, 30),
        opp_offers=((100, 1),),
    )
    decision = FrontierWalkStrategy().decide(ctx)

    assert decision.accept is False
    assert decision.counter_price == 30  # only p30 clears aspiration
    assert decision.counter_deadline == 30  # among p30, the opponent prefers the longest deadline


# --- iter 5b.1: the decision seam carries an N-attribute vector ------------------


def _vector_ctx(
    attr_grids: dict[str, tuple[int, ...]],
    current_values: dict[str, int],
    aspiration: float,
    u_me_vec: Callable[[Mapping[str, int]], float],
    u_opp_vec: Callable[[Mapping[str, int]], float],
) -> DecisionContext:
    """Build a vector ``DecisionContext`` (``attr_grids`` set -> strategy vectorises).

    Example::

        ctx = _vector_ctx({"price": (30, 100)}, {"price": 100}, 0.5, um, uo)
    """
    return DecisionContext(
        current_price=int(current_values.get("price", 0)),
        current_deadline=int(current_values.get("deadline", 0)),
        round=1,
        aspiration=aspiration,
        u_me=lambda p, d: 0.0,
        u_opp=lambda p, d: 0.0,
        price_grid=attr_grids.get("price", ()),
        deadline_grid=attr_grids.get("deadline", ()),
        opp_offers=(),
        attr_names=tuple(attr_grids),
        attr_grids=attr_grids,
        current_values=current_values,
        u_me_vec=u_me_vec,
        u_opp_vec=u_opp_vec,
    )


def test_decide_over_vector_context_returns_counters_dict() -> None:
    """Over a 3-attribute grid the default returns a full ``counters`` value-dict.

    Own utility clears only at price 30; among those the inferred opponent prefers the
    longest deadline, then the highest quality -- so the chosen counter is a genuine
    3-attribute frontier point, proving every axis (incl. quality) is swept.

    Example::

        test_decide_over_vector_context_returns_counters_dict()
    """
    grids = {"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)}

    def u_me(v: Mapping[str, int]) -> float:
        return 1.0 if v["price"] == 30 else 0.0

    def u_opp(v: Mapping[str, int]) -> float:
        return v["deadline"] * 100.0 + v["quality"]

    ctx = _vector_ctx(grids, {"price": 100, "deadline": 1, "quality": 0}, 0.5, u_me, u_opp)
    decision = FrontierWalkStrategy().decide(ctx)

    assert decision.accept is False
    assert decision.counters is not None
    assert decision.counters["price"] == 30  # only p30 clears aspiration
    assert decision.counters["deadline"] == 30  # opponent prefers the longest deadline
    assert decision.counters["quality"] == 10  # then the highest quality


def test_two_attr_vector_decision_equals_scalar_fast_path() -> None:
    """At ``(price, deadline)`` the vector product sweep picks the scalar path's point.

    Proves the byte-identity proof obligation: ``itertools.product(price_grid,
    deadline_grid)`` is row-major -- the identical order to the nested loops -- so both
    paths choose the same counter on a representative offer.

    Example::

        test_two_attr_vector_decision_equals_scalar_fast_path()
    """
    price_grid = (30, 65, 100)
    deadline_grid = (1, 15, 30)

    def u_me_s(p: int, d: int) -> float:
        return 1.0 if p in (30, 65) else 0.0

    def u_opp_s(p: int, d: int) -> float:
        return float(d)  # opponent prefers a long deadline

    scalar = DecisionContext(
        current_price=100,
        current_deadline=1,
        round=1,
        aspiration=0.5,
        u_me=u_me_s,
        u_opp=u_opp_s,
        price_grid=price_grid,
        deadline_grid=deadline_grid,
        opp_offers=((100, 1),),
    )
    vector = _vector_ctx(
        {"price": price_grid, "deadline": deadline_grid},
        {"price": 100, "deadline": 1},
        0.5,
        lambda v: u_me_s(v["price"], v["deadline"]),
        lambda v: u_opp_s(v["price"], v["deadline"]),
    )

    ds = FrontierWalkStrategy().decide(scalar)
    dv = FrontierWalkStrategy().decide(vector)

    assert ds.accept is False
    assert dv.accept is False
    assert dv.counters is not None
    assert dv.counters["price"] == ds.counter_price
    assert dv.counters["deadline"] == ds.counter_deadline


async def test_respond_emits_counter_over_three_attributes() -> None:
    """A 3-attribute plugin emits a counter carrying *every* non-price attribute.

    Locks the removal of the held-at-ideal stub: ``respond`` reads price + deadline +
    quality off the wire and the counter's ``Terms.conditions`` carries both deadline
    and quality (the extra attribute genuinely varies; it is not pinned at its ideal).

    Example::

        await test_respond_emits_counter_over_three_attributes()
    """
    neg = ChainAimMultiAttributeNegotiation(
        AgentId("buyer-0"),
        weights={"price": 0.5, "deadline": 0.3, "quality": 0.2},
        bounds={"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
        direction={"price": -1, "deadline": 1, "quality": 1},
        attributes=("price", "deadline", "quality"),
    )
    poor = Terms(price=Money(amount=100), conditions={"deadline": 30, "quality": 0})
    session = await neg.open(AgentId("seller-0"), poor)
    resp = await neg.respond(session)

    assert resp.accepted is False
    assert resp.counter_terms is not None
    assert resp.counter_terms.price is not None
    assert "deadline" in resp.counter_terms.conditions
    assert "quality" in resp.counter_terms.conditions
