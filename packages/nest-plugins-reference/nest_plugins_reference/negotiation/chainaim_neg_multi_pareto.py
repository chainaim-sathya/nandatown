# SPDX-License-Identifier: Apache-2.0
"""ChainAim multi-attribute Pareto negotiation plugin.

A non-mock negotiation strategy that bargains over a configurable **attribute vector**
-- by default ``price`` (carried in :class:`~nest_core.types.Money`) and ``deadline``
(carried as an ``int`` in :attr:`~nest_core.types.Terms.conditions`), and optionally
further attributes such as ``quality`` -- using a private, weighted multi-attribute
utility. Unlike a single-attribute price haggler, ``respond`` is a
*monotonic-concession frontier walker*: when it cannot accept it counters with the
grid point that maximises the **opponent's inferred utility** while still clearing
its own aspiration level, so agreements emerge from the multi-round exchange and
trade off attributes along the Pareto frontier (logrolling).

**Mechanism A -- the decision seam.** The protocol *mechanism* (perception, offer
exchange, the ``open``/``offer``/``respond``/``close`` surface) is separated from
the decision *policy* via a constructor-injected ``strategy``. The shipped commodity
default :class:`FrontierWalkStrategy` is the policy every public scenario runs; a
private or learned brain implementing :class:`NegotiationStrategy` can be injected at
construction time **without** changing the protocol surface or the utility/disclosure
contract the validator depends on. The seam sees only the agent's own utility, an
*inferred* opponent model, and the offer grid -- never the opponent's disclosed
profile (Problem-07 anti-pattern (b)).

**Mechanism B -- the opponent-model seam.** *How* the opponent's private weights are
inferred is a second injectable seam (``opponent_model=``): the shipped
:class:`HeuristicOpponentModel` (concession-pattern inference, the scored default) or
the opt-in :class:`BayesianOpponentModel` (deterministic grid-posterior). Both expose
a two-attribute ``estimate`` *and* an N-attribute ``estimate_n``; ``respond`` routes
through whichever the agent was constructed with, at **every** attribute count -- so
the heuristic and Bayesian modes are both available for genuine 3+-attribute runs.

**N attributes (iter 5b).** The attribute count is configurable via the ``attributes``
constructor kwarg (default ``("price", "deadline")``). At the default, every internal
path delegates to the unchanged two-scalar fast path, so the shipped 2-attribute
scenarios -- and the committed golden trace -- are byte-identical. Naming more
attributes (with matching ``weights`` / ``bounds`` / ``direction`` entries) switches
``respond`` to the **vector path**: the strategy sweeps the full per-attribute grid
product and scores candidates over *real* per-attribute values (no held-at-ideal
stub), and the opponent model infers a full weight vector over the simplex.

The canonical utility (see ``_ideal_worst`` / ``_u_attr`` / ``_utility`` / ``_utility_n``)
is duplicated **byte-for-byte** in ``nest_core.validators`` because the validator
cannot import this plugin (layering). The two copies MUST stay identical.

Example::

    neg = ChainAimMultiAttributeNegotiation(
        AgentId("buyer-0"),
        weights={"price": 0.5, "deadline": 0.3, "quality": 0.2},
        bounds={"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
        direction={"price": -1, "deadline": 1, "quality": 1},
        attributes=("price", "deadline", "quality"),
        opponent_model="bayesian",
    )
    session = await neg.open(AgentId("seller-0"), Terms(price=Money(amount=30)))

Inject a custom decision brain (Mechanism A) while keeping the same plugin::

    neg = ChainAimMultiAttributeNegotiation(AgentId("buyer-0"), strategy=MyBrain())
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from nest_core.types import (
    AgentId,
    Agreement,
    Money,
    NegotiationResponse,
    NegotiationSession,
    NegotiationStatus,
    Terms,
)

# ---------------------------------------------------------------------------
# Canonical utility formula (§D.2) -- DUPLICATED VERBATIM in nest_core.validators.
# Keep these functions byte-for-byte identical across both files.
# ---------------------------------------------------------------------------


def _ideal_worst(lo: int, hi: int, direction: int) -> tuple[int, int]:
    """Return ``(ideal, worst)`` for an attribute over ``[lo, hi]`` given direction.

    ``direction >= 0`` means higher is better (``ideal=hi``); otherwise lower is
    better (``ideal=lo``).

    Example::

        ideal, worst = _ideal_worst(30, 100, -1)  # (30, 100): cheap is ideal
    """
    if direction >= 0:
        return hi, lo
    return lo, hi


def _u_attr(value: float, ideal: int, worst: int) -> float:
    """Single-attribute utility in ``[0, 1]`` (higher is better).

    Computes ``clamp((value - worst) / (ideal - worst), 0.0, 1.0)``.

    Example::

        u = _u_attr(30.0, 30, 100)  # 1.0
    """
    span = ideal - worst
    if span == 0:
        return 0.0
    u = (value - worst) / span
    if u < 0.0:
        return 0.0
    if u > 1.0:
        return 1.0
    return u


def _utility(
    price: float,
    deadline: float,
    w_price: float,
    w_deadline: float,
    price_lo: int,
    price_hi: int,
    dir_price: int,
    deadline_lo: int,
    deadline_hi: int,
    dir_deadline: int,
) -> float:
    """Weighted multi-attribute utility ``U = w_price*u_price + w_deadline*u_deadline``.

    Weights are expected to sum to ``1.0``; bounds/direction encode each party's
    ideal and worst per attribute. Generic over the two disclosed attributes.

    Example::

        u = _utility(30, 1, 0.7, 0.3, 30, 100, -1, 1, 30, 1)
    """
    ip, wp = _ideal_worst(price_lo, price_hi, dir_price)
    id_, wd = _ideal_worst(deadline_lo, deadline_hi, dir_deadline)
    return w_price * _u_attr(price, ip, wp) + w_deadline * _u_attr(deadline, id_, wd)


def _utility_n(
    values: Mapping[str, float],
    weights: Mapping[str, float],
    bounds: Mapping[str, tuple[int, int]],
    direction: Mapping[str, int],
    attrs: Sequence[str],
) -> float:
    """N-attribute generalisation of :func:`_utility` over ``attrs`` in order.

    Computes ``U = sum(weights[a] * u_attr(values[a]))`` for each attribute ``a`` in
    ``attrs``, reusing the byte-identical ``_ideal_worst`` / ``_u_attr`` primitives.
    For ``attrs == ("price", "deadline")`` this returns exactly what :func:`_utility`
    returns, so the two-attribute default path is unchanged; additional attributes
    simply add more weighted terms. This mirrors ``_party_utility`` in
    ``nest_core.validators`` (same formula, different call shape).

    Example::

        u = _utility_n(
            {"price": 30, "deadline": 1, "quality": 9},
            {"price": 0.5, "deadline": 0.3, "quality": 0.2},
            {"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
            {"price": -1, "deadline": 1, "quality": 1},
            ("price", "deadline", "quality"),
        )
    """
    total = 0.0
    for a in attrs:
        lo, hi = bounds[a]
        ideal, worst = _ideal_worst(lo, hi, direction[a])
        total += weights[a] * _u_attr(values[a], ideal, worst)
    return total


# ---------------------------------------------------------------------------
# Mechanism A -- the decision seam (policy, separable from the mechanism)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A strategy's verdict for one ``respond`` turn: accept, or counter.

    ``accept=True`` accepts the current offer; otherwise the counter is carried in
    **one** of two mutually exclusive forms:

    - the two-attribute fast path uses ``counter_price`` / ``counter_deadline`` (each
      falling back to the current offer value when ``None``);
    - the N-attribute path uses ``counters`` -- a full value-dict keyed by attribute
      name. When ``counters`` is set it is **authoritative** and the scalar fields are
      ignored.

    Exactly one form is populated per construction; the default ``("price",
    "deadline")`` plugin path always uses the scalar form so the golden is unchanged.

    Example::

        Decision(accept=False, counter_price=30, counter_deadline=1)
        Decision(accept=False, counters={"price": 30, "deadline": 1, "quality": 9})
    """

    accept: bool
    counter_price: int | None = None
    counter_deadline: int | None = None
    counters: Mapping[str, int] | None = None


@dataclass(frozen=True)
class DecisionContext:
    """Everything a strategy needs to decide one turn -- the agent's own view only.

    Every field is derived from the agent's *own* utility and the *observed* offers.
    ``u_opp`` is the agent's **inferred** model of the opponent (from concession
    patterns), never the opponent's disclosed profile -- so a strategy cannot peek
    at the opponent's true utility (Problem-07 anti-pattern (b)).

    Example::

        ctx = DecisionContext(
            current_price=50, current_deadline=15, round=2, aspiration=0.81,
            u_me=my_utility, u_opp=inferred_opp_utility,
            price_grid=(30, 65, 100), deadline_grid=(1, 15, 30),
            opp_offers=((95, 15), (90, 15)),
        )
    """

    current_price: int
    current_deadline: int
    round: int
    aspiration: float
    u_me: Callable[[int, int], float]
    u_opp: Callable[[int, int], float]
    price_grid: tuple[int, ...]
    deadline_grid: tuple[int, ...]
    opp_offers: tuple[tuple[int, int], ...]
    # N-attribute vector view (iter 5b.1). Populated only when negotiating more than
    # ``(price, deadline)``; the two-attribute fast path leaves these unset (``None`` /
    # empty) so the scalar fields above remain the sole inputs and the golden trace
    # stays byte-identical. When ``attr_grids`` is set the strategy takes the vector
    # path: it sweeps ``itertools.product`` of ``attr_grids`` ordered by ``attr_names``
    # and scores candidates via ``u_me_vec`` / ``u_opp_vec`` over the full value-dict.
    attr_names: tuple[str, ...] = ()
    attr_grids: Mapping[str, tuple[int, ...]] | None = None
    current_values: Mapping[str, int] | None = None
    u_me_vec: Callable[[Mapping[str, int]], float] | None = None
    u_opp_vec: Callable[[Mapping[str, int]], float] | None = None
    opp_offers_vec: tuple[Mapping[str, int], ...] = ()


class NegotiationStrategy(Protocol):
    """The injectable decision-policy seam (Mechanism A).

    Any object with a matching ``decide`` is a strategy (structural typing), so a
    private engine can satisfy it while importing only these context types -- never
    the plugin internals.

    Example::

        class AcceptAll:
            def decide(self, ctx: DecisionContext) -> Decision:
                return Decision(accept=True)
    """

    def decide(self, ctx: DecisionContext) -> Decision:
        """Return the accept/counter decision for one ``respond`` turn."""
        ...


class FrontierWalkStrategy:
    """The shipped commodity default: a deterministic monotonic-concession walker.

    Accepts when the current offer clears the decaying aspiration; otherwise counters
    with the grid point that maximises the *inferred* opponent utility while still
    clearing the agent's own aspiration (ties broken by own utility), falling back to
    the agent's own best point when nothing clears. This is the exact policy
    Problem-07's market scenario runs -- logrolling toward the feasible Pareto
    frontier -- and reproduces the plugin's original inline behaviour byte-for-byte at
    two attributes, and generalises to N via :meth:`_decide_vector`.

    Example::

        decision = FrontierWalkStrategy().decide(ctx)
    """

    def decide(self, ctx: DecisionContext) -> Decision:
        """Accept on aspiration, else counter on the inferred-opponent frontier.

        Example::

            d = FrontierWalkStrategy().decide(ctx)
        """
        if ctx.attr_grids is not None:
            return self._decide_vector(ctx)

        if ctx.u_me(ctx.current_price, ctx.current_deadline) + 1e-12 >= ctx.aspiration:
            return Decision(accept=True)

        best: tuple[int, int] | None = None
        best_key: tuple[float, float] = (-1.0, -1.0)
        # Pass 1: candidates that clear my aspiration -> maximise opponent utility.
        for p in ctx.price_grid:
            for dl in ctx.deadline_grid:
                if ctx.u_me(p, dl) + 1e-12 < ctx.aspiration:
                    continue
                key = (ctx.u_opp(p, dl), ctx.u_me(p, dl))
                if key > best_key:
                    best_key = key
                    best = (p, dl)
        if best is None:
            # Pass 2 (early rounds, aspiration unreachable): concede to my own best.
            best_self = -1.0
            for p in ctx.price_grid:
                for dl in ctx.deadline_grid:
                    um = ctx.u_me(p, dl)
                    if um > best_self:
                        best_self = um
                        best = (p, dl)

        if best is None:
            return Decision(
                accept=False,
                counter_price=ctx.current_price,
                counter_deadline=ctx.current_deadline,
            )
        return Decision(accept=False, counter_price=best[0], counter_deadline=best[1])

    def _decide_vector(self, ctx: DecisionContext) -> Decision:
        """N-attribute generalisation of :meth:`decide` over the attribute grids.

        Sweeps ``itertools.product`` of ``attr_grids`` ordered by ``attr_names`` and
        scores each candidate value-dict with the vector utilities. The structure is
        identical to :meth:`decide`: accept on aspiration; Pass-1 maximises the
        inferred opponent's utility among points that clear the agent's aspiration
        (ties broken by own utility); Pass-2 concedes to the agent's own best when
        nothing clears.

        **Byte-identity proof obligation.** At ``attrs == ("price", "deadline")`` the
        product ``itertools.product(price_grid, deadline_grid)`` is row-major (price
        outer, deadline inner) -- the *identical* iteration order to the two nested
        loops in :meth:`decide` -- and the tie-break key ``(u_opp, u_me)`` with a
        strict ``>`` keeps the first maximiser, so the chosen point matches the scalar
        fast path exactly -- locked by the 2-attr decision-equivalence micro-test.

        Example::

            d = FrontierWalkStrategy()._decide_vector(vector_ctx)
        """
        u_me_vec = ctx.u_me_vec
        u_opp_vec = ctx.u_opp_vec
        cur = ctx.current_values
        grids = ctx.attr_grids
        if u_me_vec is None or u_opp_vec is None or cur is None or grids is None:
            # Defensive: a vector context must carry all vector inputs. Fall back to a
            # no-op counter rather than crash (never reached on the supported paths).
            return Decision(accept=False, counters=dict(cur) if cur is not None else None)

        attrs = ctx.attr_names
        if u_me_vec(dict(cur)) + 1e-12 >= ctx.aspiration:
            return Decision(accept=True)

        axes = [grids[a] for a in attrs]
        best: dict[str, int] | None = None
        best_key: tuple[float, float] = (-1.0, -1.0)
        # Pass 1: candidates that clear my aspiration -> maximise opponent utility.
        for combo in itertools.product(*axes):
            vals = {a: combo[i] for i, a in enumerate(attrs)}
            if u_me_vec(vals) + 1e-12 < ctx.aspiration:
                continue
            key = (u_opp_vec(vals), u_me_vec(vals))
            if key > best_key:
                best_key = key
                best = vals
        if best is None:
            # Pass 2 (early rounds, aspiration unreachable): concede to my own best.
            best_self = -1.0
            for combo in itertools.product(*axes):
                vals = {a: combo[i] for i, a in enumerate(attrs)}
                um = u_me_vec(vals)
                if um > best_self:
                    best_self = um
                    best = vals

        if best is None:
            return Decision(accept=False, counters=dict(cur))
        return Decision(accept=False, counters=best)


# ---------------------------------------------------------------------------
# Mechanism B -- the opponent-model seam (estimation, separable from the policy)
# ---------------------------------------------------------------------------


def _axis_grid(lo: int, hi: int, points: int) -> list[int]:
    """Deterministic inclusive integer grid over ``[lo, hi]`` with ~``points`` steps.

    Mirrors the plugin's own ``_grid`` so the Bayesian partition function ranges over
    the same offer space the agent could propose.

    Example::

        _axis_grid(30, 100, 8)  # [30, 40, 50, 60, 70, 80, 90, 100]
    """
    if hi <= lo:
        return [lo]
    step = max(1, (hi - lo) // (max(2, points) - 1))
    values = list(range(lo, hi + 1, step))
    if values[-1] != hi:
        values.append(hi)
    return values


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically stable ``log(sum(exp(v)))`` using stdlib ``math`` only.

    Example::

        _logsumexp([0.0, 0.0])  # ~0.6931 (log 2)
    """
    m = max(values)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def _simplex_grid(n: int, m: int) -> list[tuple[float, ...]]:
    """All weight vectors on the ``(n-1)``-simplex at integer resolution ``m``.

    Enumerates every composition ``(c_1, ..., c_n)`` of ``m`` into ``n`` non-negative
    integer parts and returns ``(c_1/m, ..., c_n/m)`` -- a deterministic, evenly spaced
    grid of weight vectors that each sum to ``1.0``. The count is ``C(m+n-1, n-1)``.

    For ``n == 2`` this reproduces the 1-D price-weight grid the two-attribute
    :class:`BayesianOpponentModel` uses: ``(i/m, 1 - i/m)`` for ``i`` in ``0..m``
    (``m+1 == grid_points`` points), so the Bayesian generalisation reduces exactly to
    the original at two attributes.

    Example::

        _simplex_grid(2, 2)  # [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        _simplex_grid(3, 2)  # 6 vectors over the price/deadline/quality simplex
    """
    if m <= 0:
        return [tuple(1.0 / n for _ in range(n))]

    def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
        if parts == 1:
            return [(total,)]
        out: list[tuple[int, ...]] = []
        for first in range(total + 1):
            for rest in _compositions(total - first, parts - 1):
                out.append((first, *rest))
        return out

    return [tuple(c / m for c in comp) for comp in _compositions(m, n)]


def _heuristic_opp_weights(
    offers: Sequence[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[float, float]:
    """Heuristic opponent ``(w_price, w_deadline)`` from a concession pattern.

    The attribute the opponent concedes *more* on (moves further toward this agent's
    ideal) is the one they value *less*. Returns equal weights until at least two
    opponent offers have been observed. This is the byte-identical extraction of the
    plugin's original inline ``_estimate_opp_weights``.

    Example::

        _heuristic_opp_weights(((100, 1), (30, 1)), (30, 100, 1, 30), (-1, 1))
    """
    if len(offers) < 2:
        return (0.5, 0.5)
    price_lo, price_hi, deadline_lo, deadline_hi = bounds
    dir_price, dir_deadline = direction
    first_p, first_d = offers[0]
    last_p, last_d = offers[-1]
    ip, wp_b = _ideal_worst(price_lo, price_hi, dir_price)
    id_, wd_b = _ideal_worst(deadline_lo, deadline_hi, dir_deadline)
    c_price = max(0.0, _u_attr(last_p, ip, wp_b) - _u_attr(first_p, ip, wp_b))
    c_deadline = max(0.0, _u_attr(last_d, id_, wd_b) - _u_attr(first_d, id_, wd_b))
    inv_p = max(1e-3, 1.0 - c_price)
    inv_d = max(1e-3, 1.0 - c_deadline)
    total = inv_p + inv_d
    return (inv_p / total, inv_d / total)


def _heuristic_opp_weights_n(
    offers: Sequence[Mapping[str, int]],
    attrs: Sequence[str],
    bounds: Mapping[str, tuple[int, int]],
    direction: Mapping[str, int],
) -> dict[str, float]:
    """N-attribute generalisation of :func:`_heuristic_opp_weights`.

    Applies the same rule term-by-term over ``attrs``: the attribute the opponent
    concedes *more* on (the larger gain in *this* agent's per-attribute utility
    between the opponent's first and latest observed offer) is valued *less*, so its
    inferred weight is ``max(1e-3, 1 - concession)``; the inverse weights are then
    normalised to sum to ``1.0``. Returns equal weights until at least two opponent
    offers have been observed. For ``attrs == ("price", "deadline")`` the two outputs
    are numerically identical to :func:`_heuristic_opp_weights` (same formula).

    Example::

        _heuristic_opp_weights_n(
            ({"price": 100, "deadline": 30, "quality": 0},
             {"price": 30, "deadline": 30, "quality": 0}),
            ("price", "deadline", "quality"),
            {"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
            {"price": -1, "deadline": 1, "quality": 1},
        )
    """
    attrs = tuple(attrs)
    n = len(attrs)
    if len(offers) < 2:
        return {a: 1.0 / n for a in attrs}
    first = offers[0]
    last = offers[-1]
    inv: dict[str, float] = {}
    for a in attrs:
        lo, hi = bounds[a]
        ideal, worst = _ideal_worst(lo, hi, direction[a])
        concession = max(0.0, _u_attr(last[a], ideal, worst) - _u_attr(first[a], ideal, worst))
        inv[a] = max(1e-3, 1.0 - concession)
    total = sum(inv.values())
    return {a: inv[a] / total for a in attrs}


class OpponentModel(Protocol):
    """The injectable opponent-weight estimator seam (Mechanism B).

    Any object with matching ``estimate`` / ``estimate_n`` is an opponent model
    (structural typing). It receives only the *observed* offers plus this agent's own
    bounds/direction -- never the opponent's disclosed profile (Problem-07 anti-pattern
    (b)) -- and returns inferred opponent weights summing to ``1.0``. ``estimate`` is
    the two-attribute ``(w_price, w_deadline)`` form (used by the byte-identical
    two-attribute fast path); ``estimate_n`` is the N-attribute form returning a weight
    dict keyed by attribute name (used by the vector path).

    Example::

        class AlwaysEqual:
            def estimate(self, opp_offers, bounds, direction):
                return (0.5, 0.5)

            def estimate_n(self, opp_offers, attrs, bounds, direction):
                return {a: 1.0 / len(attrs) for a in attrs}
    """

    def estimate(
        self,
        opp_offers: Sequence[tuple[int, int]],
        bounds: tuple[int, int, int, int],
        direction: tuple[int, int],
    ) -> tuple[float, float]:
        """Return inferred opponent ``(w_price, w_deadline)`` from observed offers."""
        ...

    def estimate_n(
        self,
        opp_offers: Sequence[Mapping[str, int]],
        attrs: Sequence[str],
        bounds: Mapping[str, tuple[int, int]],
        direction: Mapping[str, int],
    ) -> dict[str, float]:
        """Return inferred opponent per-attribute weights (keyed by name, sum 1.0)."""
        ...


class HeuristicOpponentModel:
    """Shipped commodity default: concession-pattern opponent-weight inference.

    Stateless. ``estimate`` delegates to :func:`_heuristic_opp_weights` (reproducing the
    plugin's original inline inference byte-for-byte, so the golden trace is unchanged
    when this default is used at two attributes); ``estimate_n`` delegates to
    :func:`_heuristic_opp_weights_n` for the N-attribute vector path.

    Example::

        HeuristicOpponentModel().estimate(((100, 1), (30, 1)), (30, 100, 1, 30), (-1, 1))
    """

    def estimate(
        self,
        opp_offers: Sequence[tuple[int, int]],
        bounds: tuple[int, int, int, int],
        direction: tuple[int, int],
    ) -> tuple[float, float]:
        """Infer opponent ``(w_price, w_deadline)`` from the concession pattern.

        Example::

            HeuristicOpponentModel().estimate(((100, 1), (30, 1)), (30, 100, 1, 30), (-1, 1))
        """
        return _heuristic_opp_weights(opp_offers, bounds, direction)

    def estimate_n(
        self,
        opp_offers: Sequence[Mapping[str, int]],
        attrs: Sequence[str],
        bounds: Mapping[str, tuple[int, int]],
        direction: Mapping[str, int],
    ) -> dict[str, float]:
        """Infer per-attribute opponent weights from the concession pattern.

        Example::

            HeuristicOpponentModel().estimate_n(
                ({"price": 100, "deadline": 30, "quality": 0},
                 {"price": 30, "deadline": 30, "quality": 0}),
                ("price", "deadline", "quality"),
                {"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
                {"price": -1, "deadline": 1, "quality": 1},
            )
        """
        return _heuristic_opp_weights_n(opp_offers, attrs, bounds, direction)


class BayesianOpponentModel:
    """Deterministic grid-posterior estimator of the opponent's weights (opt-in).

    Models the opponent as a soft-rational (Boltzmann) utility maximiser and infers
    their weight vector by exact Bayesian update over a fixed hypothesis grid. Pure
    stdlib ``math`` -- no numpy/scipy, no sampling, no RNG, no wall-clock -- so the same
    offer sequence always yields the same estimate (byte-deterministic,
    charter-compliant). Stateless: it reads only the offers passed in, so it never
    learns across sessions.

    ``estimate`` infers the two-attribute price-weight ``w`` over a 1-D ``[0, 1]`` grid
    (``w_deadline = 1 - w``); ``estimate_n`` infers the full weight vector over the
    ``(n-1)``-simplex grid (:func:`_simplex_grid`). At two attributes the simplex grid
    reduces to the 1-D grid, so the two forms agree.

    Args:
        grid_points: number ``K`` of weight hypotheses per axis (default 21); the
            simplex resolution is ``m = K - 1``.
        beta: Boltzmann inverse-temperature / assumed opponent rationality (default 4.0).
        warmup: offers to observe before updating; ``0`` updates from the first.
        offer_grid_points: resolution of each attribute's grid for the partition
            function (default 15).

    Example::

        BayesianOpponentModel(beta=4.0).estimate(((100, 30),), (30, 100, 1, 30), (-1, 1))
    """

    def __init__(
        self,
        grid_points: int = 21,
        beta: float = 4.0,
        warmup: int = 0,
        offer_grid_points: int = 15,
    ) -> None:
        self._k = max(2, int(grid_points))
        self._beta = float(beta)
        self._warmup = max(0, int(warmup))
        self._offer_grid_points = max(2, int(offer_grid_points))

    def estimate(
        self,
        opp_offers: Sequence[tuple[int, int]],
        bounds: tuple[int, int, int, int],
        direction: tuple[int, int],
    ) -> tuple[float, float]:
        """Posterior-mean opponent ``(w_price, w_deadline)`` from observed offers.

        Returns the prior mean ``(0.5, 0.5)`` until ``max(1, warmup)`` offers are seen.

        Example::

            BayesianOpponentModel().estimate(((100, 30),), (30, 100, 1, 30), (-1, 1))
        """
        offers = list(opp_offers)
        if len(offers) < max(1, self._warmup):
            return (0.5, 0.5)

        price_lo, price_hi, deadline_lo, deadline_hi = bounds
        dir_price, dir_deadline = direction
        # Opponent's view: this agent's bounds with mirrored direction (as in ``_u_opp``).
        ip, wp = _ideal_worst(price_lo, price_hi, -dir_price)
        idl, wdl = _ideal_worst(deadline_lo, deadline_hi, -dir_deadline)

        hyps = [i / (self._k - 1) for i in range(self._k)]
        pgrid = _axis_grid(price_lo, price_hi, self._offer_grid_points)
        dgrid = _axis_grid(deadline_lo, deadline_hi, self._offer_grid_points)

        def u_opp(h: float, p: int, d: int) -> float:
            return h * _u_attr(p, ip, wp) + (1.0 - h) * _u_attr(d, idl, wdl)

        # Unnormalised log-posterior per hypothesis (uniform prior is a dropped constant).
        log_post: list[float] = []
        for h in hyps:
            log_z = _logsumexp([self._beta * u_opp(h, p, d) for p in pgrid for d in dgrid])
            ll = 0.0
            for p, d in offers:
                ll += self._beta * u_opp(h, p, d) - log_z
            log_post.append(ll)

        norm = _logsumexp(log_post)
        post = [math.exp(lp - norm) for lp in log_post]
        w_price = sum(h * pw for h, pw in zip(hyps, post, strict=True))
        return (w_price, 1.0 - w_price)

    def estimate_n(
        self,
        opp_offers: Sequence[Mapping[str, int]],
        attrs: Sequence[str],
        bounds: Mapping[str, tuple[int, int]],
        direction: Mapping[str, int],
    ) -> dict[str, float]:
        """Posterior-mean opponent weight vector over the ``(n-1)``-simplex.

        Grids the simplex of opponent weight vectors at resolution ``m = grid_points-1``
        (:func:`_simplex_grid`), scores each hypothesis by the Boltzmann log-likelihood
        of the observed offers (partition function over the per-attribute offer grids),
        and returns the posterior-mean weight vector keyed by attribute name. The
        opponent is modelled as sharing the market bounds with each attribute's
        direction mirrored. Returns equal weights until ``max(1, warmup)`` offers are
        seen. At two attributes this reduces to :meth:`estimate` (same grid, same
        likelihood), so the generalisation is faithful.

        Example::

            BayesianOpponentModel().estimate_n(
                ({"price": 95, "deadline": 30, "quality": 0},),
                ("price", "deadline", "quality"),
                {"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
                {"price": -1, "deadline": 1, "quality": 1},
            )
        """
        attrs = tuple(attrs)
        n = len(attrs)
        offers = list(opp_offers)
        if len(offers) < max(1, self._warmup):
            return {a: 1.0 / n for a in attrs}

        # Opponent's view: market bounds, mirrored direction (as in ``_u_opp_vec``).
        iw: dict[str, tuple[int, int]] = {}
        grids: dict[str, list[int]] = {}
        for a in attrs:
            lo, hi = bounds[a]
            iw[a] = _ideal_worst(lo, hi, -direction[a])
            grids[a] = _axis_grid(lo, hi, self._offer_grid_points)

        hyps = _simplex_grid(n, self._k - 1)
        candidates = [
            {attrs[i]: combo[i] for i in range(n)}
            for combo in itertools.product(*(grids[a] for a in attrs))
        ]

        def u_opp(wvec: tuple[float, ...], vals: Mapping[str, int]) -> float:
            return sum(
                wvec[i] * _u_attr(vals[attrs[i]], iw[attrs[i]][0], iw[attrs[i]][1])
                for i in range(n)
            )

        # Unnormalised log-posterior per weight-vector hypothesis (uniform prior dropped).
        log_post: list[float] = []
        for wvec in hyps:
            log_z = _logsumexp([self._beta * u_opp(wvec, c) for c in candidates])
            ll = 0.0
            for off in offers:
                ll += self._beta * u_opp(wvec, off) - log_z
            log_post.append(ll)

        norm = _logsumexp(log_post)
        post = [math.exp(lp - norm) for lp in log_post]
        mean = [0.0] * n
        for pw, wvec in zip(post, hyps, strict=True):
            for i in range(n):
                mean[i] += pw * wvec[i]
        return {attrs[i]: mean[i] for i in range(n)}


def _resolve_opponent_model(
    spec: OpponentModel | Literal["heuristic", "bayesian"],
    *,
    grid_points: int,
    beta: float,
    warmup: int,
) -> OpponentModel:
    """Resolve a string name or an injected object to a concrete :class:`OpponentModel`.

    Example::

        _resolve_opponent_model("bayesian", grid_points=21, beta=4.0, warmup=0)
    """
    if isinstance(spec, str):
        if spec == "heuristic":
            return HeuristicOpponentModel()
        if spec == "bayesian":
            return BayesianOpponentModel(grid_points=grid_points, beta=beta, warmup=warmup)
        msg = f"unknown opponent_model {spec!r}; expected 'heuristic' or 'bayesian'"
        raise ValueError(msg)
    return spec


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

_DEFAULT_BOUNDS: dict[str, tuple[int, int]] = {"price": (1, 100), "deadline": (1, 30)}
_DEFAULT_WEIGHTS: dict[str, float] = {"price": 0.5, "deadline": 0.5}
_DEFAULT_DIRECTION: dict[str, int] = {"price": -1, "deadline": 1}


class ChainAimMultiAttributeNegotiation:
    """N-attribute Pareto-frontier-walking negotiation strategy.

    Each agent owns a private utility (its own ``weights``); ``respond`` accepts when
    the opponent's current offer clears a decaying aspiration, otherwise it counters
    via the injected ``strategy`` (default :class:`FrontierWalkStrategy`) -- the grid
    point that maximises the opponent's *inferred* utility subject to its own
    aspiration, a deterministic monotonic concession.

    Two seams are swappable. **Mechanism A** (``strategy=``) is the decision policy;
    **Mechanism B** (``opponent_model=``) is how the opponent's weights are inferred
    (``"heuristic"`` default, scored; ``"bayesian"`` opt-in). Everything else
    (perception, the wire protocol, the utility/disclosure contract) is fixed.

    Attribute count is configurable via the ``attributes`` constructor kwarg, which
    defaults to ``("price", "deadline")``. With the default, every internal path uses
    the unchanged two-scalar fast path, so the shipped scenarios and the golden trace
    are byte-identical; naming more attributes (with matching ``weights`` / ``bounds``
    / ``direction`` keys) switches ``respond`` to the vector path, scoring candidates
    over real per-attribute values (no held-at-ideal stub) and inferring a full
    opponent weight vector.

    Example::

        neg = ChainAimMultiAttributeNegotiation(AgentId("a1"))
        session = await neg.open(AgentId("a2"), Terms(price=Money(amount=50)))
    """

    def __init__(
        self,
        agent_id: AgentId,
        weights: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[int, int]] | None = None,
        direction: Mapping[str, int] | None = None,
        patience: float = 0.9,
        reservation: float = 0.0,
        grid_points: int = 15,
        strategy: NegotiationStrategy | None = None,
        attributes: Sequence[str] = ("price", "deadline"),
        opponent_model: OpponentModel | Literal["heuristic", "bayesian"] = "heuristic",
        bayes_grid_points: int = 21,
        bayes_beta: float = 4.0,
        bayes_warmup: int = 0,
    ) -> None:
        self._agent_id = agent_id
        w = dict(_DEFAULT_WEIGHTS if weights is None else weights)
        b = dict(_DEFAULT_BOUNDS if bounds is None else bounds)
        d = dict(_DEFAULT_DIRECTION if direction is None else direction)
        # N-attribute representation (iter 5a/5b). ``attributes`` is the canonical,
        # ordered list the agent negotiates over; it defaults to the two-attribute
        # ``(price, deadline)`` case, for which every code path below delegates to the
        # unchanged two-scalar fast-path so the golden trace stays byte-identical.
        self._attrs: tuple[str, ...] = tuple(attributes)
        if not self._attrs:
            msg = "attributes must name at least one attribute"
            raise ValueError(msg)
        self._w: dict[str, float] = {a: float(w[a]) for a in self._attrs}
        self._bounds: dict[str, tuple[int, int]] = {
            a: (int(b[a][0]), int(b[a][1])) for a in self._attrs
        }
        self._dir: dict[str, int] = {a: int(d[a]) for a in self._attrs}
        self._is_two_attr = self._attrs == ("price", "deadline")
        # Two-attribute scalar fields (unchanged) -- the fast path and the public
        # disclosure properties read these directly.
        self._w_price = float(w["price"])
        self._w_deadline = float(w["deadline"])
        self._price_lo, self._price_hi = (int(b["price"][0]), int(b["price"][1]))
        self._deadline_lo, self._deadline_hi = (int(b["deadline"][0]), int(b["deadline"][1]))
        self._dir_price = int(d["price"])
        self._dir_deadline = int(d["deadline"])
        self._patience = float(patience)
        self._reservation = float(reservation)
        self._grid_points = max(2, int(grid_points))
        # Mechanism A: the decision policy is injectable; default ships and is scored.
        self._strategy: NegotiationStrategy = (
            strategy if strategy is not None else FrontierWalkStrategy()
        )
        # Mechanism B: the opponent-weight estimator is injectable; default ships and
        # is scored. The default heuristic reproduces the original inline
        # ``_estimate_opp_weights`` byte-for-byte (golden trace unchanged); ``"bayesian"``
        # is an opt-in deterministic grid-posterior estimator. Both expose ``estimate``
        # (2-attr) and ``estimate_n`` (N-attr), so the selector controls inference at
        # every attribute count.
        self._opp_model: OpponentModel = _resolve_opponent_model(
            opponent_model,
            grid_points=bayes_grid_points,
            beta=bayes_beta,
            warmup=bayes_warmup,
        )
        self._sessions: dict[str, NegotiationSession] = {}
        self._opp_offers: dict[str, list[dict[str, int]]] = {}
        self._session_counter = 0

    # -- disclosed profile (used by the scenario agent to emit the wire token) --

    @property
    def weights(self) -> tuple[float, float]:
        """This agent's ``(w_price, w_deadline)`` weights.

        Example::

            wp, wd = neg.weights
        """
        return (self._w_price, self._w_deadline)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """This agent's ``(price_lo, price_hi, deadline_lo, deadline_hi)`` bounds.

        Example::

            plo, phi, dlo, dhi = neg.bounds
        """
        return (self._price_lo, self._price_hi, self._deadline_lo, self._deadline_hi)

    @property
    def direction(self) -> tuple[int, int]:
        """This agent's ``(dir_price, dir_deadline)`` (+1 higher-better, -1 lower-better).

        Example::

            dp, dd = neg.direction
        """
        return (self._dir_price, self._dir_deadline)

    # -- N-attribute disclosure accessors (iter 5a; used by the driver to emit
    # -- profile/offer tokens generically once >2 attributes are configured). For the
    # -- default (price, deadline) these are consistent with the 2-tuple properties above.

    def attr_names(self) -> tuple[str, ...]:
        """Ordered attribute names this agent negotiates over.

        Example::

            names = neg.attr_names()  # ("price", "deadline") by default
        """
        return self._attrs

    def attr_weights(self) -> dict[str, float]:
        """This agent's per-attribute weights keyed by attribute name.

        Example::

            w = neg.attr_weights()  # {"price": 0.7, "deadline": 0.3}
        """
        return dict(self._w)

    def attr_bounds(self) -> dict[str, tuple[int, int]]:
        """This agent's per-attribute ``(lo, hi)`` bounds keyed by attribute name.

        Example::

            b = neg.attr_bounds()  # {"price": (30, 100), "deadline": (1, 30)}
        """
        return dict(self._bounds)

    def attr_direction(self) -> dict[str, int]:
        """This agent's per-attribute direction keyed by attribute name.

        ``+1`` means higher-is-better, ``-1`` means lower-is-better.

        Example::

            d = neg.attr_direction()  # {"price": -1, "deadline": 1}
        """
        return dict(self._dir)

    # -- internal helpers --

    def _u_me(self, price: float, deadline: float) -> float:
        """Own two-attribute utility -- the byte-identical ``(price, deadline)`` fast path.

        The N-attribute path no longer routes through this scalar seam: the strategy
        receives **real** per-attribute candidate values via :meth:`_u_me_vec`. The
        former *held-at-ideal* stub (which pinned every attribute beyond price/deadline
        at its ideal, so extras never varied -- the Problem-07 anti-pattern (a) for
        N > 2) is therefore removed.

        Example::

            u = neg._u_me(30, 1)
        """
        return _utility(
            price,
            deadline,
            self._w_price,
            self._w_deadline,
            self._price_lo,
            self._price_hi,
            self._dir_price,
            self._deadline_lo,
            self._deadline_hi,
            self._dir_deadline,
        )

    def _u_me_vec(self, values: Mapping[str, int]) -> float:
        """Own N-attribute utility over a full candidate value-dict (no stub).

        Feeds real per-attribute values into :func:`_utility_n`, so every configured
        attribute genuinely varies and is traded along the frontier. This is the seam
        the vector :class:`FrontierWalkStrategy` path consumes.

        Example::

            u = neg._u_me_vec({"price": 30, "deadline": 1, "quality": 9})
        """
        return _utility_n(values, self._w, self._bounds, self._dir, self._attrs)

    def _u_opp(self, price: float, deadline: float, wp_opp: float, wd_opp: float) -> float:
        # Opponent shares the market bounds with mirrored direction.
        return _utility(
            price,
            deadline,
            wp_opp,
            wd_opp,
            self._price_lo,
            self._price_hi,
            -self._dir_price,
            self._deadline_lo,
            self._deadline_hi,
            -self._dir_deadline,
        )

    def _u_opp_vec(self, values: Mapping[str, int], w_opp: Mapping[str, float]) -> float:
        """Inferred-opponent N-attribute utility: market bounds, mirrored direction.

        The N-attribute counterpart of :meth:`_u_opp`. The opponent is modelled as
        sharing the market bounds with each attribute's direction flipped; weights are
        the *inferred* ``w_opp`` (never the opponent's disclosed profile). At two
        attributes this returns exactly what :meth:`_u_opp` returns.

        Example::

            u = neg._u_opp_vec({"price": 95, "deadline": 1}, {"price": 0.5, "deadline": 0.5})
        """
        mirrored = {a: -self._dir[a] for a in self._attrs}
        return _utility_n(values, w_opp, self._bounds, mirrored, self._attrs)

    def _grid(self, lo: int, hi: int) -> list[int]:
        if hi <= lo:
            return [lo]
        step = max(1, (hi - lo) // (self._grid_points - 1))
        values = list(range(lo, hi + 1, step))
        if values[-1] != hi:
            values.append(hi)
        return values

    # -- protocol surface (identical signatures to AlternatingOffers) --

    async def open(self, partner: AgentId, terms: Terms) -> NegotiationSession:
        """Open a negotiation session with initial ``terms``.

        Example::

            session = await neg.open(AgentId("a2"), Terms(price=Money(amount=30)))
        """
        sid = f"{self._agent_id}-s{self._session_counter}"
        self._session_counter += 1
        session = NegotiationSession(
            id=sid,
            initiator=self._agent_id,
            partner=partner,
            status=NegotiationStatus.OPEN,
            current_terms=terms,
            history=[terms],
        )
        self._sessions[session.id] = session
        return session

    async def offer(self, session: NegotiationSession, terms: Terms) -> None:
        """Record an offer (incoming or outgoing) as the session's current terms.

        Example::

            await neg.offer(session, Terms(price=Money(amount=80)))
        """
        session.current_terms = terms
        session.history.append(terms)

    async def respond(self, session: NegotiationSession) -> NegotiationResponse:
        """Perceive the current offer, then delegate accept/counter to the strategy.

        Reads and records the opponent's current offer over **every** configured
        attribute, computes the decaying aspiration and the *inferred* opponent model
        (via the injected ``opponent_model`` -- heuristic or Bayesian -- at the right
        attribute count), assembles a :class:`DecisionContext`, and delegates the
        accept/counter choice to ``self._strategy``. At two attributes the fast path
        reproduces the original inline policy exactly, so behaviour (and the golden
        trace) is unchanged; with more attributes the vector path is taken.

        Example::

            resp = await neg.respond(session)
        """
        cur = session.current_terms
        if cur is None or cur.price is None:
            return NegotiationResponse(accepted=True)

        # Perceive the current offer over every configured attribute: ``price`` from
        # ``Money.amount``; every other attribute from ``Terms.conditions`` (defaulting
        # to that attribute's upper bound when absent -- the same fallback the legacy
        # two-attribute path used for a missing deadline).
        offer_vals: dict[str, int] = {}
        for a in self._attrs:
            if a == "price":
                offer_vals[a] = int(cur.price.amount)
            else:
                offer_vals[a] = int(cur.conditions.get(a, self._bounds[a][1]))
        self._opp_offers.setdefault(session.id, []).append(offer_vals)

        my_round = len(session.history)
        aspiration = max(self._reservation, self._patience**my_round)
        stored = self._opp_offers.get(session.id, [])

        if self._is_two_attr:
            # --- two-attribute fast path: byte-identical to the shipped behaviour ---
            price = offer_vals["price"]
            deadline = offer_vals["deadline"]
            wp_opp, wd_opp = self._opp_model.estimate(
                tuple((o["price"], o["deadline"]) for o in stored),
                (self._price_lo, self._price_hi, self._deadline_lo, self._deadline_hi),
                (self._dir_price, self._dir_deadline),
            )

            def _u_opp_bound(p: int, dl: int) -> float:
                return self._u_opp(p, dl, wp_opp, wd_opp)

            ctx = DecisionContext(
                current_price=price,
                current_deadline=deadline,
                round=my_round,
                aspiration=aspiration,
                u_me=self._u_me,
                u_opp=_u_opp_bound,
                price_grid=tuple(self._grid(self._price_lo, self._price_hi)),
                deadline_grid=tuple(self._grid(self._deadline_lo, self._deadline_hi)),
                opp_offers=tuple((o["price"], o["deadline"]) for o in stored),
            )
            decision = self._strategy.decide(ctx)
            if decision.accept:
                return NegotiationResponse(accepted=True)

            bp = decision.counter_price if decision.counter_price is not None else price
            bd = decision.counter_deadline if decision.counter_deadline is not None else deadline
            counter = Terms(price=Money(amount=bp), conditions={"deadline": bd})
            return NegotiationResponse(accepted=False, counter_terms=counter)

        # --- N-attribute path (more than price + deadline configured) ---
        w_opp = self._opp_model.estimate_n(stored, self._attrs, self._bounds, self._dir)

        def _u_opp_vec_bound(values: Mapping[str, int]) -> float:
            return self._u_opp_vec(values, w_opp)

        def _u_scalar_unused(p: int, dl: int) -> float:
            # The vector decision path never reads these scalar callables; they exist
            # only to satisfy the (required) two-attribute fields of DecisionContext.
            return 0.0

        attr_grids = {a: tuple(self._grid(*self._bounds[a])) for a in self._attrs}
        ctx = DecisionContext(
            current_price=int(offer_vals["price"]),
            current_deadline=int(offer_vals.get("deadline", 0)),
            round=my_round,
            aspiration=aspiration,
            u_me=_u_scalar_unused,
            u_opp=_u_scalar_unused,
            price_grid=attr_grids.get("price", ()),
            deadline_grid=attr_grids.get("deadline", ()),
            opp_offers=(),
            attr_names=self._attrs,
            attr_grids=attr_grids,
            current_values=dict(offer_vals),
            u_me_vec=self._u_me_vec,
            u_opp_vec=_u_opp_vec_bound,
            opp_offers_vec=tuple(stored),
        )
        decision = self._strategy.decide(ctx)
        if decision.accept:
            return NegotiationResponse(accepted=True)

        counters = decision.counters if decision.counters is not None else offer_vals
        counter_terms = Terms(
            price=Money(amount=int(counters["price"])),
            conditions={a: int(counters[a]) for a in self._attrs if a != "price"},
        )
        return NegotiationResponse(accepted=False, counter_terms=counter_terms)

    async def close(self, session: NegotiationSession) -> Agreement | None:
        """Close a session, returning an :class:`Agreement` if terms exist.

        Example::

            agreement = await neg.close(session)
        """
        if session.status == NegotiationStatus.AGREED or session.current_terms is not None:
            session.status = NegotiationStatus.AGREED
            return Agreement(
                session_id=session.id,
                terms=session.current_terms or Terms(),
                parties=[session.initiator, session.partner],
            )
        session.status = NegotiationStatus.REJECTED
        return None
