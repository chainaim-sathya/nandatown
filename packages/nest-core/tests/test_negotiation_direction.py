# SPDX-License-Identifier: Apache-2.0
"""Public-surface tests for ATTRIBUTE DIRECTION in multi-attribute negotiation.

Why this file exists -- the insight
-----------------------------------
Each negotiable attribute carries, per party, a *direction* sign: ``+1`` = "higher is
better for me", ``-1`` = "lower is better for me". The negotiation validators score every
outcome from the directions each party DISCLOSES in its ``nego:profile`` token -- they do
NOT hard-code what an attribute "means". One consequence, demonstrated here, is that the
SAME ``deadline`` attribute can carry two opposite economic readings with **zero code
change**:

* **Settlement window** (the shipped ``chainaim_multi_attribute_market.yaml``): the buyer wants the
  window LONG (keep cash longer, ``dir_deadline=+1``); the seller wants it SHORT (collect
  sooner, ``dir_deadline=-1``).
* **Delivery lead time** (``multi_attribute_leadtime.yaml``, exercised here): the signs
  flip -- the buyer wants delivery SOON (``dir_deadline=-1``); the seller wants MORE lead
  time (``dir_deadline=+1``).

In both readings the attributes stay OPPOSED on each axis and the weights stay asymmetric
(buyer price-heavy, seller deadline-heavy), so a *logroll* exists: each side concedes the
axis it weights lightly to win the axis it weights heavily. These tests feed the validators
profiles with the FLIPPED (delivery) signs and confirm the verdicts come out right -- the
proof that the frontier check is interpretation-agnostic, not wired to the settlement reading.

Convention (see ``test_negotiation_nattr.py`` / ``test_negotiation_ir.py``): exercise only
through the public ``validate_events`` / ``negotiation_metrics`` entry points -- no
underscore-prefixed imports. The traces below are built by hand on the wire format.

Hand-checked utilities (delivery signs; price in [30,100], deadline in [1,30]; buyer
weights price 0.7 / deadline 0.3, seller weights price 0.3 / deadline 0.7;
U = w_price*u_price + w_deadline*u_deadline, each u in [0,1], 1.0 at the party's ideal end)::

    (30,30): U_buyer=0.700 U_seller=0.700  -> logroll (cheap price, slow delivery): ON frontier
    (65,15): U_buyer=0.505 U_seller=0.488  -> dominated by feasible (35,25): OFF frontier

Example::

    pytest packages/nest-core/tests/test_negotiation_direction.py -v
"""

from __future__ import annotations

from typing import Any

from nest_core.validators import negotiation_metrics, validate_events

# --- Disclosed profiles with DELIVERY-lead-time direction signs (deadline flipped) ----------
# 12-field tokens: wp/wd weights, pmin/pmax/dmin/dmax bounds, dir_p/dir_d directions.
# Buyer:  cheap good (dir_p-1), FAST delivery good (dir_d-1).
# Seller: dear good  (dir_p+1), MORE lead time good (dir_d+1).
_BUYER_DELIVERY = (
    "nego:profile:s1:buyer-0:wp0.70:wd0.30:pmin30:pmax100:dmin1:dmax30:dir_p-1:dir_d-1"
)
_SELLER_DELIVERY = (
    "nego:profile:s1:seller-0:wp0.30:wd0.70:pmin30:pmax100:dmin1:dmax30:dir_p+1:dir_d+1"
)


def _send(agent: str, to: str, msg: str) -> dict[str, Any]:
    """Build a minimal ``send`` trace event carrying ``msg``."""
    return {"kind": "send", "agent": agent, "to": to, "msg": msg}


def _delivery_trace(agree_price: int, agree_deadline: int) -> list[dict[str, Any]]:
    """A complete 2-party delivery-interpretation trace settling at ``(price, deadline)``.

    Both parties disclose the flipped (delivery) profile, then exchange the two corner offers
    -- buyer opens at its bliss (cheap + fast, ``p30/d1``); seller counters at its bliss (dear
    + slow, ``p100/d30``) -- and accept at the requested settlement. The two exchanged offers
    form a Pareto antichain (each is one side's ideal corner), so the *exchanged-offers* check
    cannot flag the settlement; only the *feasible-frontier* check can.
    """
    return [
        _send("buyer-0", "seller-0", _BUYER_DELIVERY),
        _send("seller-0", "buyer-0", _SELLER_DELIVERY),
        _send("buyer-0", "seller-0", "nego:offer:s1:r0:p30:d1"),
        _send("seller-0", "buyer-0", "nego:counter:s1:r1:p100:d30"),
        _send("seller-0", "buyer-0", f"nego:accept:s1:r2:p{agree_price}:d{agree_deadline}"),
    ]


def _by_name(events: list[dict[str, Any]]) -> dict[str, bool]:
    """Run the negotiation validators and return ``{validator_name: passed}``."""
    return {r.name: r.passed for r in validate_events(events, "negotiation")}


def test_delivery_logroll_is_on_the_feasible_frontier() -> None:
    """Under DELIVERY signs, the logroll settlement (cheap price + slow delivery) PASSES.

    ``(price=30, deadline=30)`` gives the buyer its heavily-weighted win (cheapest price) and
    the seller its heavily-weighted win (longest lead time): U=(0.700, 0.700). No feasible
    point makes one side better without making the other worse, so the feasible-frontier check
    PASSES -- exactly as the validator should score an efficient logroll once it reads the
    flipped deadline direction from the disclosed profiles.
    """
    verdicts = _by_name(_delivery_trace(30, 30))
    assert verdicts["negotiation_frontier_efficient"] is True
    assert verdicts["negotiation_pareto_efficient"] is True
    assert verdicts["negotiation_profile_disclosed"] is True
    # No reservation disclosed -> IR no-ops to PASS.
    assert verdicts["negotiation_individually_rational"] is True


def test_delivery_naive_middle_fails_frontier_but_passes_exchanged_offers() -> None:
    """Under DELIVERY signs, a naive middle settlement is caught by the frontier check only.

    ``(price=65, deadline=15)`` U=(0.505, 0.488) is strictly dominated by a feasible logroll
    such as ``(35, 25)`` U=(0.702, 0.601) -- both parties strictly better -- so
    ``negotiation_frontier_efficient`` FAILS. The two exchanged offers are opposite ideal
    corners (an antichain), so neither dominates the settlement and
    ``negotiation_pareto_efficient`` still PASSES. This is the same "exchanged-offers is
    necessary-but-not-sufficient" insight as the settlement scenario, reproduced here purely
    by flipping the disclosed deadline direction -- the validator code is identical.
    """
    verdicts = _by_name(_delivery_trace(65, 15))
    assert verdicts["negotiation_frontier_efficient"] is False
    assert verdicts["negotiation_pareto_efficient"] is True


def test_anac_metrics_agree_with_the_frontier_verdict_under_delivery_signs() -> None:
    """The reported ``pareto_distance`` is consistent with the frontier verdict (display sync).

    These are the same numbers ``nest validate --metrics`` prints. For the on-frontier logroll
    the distance to the feasible Pareto frontier is exactly 0; for the dominated naive middle it
    is strictly positive -- so the reporting metric and the PASS/FAIL gate tell the same story.
    """
    on_frontier = negotiation_metrics(_delivery_trace(30, 30))["aggregate"]["mean_pareto_distance"]
    dominated = negotiation_metrics(_delivery_trace(65, 15))["aggregate"]["mean_pareto_distance"]
    assert on_frontier == 0.0
    assert dominated is not None and dominated > 0.0
