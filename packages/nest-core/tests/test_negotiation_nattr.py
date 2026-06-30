# SPDX-License-Identifier: Apache-2.0
"""Public-surface tests for the N-attribute negotiation generalisation (iter 4 + 5a).

Following the repo convention (see ``test_negotiation_ir.py``: "no private imports"),
these tests exercise the generalised machinery **only through public entry points** --
the negotiation validators and the plugin's public accessors -- never by importing
underscore-prefixed internals. The two-attribute byte-identity of the generalised
wire path is already locked by ``test_negotiation_golden`` (the legacy encoders now
delegate to the N-attribute encoders, so a byte-identical golden trace *is* that
proof). What remains, and is covered here, is:

1. The validators parse and score genuine **three-attribute** profiles/offers built
   on the wire, through the public ``validate_events`` registry path.
2. The plugin's public ``attributes`` kwarg and ``attr_*`` accessors behave for both
   the two-attribute default and a three-attribute configuration.

Example::

    pytest packages/nest-core/tests/test_negotiation_nattr.py -v
"""

from __future__ import annotations

from typing import Any

from nest_core.types import AgentId
from nest_core.validators import validate_events
from nest_plugins_reference.negotiation.chainaim_neg_multi_pareto import (
    ChainAimMultiAttributeNegotiation,
)


def _send(agent: str, to: str, msg: str) -> dict[str, Any]:
    """Build a minimal ``send`` trace event carrying ``msg``."""
    return {"kind": "send", "agent": agent, "to": to, "msg": msg}


def test_validators_score_two_attribute_trace_through_public_path() -> None:
    """A 2-attribute disclosed trace is scored (sanity anchor for the public path).

    Builds two opposed 12-field profiles and an accept on the wire, then runs the
    public negotiation validator registry; disclosure must PASS (both parties
    disclosed) and IR must no-op PASS (no reservation).
    """
    buyer = "nego:profile:s1:buyer-0:wp0.70:wd0.30:pmin30:pmax100:dmin1:dmax30:dir_p-1:dir_d+1"
    seller = "nego:profile:s1:seller-0:wp0.30:wd0.70:pmin30:pmax100:dmin1:dmax30:dir_p+1:dir_d-1"
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _send("buyer-0", "seller-0", "nego:offer:s1:r0:p30:d1"),
        _send("seller-0", "buyer-0", "nego:accept:s1:r1:p30:d1"),
    ]
    by_name = {r.name: r.passed for r in validate_events(events, "negotiation")}
    assert by_name["negotiation_profile_disclosed"] is True
    assert by_name["negotiation_individually_rational"] is True


def test_validators_score_three_attribute_trace_through_public_path() -> None:
    """A genuine THREE-attribute disclosed trace is parsed and scored end-to-end (iter 5b.0).

    This is the first test that drives a real 3-attribute negotiation through the public
    ``validate_events`` registry -- the schema is auto-detected from the disclosed
    ``nego:profile`` tokens (weight tokens ``wp``/``wd``/``wq`` -> attribute axes
    ``p``/``d``/``q``), and every validator must parse the 16-field profile and the
    3-value offer/accept tokens without error.

    Profiles are opposed on price (buyer wants cheap, seller dear) and deadline (buyer
    wants a long settlement window, seller short), and SHARED on quality (both prefer
    higher). The agreement settles at the buyer's price ideal, the seller's deadline
    ideal, and the shared quality ideal (``p30/d1/q10``): a Pareto-optimal logroll. The
    structurally-guaranteed verdicts are asserted (disclosure, IR no-op, and the
    exchanged-offers Pareto check), and all four negotiation validators must run.
    """
    buyer = (
        "nego:profile:s1:buyer-0:wp0.50:wd0.30:wq0.20:"
        "pmin30:pmax100:dmin1:dmax30:qmin0:qmax10:dir_p-1:dir_d+1:dir_q+1"
    )
    seller = (
        "nego:profile:s1:seller-0:wp0.30:wd0.20:wq0.50:"
        "pmin30:pmax100:dmin1:dmax30:qmin0:qmax10:dir_p+1:dir_d-1:dir_q+1"
    )
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _send("buyer-0", "seller-0", "nego:offer:s1:r0:p30:d1:q10"),
        _send("seller-0", "buyer-0", "nego:accept:s1:r1:p30:d1:q10"),
    ]
    by_name = {r.name: r for r in validate_events(events, "negotiation")}
    # All four negotiation validators ran against the 3-attribute trace.
    assert set(by_name) == {
        "negotiation_pareto_efficient",
        "negotiation_profile_disclosed",
        "negotiation_frontier_efficient",
        "negotiation_individually_rational",
    }
    # Structurally-guaranteed verdicts for a well-formed disclosed 3-attribute trace.
    assert by_name["negotiation_profile_disclosed"].passed is True
    assert by_name["negotiation_individually_rational"].passed is True
    # The agreement is the only exchanged offer, so no exchanged offer can dominate it.
    assert by_name["negotiation_pareto_efficient"].passed is True


def test_validators_three_attribute_frontier_sweeps_quality_axis() -> None:
    """The feasible-frontier check genuinely SWEEPS the third (quality) axis (iter 5b.0).

    Both parties share all three directions (cheaper, sooner, higher-quality are better
    for both -- a fully cooperative / shared-push configuration), so the unique feasible
    optimum for both is ``p=30, d=1, q=10``. The agreement is placed at a dominated
    interior point with quality at its WORST (``p65/d15/q0``); the feasible sweep must
    therefore find a strictly-dominating point and FAIL. Because the dominator detail
    renders every swept axis, a quality token must appear -- proof the quality axis was
    reconstructed and swept, not ignored. Disclosure still PASSes (both disclosed).
    """
    buyer = (
        "nego:profile:s2:buyer-0:wp0.40:wd0.30:wq0.30:"
        "pmin30:pmax100:dmin1:dmax30:qmin0:qmax10:dir_p-1:dir_d-1:dir_q+1"
    )
    seller = (
        "nego:profile:s2:seller-0:wp0.30:wd0.30:wq0.40:"
        "pmin30:pmax100:dmin1:dmax30:qmin0:qmax10:dir_p-1:dir_d-1:dir_q+1"
    )
    events = [
        _send("buyer-0", "seller-0", buyer),
        _send("seller-0", "buyer-0", seller),
        _send("buyer-0", "seller-0", "nego:offer:s2:r0:p65:d15:q0"),
        _send("seller-0", "buyer-0", "nego:accept:s2:r1:p65:d15:q0"),
    ]
    by_name = {r.name: r for r in validate_events(events, "negotiation")}
    assert by_name["negotiation_profile_disclosed"].passed is True
    frontier = by_name["negotiation_frontier_efficient"]
    assert frontier.passed is False, frontier.detail
    # The dominator detail renders all swept axes as <first-char><value>; a quality
    # token ("/q...") proves the third axis was reconstructed and swept.
    assert "/q" in frontier.detail, frontier.detail


def test_plugin_attributes_kwarg_defaults_and_accessors() -> None:
    """The ``attributes`` kwarg defaults to (price, deadline) with consistent accessors."""
    neg = ChainAimMultiAttributeNegotiation(
        AgentId("buyer-0"),
        weights={"price": 0.7, "deadline": 0.3},
        bounds={"price": (30, 100), "deadline": (1, 30)},
        direction={"price": -1, "deadline": 1},
    )
    assert neg.attr_names() == ("price", "deadline")
    assert neg.attr_weights() == {"price": 0.7, "deadline": 0.3}
    assert neg.attr_bounds() == {"price": (30, 100), "deadline": (1, 30)}
    assert neg.attr_direction() == {"price": -1, "deadline": 1}
    # The legacy 2-tuple disclosure properties remain consistent with the accessors.
    assert neg.weights == (0.7, 0.3)
    assert neg.bounds == (30, 100, 1, 30)
    assert neg.direction == (-1, 1)


def test_plugin_three_attribute_construction_and_accessors() -> None:
    """A 3-attribute plugin instance constructs and reports all three attributes."""
    neg = ChainAimMultiAttributeNegotiation(
        AgentId("buyer-0"),
        weights={"price": 0.5, "deadline": 0.3, "quality": 0.2},
        bounds={"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
        direction={"price": -1, "deadline": 1, "quality": 1},
        attributes=("price", "deadline", "quality"),
    )
    assert neg.attr_names() == ("price", "deadline", "quality")
    assert neg.attr_weights() == {"price": 0.5, "deadline": 0.3, "quality": 0.2}
    assert neg.attr_bounds()["quality"] == (0, 10)
    assert neg.attr_direction()["quality"] == 1


def test_plugin_requires_named_attributes_present_in_maps() -> None:
    """Naming an attribute absent from ``weights`` raises -- no silent partial config."""
    raised = False
    try:
        ChainAimMultiAttributeNegotiation(
            AgentId("buyer-0"),
            weights={"price": 0.7, "deadline": 0.3},  # 'quality' missing
            bounds={"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
            direction={"price": -1, "deadline": 1, "quality": 1},
            attributes=("price", "deadline", "quality"),
        )
    except KeyError:
        raised = True
    assert raised, "expected KeyError when a named attribute is absent from weights"


def test_plugin_rejects_empty_attributes() -> None:
    """An empty ``attributes`` tuple is rejected at construction."""
    raised = False
    try:
        ChainAimMultiAttributeNegotiation(AgentId("buyer-0"), attributes=())
    except ValueError:
        raised = True
    assert raised, "expected ValueError for empty attributes"
