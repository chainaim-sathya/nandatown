# SPDX-License-Identifier: Apache-2.0
# This is a deliberately white-box codec test: it imports the module-internal wire
# encoders/decoders directly (see the module docstring), so private-usage is expected.
# pyright: reportPrivateUsage=false
"""Byte-identity guard for the N-attribute wire path (iteration 5b.2).

5b.2 vectorises the negotiation agent + factory from the hard-coded ``(price,
deadline)`` pair to an arbitrary attribute vector. This module is the focused proof
that the generalisation is **behaviour-preserving at two attributes** -- the property
the committed golden trace depends on -- and **correct at N**:

* the N-attribute encoders reduce to the legacy 2-attribute tokens byte-for-byte when
  ``attrs == ("price", "deadline")`` with the canonical prefixes / field-keys;
* a 2-attribute token decodes identically through the legacy and the vector decoder;
* an N-attribute offer round-trips through ``encode -> decode`` unchanged, and the
  legacy 2-attribute decoder *rejects* it (wrong token count, not silent truncation);
* a grown 3-attribute profile token carries the expected field count.

The full-scenario byte lock lives in ``test_negotiation_golden`` (the market trace);
the decision-seam byte-identity lives in ``test_negotiation_strategy_seam``. Together
with this codec guard they pin every layer the 2-attribute golden rides on.

These are deliberately white-box checks on the wire codec, so they import the
module-internal encoders/decoders directly -- exercising the byte boundary *is* the
point of the guard.

Example::

    pytest packages/nest-core/tests/test_negotiation_nattr_equivalence.py -v
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from nest_core.plugins import PluginRegistry
from nest_core.scenarios_builtin.negotiation_multi import (
    NegotiatorAgent,
    _decode,
    _decode_n,
    _encode,
    _encode_n,
    _encode_profile,
    _encode_profile_n,
)
from nest_core.types import AgentId

#: Wire prefixes / profile field-keys for a genuine 3-attribute (price, deadline,
#: quality) scenario -- the exact maps the factory derives for an extra ``quality``
#: axis (single-char wire key ``q``, matching the validator's N-axis recovery).
_3ATTR = ("price", "deadline", "quality")
_3PREFIX = {"price": "p", "deadline": "d", "quality": "q"}
_3FIELD_KEYS = {
    "price": ("wp", "pmin", "pmax", "dir_p"),
    "deadline": ("wd", "dmin", "dmax", "dir_d"),
    "quality": ("wq", "qmin", "qmax", "dir_q"),
}


def test_offer_encoder_two_attr_matches_legacy() -> None:
    """``_encode_n`` at (price, deadline) is byte-identical to the legacy ``_encode``.

    Covers every offer-family kind so no token type can drift.

    Example::

        test_offer_encoder_two_attr_matches_legacy()
    """
    for kind in ("offer", "counter", "accept", "close"):
        legacy = _encode(kind, "abcdef1234", 3, 42, 7)
        vector = _encode_n(kind, "abcdef1234", 3, {"price": 42, "deadline": 7})
        assert vector == legacy, f"{kind}: vector token {vector!r} != legacy {legacy!r}"


def test_profile_encoder_two_attr_matches_legacy() -> None:
    """``_encode_profile_n`` at (price, deadline) matches the legacy profile token.

    Example::

        test_profile_encoder_two_attr_matches_legacy()
    """
    legacy = _encode_profile("abcdef1234", AgentId("buyer-0"), 0.7, 0.3, 30, 100, 1, 30, -1, 1)
    vector = _encode_profile_n(
        "abcdef1234",
        AgentId("buyer-0"),
        {"price": 0.7, "deadline": 0.3},
        {"price": (30, 100), "deadline": (1, 30)},
        {"price": -1, "deadline": 1},
    )
    assert vector == legacy


def test_two_attr_offer_round_trips_through_legacy_and_vector() -> None:
    """A 2-attr token decodes identically via the legacy and vector decoders.

    Example::

        test_two_attr_offer_round_trips_through_legacy_and_vector()
    """
    payload = _encode_n("offer", "sess0000", 0, {"price": 30, "deadline": 1})
    assert _decode(payload) == ("offer", "sess0000", 0, 30, 1)
    vector = _decode_n(payload)
    assert vector is not None
    kind, sid8, rnd, values = vector
    assert (kind, sid8, rnd) == ("offer", "sess0000", 0)
    assert values == {"price": 30, "deadline": 1}


def test_three_attr_offer_round_trips() -> None:
    """A 3-attribute offer survives ``encode -> decode`` unchanged (genuine quality axis).

    The legacy 2-attribute decoder must REJECT the grown token rather than silently
    parse a truncated 2-attribute view of it.

    Example::

        test_three_attr_offer_round_trips()
    """
    values = {"price": 55, "deadline": 4, "quality": 9}
    payload = _encode_n("counter", "sess0001", 2, values, _3ATTR, _3PREFIX)
    assert b":q9" in payload  # the quality axis genuinely rides on the wire
    decoded = _decode_n(payload, _3ATTR, _3PREFIX)
    assert decoded is not None
    kind, sid8, rnd, got = decoded
    assert (kind, sid8, rnd) == ("counter", "sess0001", 2)
    assert got == values
    assert _decode(payload) is None  # legacy 2-attr decoder rejects the 3-attr token


def test_three_attr_profile_has_sixteen_fields() -> None:
    """A 3-attribute profile token grows to the expected 16 colon-separated fields.

    head ``nego:profile:<sid8>:<agent>`` (4) + 3 weight + 6 bound + 3 dir = 16, with
    the quality axis disclosed as ``wq`` / ``qmin`` / ``qmax`` / ``dir_q``.

    Example::

        test_three_attr_profile_has_sixteen_fields()
    """
    token = _encode_profile_n(
        "abcdef12",
        AgentId("buyer-0"),
        {"price": 0.5, "deadline": 0.3, "quality": 0.2},
        {"price": (30, 100), "deadline": (1, 30), "quality": (0, 10)},
        {"price": -1, "deadline": 1, "quality": 1},
        _3ATTR,
        _3FIELD_KEYS,
    )
    text = token.decode()
    assert text.count(":") + 1 == 16, f"expected 16 fields, got {text!r}"
    assert "wq0.20" in text  # quality weight disclosed to 2 dp
    assert "qmin0" in text and "qmax10" in text  # quality bounds disclosed


# --- iter 5b.2: full agent-turn byte-identity at two attributes -----------------
#
# The codec checks above pin the encoder/decoder bytes; this section pins the bytes
# the generalised ``NegotiatorAgent`` actually *emits* on a real turn. It drives the
# agent (not just the codec) through ``on_start`` and ``on_message`` at the default
# ``(price, deadline)`` vector and asserts every wire token is byte-identical to the
# legacy two-attribute encoders -- the unit-level companion to the full-scenario
# golden trace lock in ``test_negotiation_golden``.


class _CapturingContext:
    """A minimal ``AgentContext`` that records every ``send`` for byte inspection.

    Implements the full ``AgentContext`` protocol so the real ``NegotiatorAgent`` runs
    unmodified. ``send`` appends ``(target, payload)`` to :attr:`sent`; ``broadcast``
    and ``schedule`` record too so no parameter is unused; the rest return inert
    defaults.

    Example::

        ctx = _CapturingContext(AgentId("buyer-0"), {"negotiation": neg})
    """

    def __init__(self, agent_id: AgentId, plugins: dict[str, Any]) -> None:
        self._agent_id = agent_id
        self._plugins = plugins
        self.sent: list[tuple[AgentId, bytes]] = []
        self.scheduled: list[tuple[float, bytes]] = []

    @property
    def agent_id(self) -> AgentId:
        """This agent's id."""
        return self._agent_id

    @property
    def time(self) -> float:
        """Current simulation time (inert in the harness)."""
        return 0.0

    @property
    def rng(self) -> random.Random:
        """A deterministic RNG (unused by the negotiation agent)."""
        return random.Random(0)

    @property
    def plugins(self) -> dict[str, Any]:
        """Resolved layer plugins available to the agent."""
        return self._plugins

    async def send(self, to: AgentId, payload: bytes) -> None:
        """Record an outbound message for byte-level inspection."""
        self.sent.append((to, payload))

    async def broadcast(self, payload: bytes) -> None:
        """Record a broadcast (unused on the negotiation path)."""
        self.sent.append((self._agent_id, payload))

    async def schedule(self, delay: float, payload: bytes) -> None:
        """Record a self-scheduled message (unused on the negotiation path)."""
        self.scheduled.append((delay, payload))


def _two_attr_agent_and_ctx() -> tuple[NegotiatorAgent, _CapturingContext]:
    """Build a buyer ``NegotiatorAgent`` + capturing ctx at ``(price, deadline)``.

    The ChainAim plugin is resolved through the same :class:`PluginRegistry` the
    runtime/factory use (never a direct cross-layer import), constructed with the
    canonical two-attribute profile so its emissions match the legacy path.

    Example::

        agent, ctx = _two_attr_agent_and_ctx()
    """
    buyer = AgentId("buyer-0")
    seller = AgentId("seller-0")
    weights = {"price": 0.7, "deadline": 0.3}
    bounds = {"price": (30, 100), "deadline": (1, 30)}
    direction = {"price": -1, "deadline": 1}

    plugin_cls = PluginRegistry().resolve("negotiation", "chainaim_neg_multi_pareto")
    assert isinstance(plugin_cls, type)
    neg = plugin_cls(
        buyer,
        weights=weights,
        bounds=bounds,
        direction=direction,
        patience=0.9,
        attributes=("price", "deadline"),
    )
    ctx = _CapturingContext(buyer, {"negotiation": neg})
    agent = NegotiatorAgent(
        buyer,
        seller,
        is_initiator=True,
        attrs=("price", "deadline"),
        weights=weights,
        bounds=bounds,
        direction=direction,
        opening={"price": 30, "deadline": 1},
        concession={"price": ("step", 80, 5), "deadline": ("pin", 15)},
        max_rounds=12,
    )
    return agent, ctx


async def _drive_two_attr_turn() -> None:
    """Drive ``on_start`` then ``on_message`` and assert legacy-identical wire bytes."""
    buyer = AgentId("buyer-0")
    seller = AgentId("seller-0")
    agent, ctx = _two_attr_agent_and_ctx()

    # on_start: the initiator opens, discloses its profile, sends the opening offer.
    await agent.on_start(ctx)
    assert len(ctx.sent) == 2
    (to_profile, profile_bytes), (to_offer, offer_bytes) = ctx.sent
    assert to_profile == seller
    assert to_offer == seller

    # The opening offer carries exactly price=30/deadline=1 at round 0.
    dec_offer = _decode(offer_bytes)
    assert dec_offer is not None
    okind, sid8, ornd, price0, deadline0 = dec_offer
    assert (okind, ornd, price0, deadline0) == ("offer", 0, 30, 1)

    # The profile disclosure is byte-identical to the legacy 12-field profile encoder.
    assert profile_bytes == _encode_profile(sid8, buyer, 0.7, 0.3, 30, 100, 1, 30, -1, 1)
    # ...and the opening offer is byte-identical to the legacy offer encoder.
    assert offer_bytes == _encode("offer", sid8, 0, 30, 1)

    # on_message: a poor seller offer must be answered on the canonical 2-attr wire.
    ctx.sent.clear()
    seller_offer = _encode_n("offer", sid8, 0, {"price": 95, "deadline": 30})
    await agent.on_message(ctx, seller, seller_offer)

    assert len(ctx.sent) == 1
    to_counter, counter_bytes = ctx.sent[0]
    assert to_counter == seller
    # The legacy 2-attribute decoder must ACCEPT the emitted token -- proving exactly
    # price+deadline rode the wire (no extra axis, canonical prefixes/order), the same
    # rejection contract ``test_three_attr_offer_round_trips`` relies on.
    decoded = _decode(counter_bytes)
    assert decoded is not None
    kind, csid8, rnd, cprice, cdeadline = decoded
    assert kind in ("counter", "accept")
    assert counter_bytes == _encode(kind, csid8, rnd, cprice, cdeadline)


def test_two_attr_agent_turn_emits_legacy_tokens() -> None:
    """A full 2-attribute agent turn emits byte-identical legacy wire tokens.

    Runs the async drive helper to completion; ``asyncio.run`` keeps the check
    independent of any pytest async-mode configuration.

    Example::

        test_two_attr_agent_turn_emits_legacy_tokens()
    """
    asyncio.run(_drive_two_attr_turn())
