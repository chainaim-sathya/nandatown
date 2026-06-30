# SPDX-License-Identifier: Apache-2.0
"""Negotiation scenario — buyers and sellers bargain via the negotiation layer.

This is the first scenario that actually *drives* the negotiation layer plugin
(``alternating_offers`` by default). Every other built-in scenario resolves the
negotiation plugin but never calls it, so before this scenario existed there was
no simulator run that exercised ``open/offer/respond/close`` and therefore no
real negotiation trace to inspect.

Design (genuine two-party alternating offers, no mocks):

- Each agent owns its *own* ``AlternatingOffers`` instance (per-agent, like the
  identity layer), wired in through ``ctx.plugins["negotiation"]``.
- A buyer opens a session, offers a low price, and emits a ``nego:offer`` message.
- The seller, on each inbound offer, records it into its own session via the
  plugin's ``offer()``, calls the plugin's ``respond()`` to decide accept/reject,
  and either ``nego:accept``s (then ``close()``s) or counters higher.
- The buyer mirrors this: on each counter it records terms, calls ``respond()``,
  and either accepts/closes or concedes upward.

Because the work happens as real messages, it shows up in the JSONL trace and is
summarized by ``nest inspect``.

Baseline honesty: the reference ``alternating_offers.respond()`` accepts once a
session's history reaches ~10 entries (its price test is effectively inert for
positive prices). So this baseline converges by *running out the clock* rather
than by reaching a genuinely good price — exactly the single-attribute weakness
that a multi-attribute replacement / adversarial validator is meant to expose.
This module does not modify the plugin; it lets it behave authentically.

Example::

    agents = negotiation_factory(config, plugins)
"""

from __future__ import annotations

from typing import Any

from nest_core.scenario import ScenarioConfig
from nest_core.sim.agent import AgentContext, StateMachineAgent
from nest_core.types import AgentId, Money, Terms

_PREFIX = "nego"


def _encode(kind: str, sid: str, rnd: int, price: int) -> bytes:
    """Encode a negotiation message as ``nego:<kind>:<sid8>:<round>:<price>``."""
    return f"{_PREFIX}:{kind}:{sid[:8]}:{rnd}:{price}".encode()


def _decode(payload: bytes) -> tuple[str, str, int, int] | None:
    """Decode a negotiation message; return ``(kind, sid8, round, price)`` or None."""
    text = payload.decode("utf-8", errors="replace")
    parts = text.split(":")
    if len(parts) != 5 or parts[0] != _PREFIX:
        return None
    kind = parts[1]
    sid8 = parts[2]
    try:
        rnd = int(parts[3])
        price = int(parts[4])
    except ValueError:
        return None
    return kind, sid8, rnd, price


class BuyerAgent(StateMachineAgent):
    """A buyer that bargains down toward a seller using the negotiation plugin.

    The buyer opens low and concedes upward by ``price_step`` per round, capped at
    the seller's most recent counter, while delegating the accept/reject decision
    to the negotiation plugin's ``respond()``.

    Example::

        agent = BuyerAgent(
            AgentId("buyer-0"), AgentId("seller-0"),
            start_price=50, max_price=90, price_step=5,
            sessions=1, max_rounds=10,
        )
    """

    def __init__(
        self,
        agent_id: AgentId,
        seller_id: AgentId,
        start_price: int = 50,
        max_price: int = 90,
        price_step: int = 5,
        sessions: int = 1,
        max_rounds: int = 10,
    ) -> None:
        self._id = agent_id
        self._seller = seller_id
        self._start_price = start_price
        self._max_price = max_price
        self._step = price_step
        self._sessions = sessions
        self._max_rounds = max_rounds
        self._sessions_done = 0
        self._session: Any = None
        self._agreements = 0

    async def _begin_session(self, ctx: AgentContext) -> None:
        """Open a fresh negotiation session and send the opening offer."""
        neg = ctx.plugins.get("negotiation")
        if neg is None:
            return
        # Small deterministic private jitter so pairs are not identical (seeded RNG).
        jitter = ctx.rng.randint(0, self._step)
        open_price = max(1, self._start_price - jitter)
        session = await neg.open(self._seller, Terms(price=Money(amount=open_price)))
        self._session = session
        await ctx.send(self._seller, _encode("offer", session.id, 0, open_price))

    async def on_start(self, ctx: AgentContext) -> None:
        """Kick off the first negotiation session.

        Example::

            await agent.on_start(ctx)
        """
        await self._begin_session(ctx)

    async def on_message(self, ctx: AgentContext, sender: AgentId, payload: bytes) -> None:
        """Handle seller counters/acceptances, driving the plugin each step.

        Example::

            await agent.on_message(ctx, AgentId("seller-0"), b"nego:counter:abc:1:80")
        """
        decoded = _decode(payload)
        if decoded is None or self._session is None:
            return
        kind, _sid8, rnd, price = decoded
        neg = ctx.plugins.get("negotiation")
        if neg is None:
            return

        if kind == "accept":
            # Seller accepted our latest offer — finalize.
            agreement = await neg.close(self._session)
            if agreement is not None:
                self._agreements += 1
            await ctx.send(self._seller, _encode("close", self._session.id, rnd, price))
            await self._maybe_next(ctx)
            return

        if kind == "counter":
            # Record the seller's counter into our session, then let the plugin decide.
            await neg.offer(self._session, Terms(price=Money(amount=price)))
            resp = await neg.respond(self._session)
            next_round = rnd + 1
            if resp.accepted or next_round >= self._max_rounds:
                agreement = await neg.close(self._session)
                if agreement is not None:
                    self._agreements += 1
                await ctx.send(self._seller, _encode("accept", self._session.id, next_round, price))
                await self._maybe_next(ctx)
                return
            # Concede upward by one step, never above the seller's counter or our ceiling.
            my_price = min(price, self._max_price, self._start_price + self._step * next_round)
            await neg.offer(self._session, Terms(price=Money(amount=my_price)))
            await ctx.send(self._seller, _encode("offer", self._session.id, next_round, my_price))

    async def _maybe_next(self, ctx: AgentContext) -> None:
        """Start the next session if more are configured, else stop."""
        self._sessions_done += 1
        self._session = None
        if self._sessions_done < self._sessions:
            await self._begin_session(ctx)


class SellerAgent(StateMachineAgent):
    """A seller that bargains up from buyers using the negotiation plugin.

    The seller anchors high and concedes downward by ``price_step`` per round but
    never below ``reservation``, delegating accept/reject to the plugin's
    ``respond()``.

    Example::

        agent = SellerAgent(
            AgentId("seller-0"), anchor=90, reservation=40,
            price_step=5, max_rounds=10,
        )
    """

    def __init__(
        self,
        agent_id: AgentId,
        anchor: int = 90,
        reservation: int = 40,
        price_step: int = 5,
        max_rounds: int = 10,
    ) -> None:
        self._id = agent_id
        self._anchor = anchor
        self._reservation = reservation
        self._step = price_step
        self._max_rounds = max_rounds
        self._sessions: dict[AgentId, Any] = {}
        self._agreements = 0

    async def on_message(self, ctx: AgentContext, sender: AgentId, payload: bytes) -> None:
        """Handle buyer offers, driving the plugin each step.

        Example::

            await agent.on_message(ctx, AgentId("buyer-0"), b"nego:offer:abc:0:50")
        """
        decoded = _decode(payload)
        if decoded is None:
            return
        kind, _sid8, rnd, price = decoded
        neg = ctx.plugins.get("negotiation")
        if neg is None:
            return

        if kind == "close":
            # Buyer finalized our acceptance; nothing more to send.
            return

        if kind == "accept":
            # Buyer accepted our latest counter — finalize on our side.
            session = self._sessions.get(sender)
            if session is not None:
                agreement = await neg.close(session)
                if agreement is not None:
                    self._agreements += 1
            return

        if kind == "offer":
            session = self._sessions.get(sender)
            if session is None:
                session = await neg.open(sender, Terms(price=Money(amount=price)))
                self._sessions[sender] = session
            else:
                await neg.offer(session, Terms(price=Money(amount=price)))

            resp = await neg.respond(session)
            next_round = rnd + 1
            if resp.accepted or next_round >= self._max_rounds:
                agreement = await neg.close(session)
                if agreement is not None:
                    self._agreements += 1
                await ctx.send(sender, _encode("accept", session.id, next_round, price))
                return
            # Concede downward by one step, never below reservation or below buyer's offer.
            counter = max(self._reservation, price, self._anchor - self._step * next_round)
            await neg.offer(session, Terms(price=Money(amount=counter)))
            await ctx.send(sender, _encode("counter", session.id, next_round, counter))


def negotiation_factory(
    config: ScenarioConfig,
    plugins: dict[str, Any],
) -> dict[AgentId, StateMachineAgent]:
    """Create buyer and seller agents that drive the negotiation layer plugin.

    Reads all tunables from ``config.task.config`` with sensible defaults, so the
    scenario's behavior is fully controlled by the YAML and nothing is hardcoded:

    - ``patience`` (float): plugin patience discount (default ``0.9``).
    - ``buyer_start_price`` (int): buyer's opening offer (default ``50``).
    - ``buyer_max_price`` (int): buyer's reservation ceiling (default ``90``).
    - ``seller_anchor`` (int): seller's opening ask (default ``90``).
    - ``seller_reservation`` (int): seller's floor (default ``40``).
    - ``price_step`` (int): concession size per round (default ``5``).
    - ``sessions`` (int): negotiations each buyer runs (default ``1``).
    - ``max_rounds`` (int): per-session offer cap, a safety bound (default ``10``).

    Buyers and sellers are paired deterministically: ``buyer-i`` negotiates with
    ``seller-(i % seller_count)``.

    Example::

        agents = negotiation_factory(config, plugins)
    """
    cfg = config.task.config
    patience = float(cfg.get("patience", 0.9))
    buyer_start = int(cfg.get("buyer_start_price", 50))
    buyer_max = int(cfg.get("buyer_max_price", 90))
    seller_anchor = int(cfg.get("seller_anchor", 90))
    seller_reservation = int(cfg.get("seller_reservation", 40))
    price_step = int(cfg.get("price_step", 5))
    sessions = int(cfg.get("sessions", 1))
    max_rounds = int(cfg.get("max_rounds", 10))

    # Resolve buyer/seller counts from roles, falling back to a half/half split.
    if config.agents.roles:
        buyer_count = 0
        seller_count = 0
        for role in config.agents.roles:
            if role.name == "buyer":
                buyer_count = role.count
            elif role.name == "seller":
                seller_count = role.count
    else:
        buyer_count = config.agents.count // 2
        seller_count = config.agents.count - buyer_count

    seller_count = max(seller_count, 1)
    seller_ids = [AgentId(f"seller-{i}") for i in range(seller_count)]
    buyer_ids = [AgentId(f"buyer-{i}") for i in range(buyer_count)]
    all_ids = seller_ids + buyer_ids

    _instantiate_negotiation_plugins(plugins, all_ids, patience)

    agents: dict[AgentId, StateMachineAgent] = {}

    for i in range(seller_count):
        aid = seller_ids[i]
        agents[aid] = SellerAgent(
            aid,
            anchor=seller_anchor,
            reservation=seller_reservation,
            price_step=price_step,
            max_rounds=max_rounds,
        )

    for i in range(buyer_count):
        aid = buyer_ids[i]
        partner = seller_ids[i % seller_count]
        agents[aid] = BuyerAgent(
            aid,
            partner,
            start_price=buyer_start,
            max_price=buyer_max,
            price_step=price_step,
            sessions=sessions,
            max_rounds=max_rounds,
        )

    return agents


def _instantiate_negotiation_plugins(
    plugins: dict[str, Any],
    all_ids: list[AgentId],
    patience: float,
) -> None:
    """Give every agent its own ``AlternatingOffers`` instance via per-agent overrides.

    The resolved ``plugins["negotiation"]`` is a *class*; the negotiation plugin is
    stateful and per-agent (it tracks ``self._sessions``), so — like the identity
    layer — each agent needs its own instance. We store these under
    ``plugins["_agent_plugins"][aid]["negotiation"]`` for the runner to apply, and
    remove the shared class so ``ctx.plugins["negotiation"]`` is unambiguously the
    agent's own instance. Safe to call when *plugins* is empty.

    Example::

        _instantiate_negotiation_plugins(plugins, [AgentId("buyer-0")], 0.9)
    """
    if not plugins:
        return
    neg_cls = plugins.get("negotiation")
    if neg_cls is None or not isinstance(neg_cls, type):
        return
    agent_plugins: dict[AgentId, dict[str, Any]] = plugins.setdefault("_agent_plugins", {})
    for aid in all_ids:
        agent_plugins.setdefault(aid, {})["negotiation"] = neg_cls(aid, patience=patience)
    plugins.pop("negotiation", None)
