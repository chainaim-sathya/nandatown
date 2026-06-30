# SPDX-License-Identifier: Apache-2.0
"""Multi-attribute negotiation scenario -- price + deadline buyer/seller bargaining.

Attribute meaning (load-bearing for the direction signs below): ``deadline`` denotes the
buyer's **settlement window** -- a net-30/60-style term for how soon the buyer settles the
invoice -- NOT a delivery lead time. Under that reading the default signs are the natural
ones: the buyer wants a LONG window (keep cash longer) and the seller a SHORT one (collect
sooner). A scenario may re-interpret the attribute by disclosing different signs through the
``buyer_dir`` / ``seller_dir`` task-config keys (see ``scenarios/multi_attribute_leadtime.yaml``
for the delivery-lead-time reading, where the deadline signs are flipped). Every validator
scores from the signs each party DISCLOSES, so the check is interpretation-agnostic.

This scenario drives the negotiation layer over a configurable **attribute vector**
(default ``price`` + ``deadline``) so it can exercise either the single-attribute reference
plugin (``alternating_offers``) or the multi-attribute ChainAim plugin
(``chainaim_neg_multi_pareto``) through *one* task type. The factory instantiates
whichever class the ``layers.negotiation`` selector resolved, passing per-agent
private ``weights`` / ``attributes`` / ``grid_points`` **only when the resolved
constructor accepts them** (signature introspection) -- so the same factory drives both
the PASS (multi-attribute) and the FAIL (baseline) runs with no plugin-specific branching.

Per-role strategy assignment (additive): an optional ``task.config.role_plugins``
block (e.g. ``{buyer: chainaim_neg_multi_pareto, seller: alternating_offers}``) may
name a negotiation plugin **per role**, enabling an *asymmetric* (cross-brain)
matchup in a single session. When the block is absent the factory behaves exactly
as before -- every agent uses the one ``layers.negotiation`` class -- so existing
scenarios stay byte-identical.

Every agent discloses its own private profile (weights + bounds + direction) once,
at session open, as a ``nego:profile`` wire token. This disclosure is what lets the
adversarial validator recompute both parties' utilities from the trace alone -- for
*either* plugin -- and is therefore emitted regardless of which plugin is in use.

Wire format (self-labelling short tokens; ``<sid8>`` = first 8 chars of session id)::

    nego:offer:<sid8>:r<round>:p<price>:d<deadline>[:q<quality>...]
    nego:counter:<sid8>:r<round>:p<price>:d<deadline>[:q<quality>...]
    nego:accept:<sid8>:r<round>:p<price>:d<deadline>[:q<quality>...]
    nego:close:<sid8>:r<round>:p<price>:d<deadline>[:q<quality>...]
    nego:profile:<sid8>:<agent>:wp<wp>:wd<wd>[:wq<wq>...]:pmin<>:pmax<>:dmin<>:dmax<>[...]:dir_p<>:dir_d<>[...]

Baseline honesty: ``alternating_offers.respond`` reads only ``price`` and accepts
on its ~round-10 cap, so the baseline ignores the ``deadline`` attribute entirely
and settles by running out the clock -- exactly the single-attribute weakness the
multi-attribute plugin and the Pareto validator are built to expose.

Example::

    agents = negotiation_multi_factory(config, plugins)
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Mapping, Sequence
from typing import Any

from nest_core.scenario import ScenarioConfig
from nest_core.sim.agent import AgentContext, StateMachineAgent
from nest_core.types import AgentId, Money, Terms

_PREFIX = "nego"

#: Canonical single-letter wire prefix per attribute. ``price`` -> ``p`` and
#: ``deadline`` -> ``d`` reproduce the original 2-attribute tokens byte-for-byte; a
#: scenario adding attributes supplies its own entries (e.g. ``{"quality": "q"}``).
_DEFAULT_ATTR_PREFIX: dict[str, str] = {"price": "p", "deadline": "d"}

#: Canonical attribute order for the default 2-attribute scenarios. The N-attribute
#: helpers emit/parse in this order so the bytes match the legacy
#: ``p<price>:d<deadline>`` layout exactly when ``attrs == ("price", "deadline")``.
_DEFAULT_ATTRS: tuple[str, ...] = ("price", "deadline")

#: Per-attribute profile field-token names ``(weight, min, max, dir)``. The defaults
#: reproduce the canonical 12-field profile token (``wp``/``wd``, ``pmin``/``pmax``/
#: ``dmin``/``dmax``, ``dir_p``/``dir_d``); new attributes supply their own group.
_DEFAULT_PROFILE_FIELD_KEYS: dict[str, tuple[str, str, str, str]] = {
    "price": ("wp", "pmin", "pmax", "dir_p"),
    "deadline": ("wd", "dmin", "dmax", "dir_d"),
}


# Retained as the canonical 2-attribute offer encoder: the N-attribute path delegates
# here for the (price, deadline) bytes and the equivalence test uses it as the legacy
# byte-anchor. Intentionally unused within this module post-5b.2 generalisation.
def _encode(  # pyright: ignore[reportUnusedFunction]
    kind: str,
    sid: str,
    rnd: int,
    price: int,
    deadline: int,
) -> bytes:
    """Encode an offer-family token ``nego:<kind>:<sid8>:r<round>:p<price>:d<deadline>``.

    Two-attribute convenience wrapper over :func:`_encode_n` (the canonical encoder);
    delegating keeps the price/deadline bytes identical to the generalised path by
    construction.

    Example::

        _encode("offer", "abcdef12", 0, 30, 1)
    """
    return _encode_n(kind, sid, rnd, {"price": price, "deadline": deadline})


# Retained as the canonical 2-attribute profile encoder (legacy byte-anchor for the
# equivalence test); intentionally unused within this module post-5b.2 generalisation.
def _encode_profile(  # pyright: ignore[reportUnusedFunction]
    sid: str,
    aid: AgentId,
    wp: float,
    wd: float,
    price_lo: int,
    price_hi: int,
    deadline_lo: int,
    deadline_hi: int,
    dir_price: int,
    dir_deadline: int,
) -> bytes:
    """Encode a once-per-agent profile disclosure token (weights to 2 dp).

    Two-attribute convenience wrapper over :func:`_encode_profile_n` (the canonical
    profile encoder); delegating keeps the canonical 12-field token byte-identical to
    the generalised path by construction.

    Example::

        _encode_profile("abcdef12", AgentId("buyer-0"), 0.7, 0.3, 30, 100, 1, 30, -1, 1)
    """
    return _encode_profile_n(
        sid,
        aid,
        {"price": wp, "deadline": wd},
        {"price": (price_lo, price_hi), "deadline": (deadline_lo, deadline_hi)},
        {"price": dir_price, "deadline": dir_deadline},
    )


def _encode_profile_n(
    sid: str,
    aid: AgentId,
    weights: Mapping[str, float],
    bounds: Mapping[str, tuple[int, int]],
    direction: Mapping[str, int],
    attrs: Sequence[str] = _DEFAULT_ATTRS,
    field_keys: Mapping[str, tuple[str, str, str, str]] | None = None,
) -> bytes:
    """N-attribute generalisation of :func:`_encode_profile` (iter 4).

    Emits, after the ``nego:profile:<sid8>:<agent>`` head, a weight token per
    attribute, then a ``min``/``max`` bound-token pair per attribute, then a direction
    token per attribute -- all in ``attrs`` order. For ``attrs == ("price",
    "deadline")`` with the default field-key map this reproduces the canonical
    12-field token byte-for-byte (``wp``/``wd``, ``pmin``/``pmax``/``dmin``/``dmax``,
    ``dir_p``/``dir_d``); more attributes extend each group. The matching parser is
    ``_NegProfile.parse`` in ``nest_core.validators``.

    Args:
        sid: Session id (first 8 chars are emitted).
        aid: Disclosing agent id.
        weights: Per-attribute weights (rendered to 2 dp).
        bounds: Per-attribute ``(lo, hi)`` bounds.
        direction: Per-attribute direction (``+1``/``-1``).
        attrs: Ordered attribute names to emit.
        field_keys: Attribute -> ``(weight_key, min_key, max_key, dir_key)`` token
            names; defaults reproduce the legacy price/deadline field names.

    Returns:
        The encoded profile token bytes.

    Example::

        _encode_profile_n(
            "abcdef12", AgentId("buyer-0"),
            {"price": 0.7, "deadline": 0.3},
            {"price": (30, 100), "deadline": (1, 30)},
            {"price": -1, "deadline": 1},
        )
    """
    keys = dict(_DEFAULT_PROFILE_FIELD_KEYS if field_keys is None else field_keys)
    head = f"{_PREFIX}:profile:{sid[:8]}:{aid}"
    wtoks = [f"{keys[a][0]}{weights[a]:.2f}" for a in attrs]
    btoks: list[str] = []
    for a in attrs:
        lo, hi = bounds[a]
        btoks.append(f"{keys[a][1]}{lo}")
        btoks.append(f"{keys[a][2]}{hi}")
    dtoks = [f"{keys[a][3]}{direction[a]:+d}" for a in attrs]
    return ":".join([head, *wtoks, *btoks, *dtoks]).encode()


# Retained as the canonical 2-attribute decoder (legacy byte-anchor for the equivalence
# test); intentionally unused within this module post-5b.2 generalisation.
def _decode(  # pyright: ignore[reportUnusedFunction]
    payload: bytes,
) -> tuple[str, str, int, int, int] | None:
    """Decode an offer-family token; return ``(kind, sid8, round, price, deadline)`` or None.

    Profile and unparseable tokens return ``None``. Two-attribute convenience wrapper
    over :func:`_decode_n` (the canonical decoder), repackaging its attribute dict
    into the legacy positional tuple -- identical acceptance and values by construction.

    Example::

        _decode(b"nego:offer:abcdef12:r0:p30:d1")
    """
    parsed = _decode_n(payload)
    if parsed is None:
        return None
    kind, sid8, rnd, values = parsed
    return kind, sid8, rnd, values["price"], values["deadline"]


def _encode_n(
    kind: str,
    sid: str,
    rnd: int,
    values: Mapping[str, int],
    attrs: Sequence[str] = _DEFAULT_ATTRS,
    prefix: Mapping[str, str] | None = None,
) -> bytes:
    """N-attribute generalisation of :func:`_encode` (iter 4).

    Appends one ``<prefix><value>`` token per attribute in ``attrs`` order, after the
    fixed ``nego:<kind>:<sid8>:r<round>`` head. For ``attrs == ("price", "deadline")``
    with the default prefix map this is byte-identical to :func:`_encode`; additional
    attributes append further tokens (e.g. ``:q9``). The matching decoder is
    :func:`_decode_n`.

    Args:
        kind: Token kind (``offer`` / ``counter`` / ``accept`` / ``close``).
        sid: Session id (first 8 chars are emitted).
        rnd: Round number.
        values: Attribute value per name (ints).
        attrs: Ordered attribute names to emit.
        prefix: Attribute -> single-letter prefix map (defaults to price=p/deadline=d).

    Returns:
        The encoded token bytes.

    Example::

        _encode_n("offer", "abcdef12", 0, {"price": 30, "deadline": 1})
    """
    pref = dict(_DEFAULT_ATTR_PREFIX if prefix is None else prefix)
    toks = [f"{_PREFIX}:{kind}:{sid[:8]}:r{rnd}"]
    toks.extend(f"{pref[a]}{values[a]}" for a in attrs)
    return ":".join(toks).encode()


def _decode_n(
    payload: bytes,
    attrs: Sequence[str] = _DEFAULT_ATTRS,
    prefix: Mapping[str, str] | None = None,
) -> tuple[str, str, int, dict[str, int]] | None:
    """N-attribute generalisation of :func:`_decode` (iter 4).

    Parses ``nego:<kind>:<sid8>:r<round>`` followed by exactly ``len(attrs)`` value
    tokens (one per attribute, in order). Returns ``(kind, sid8, round, values)`` with
    ``values`` keyed by attribute name, or ``None`` for profile / malformed tokens.
    For ``attrs == ("price", "deadline")`` this accepts exactly the tokens
    :func:`_decode` accepts and yields ``{"price": .., "deadline": ..}``.

    Example::

        _decode_n(b"nego:offer:abcdef12:r0:p30:d1")
    """
    pref = dict(_DEFAULT_ATTR_PREFIX if prefix is None else prefix)
    text = payload.decode("utf-8", errors="replace").rsplit("|sig:", 1)[0]
    parts = text.split(":")
    if len(parts) != 4 + len(attrs) or parts[0] != _PREFIX or parts[1] == "profile":
        return None
    kind, sid8, r_tok = parts[1], parts[2], parts[3]
    try:
        rnd = int(r_tok[1:]) if r_tok.startswith("r") else int(r_tok)
        values: dict[str, int] = {}
        for i, a in enumerate(attrs):
            tok = parts[4 + i]
            p = pref[a]
            values[a] = int(tok[len(p) :]) if tok.startswith(p) else int(tok)
    except (ValueError, IndexError):
        return None
    return kind, sid8, rnd, values


class NegotiatorAgent(StateMachineAgent):
    """An N-attribute negotiator that drives the negotiation plugin each round.

    The agent is plugin-agnostic: it records the opponent's offer over **every
    configured attribute**, asks the plugin to ``respond``, and -- when not accepting
    -- sends the plugin's ``counter_terms`` if the plugin produced a *new* counter,
    otherwise falls back to a deterministic per-attribute self-concession (each
    attribute either *steps* toward a meeting point or is *pinned* at a fixed value).
    Either way it discloses its private profile once.

    The attribute vector defaults to ``("price", "deadline")``; with that default and
    the canonical wire prefixes/field-keys every emitted token is byte-identical to
    the original two-attribute agent, so the golden trace is unchanged. Naming more
    attributes (with matching ``weights`` / ``bounds`` / ``direction`` / ``opening`` /
    ``concession`` entries) makes a genuine N-attribute negotiation travel on the wire.

    Example::

        agent = NegotiatorAgent(
            AgentId("buyer-0"), AgentId("seller-0"), is_initiator=True,
            attrs=("price", "deadline"),
            weights={"price": 0.7, "deadline": 0.3},
            bounds={"price": (30, 100), "deadline": (1, 30)},
            direction={"price": -1, "deadline": 1},
            opening={"price": 30, "deadline": 1},
            concession={"price": ("step", 80, 5), "deadline": ("pin", 30)},
            max_rounds=12,
        )
    """

    def __init__(
        self,
        agent_id: AgentId,
        partner: AgentId | None,
        is_initiator: bool,
        *,
        attrs: Sequence[str],
        weights: Mapping[str, float],
        bounds: Mapping[str, tuple[int, int]],
        direction: Mapping[str, int],
        opening: Mapping[str, int],
        concession: Mapping[str, tuple[Any, ...]],
        max_rounds: int,
        prefix: Mapping[str, str] | None = None,
        field_keys: Mapping[str, tuple[str, str, str, str]] | None = None,
    ) -> None:
        self._id = agent_id
        self._partner = partner
        self._is_initiator = is_initiator
        self._attrs: tuple[str, ...] = tuple(attrs)
        self._weights: dict[str, float] = {a: float(weights[a]) for a in self._attrs}
        self._bounds: dict[str, tuple[int, int]] = {
            a: (int(bounds[a][0]), int(bounds[a][1])) for a in self._attrs
        }
        self._direction: dict[str, int] = {a: int(direction[a]) for a in self._attrs}
        self._opening: dict[str, int] = {a: int(opening[a]) for a in self._attrs}
        self._concession: dict[str, tuple[Any, ...]] = {
            a: tuple(concession[a]) for a in self._attrs
        }
        self._max_rounds = max_rounds
        self._prefix: dict[str, str] = dict(_DEFAULT_ATTR_PREFIX if prefix is None else prefix)
        self._field_keys: dict[str, tuple[str, str, str, str]] = dict(
            _DEFAULT_PROFILE_FIELD_KEYS if field_keys is None else field_keys
        )
        self._sessions: dict[AgentId, Any] = {}
        self._my_vals: dict[AgentId, dict[str, int]] = {}
        self._profile_sent = False
        self._agreements = 0

    def _profile_payload(self, sid: str) -> bytes:
        return _encode_profile_n(
            sid,
            self._id,
            self._weights,
            self._bounds,
            self._direction,
            self._attrs,
            self._field_keys,
        )

    def _terms_from_vals(self, vals: Mapping[str, int]) -> Terms:
        """Pack a value-dict into ``Terms`` (``price`` in ``Money``, rest in ``conditions``).

        At ``("price", "deadline")`` this builds exactly ``Terms(price=Money(price),
        conditions={"deadline": deadline})`` -- byte-identical to the original.

        Example::

            terms = agent._terms_from_vals({"price": 30, "deadline": 1})
        """
        return Terms(
            price=Money(amount=int(vals["price"])),
            conditions={a: int(vals[a]) for a in self._attrs if a != "price"},
        )

    def _vals_from_terms(self, terms: Terms, fallback: Mapping[str, int]) -> dict[str, int]:
        """Read every configured attribute off ``terms`` (``conditions`` for non-price).

        Missing attributes fall back to ``fallback[attr]`` -- the original used the
        incoming offer's value as the deadline fallback, reproduced here per attribute.

        Example::

            vals = agent._vals_from_terms(counter_terms, opp_vals)
        """
        vals: dict[str, int] = {}
        for a in self._attrs:
            if a == "price":
                if terms.price is not None:
                    vals[a] = int(terms.price.amount)
                else:
                    vals[a] = int(fallback[a])
            else:
                vals[a] = int(terms.conditions.get(a, fallback[a]))
        return vals

    async def on_start(self, ctx: AgentContext) -> None:
        """Initiators open a session and send the opening offer; responders wait.

        Example::

            await agent.on_start(ctx)
        """
        if not self._is_initiator or self._partner is None:
            return
        neg = ctx.plugins.get("negotiation")
        if neg is None:
            return
        terms = self._terms_from_vals(self._opening)
        session = await neg.open(self._partner, terms)
        self._sessions[self._partner] = session
        self._my_vals[self._partner] = dict(self._opening)
        if not self._profile_sent:
            await ctx.send(self._partner, self._profile_payload(session.id))
            self._profile_sent = True
        await ctx.send(
            self._partner,
            _encode_n("offer", session.id, 0, self._opening, self._attrs, self._prefix),
        )

    def _self_concede(self, sender: AgentId, opp_vals: Mapping[str, int]) -> dict[str, int]:
        """Per-attribute deterministic concession: each attribute steps or is pinned.

        ``("step", floor, signed_step)`` walks the attribute from its last value toward
        ``floor`` (bounded by the opponent's current value, never overshooting it),
        reproducing the original price concession. ``("pin", value)`` holds the
        attribute at ``value`` (the original naive-deadline behaviour). At ``("price",
        "deadline")`` this returns exactly the original ``(price, deadline)`` concession.
        """
        last_vals = self._my_vals.get(sender, dict(self._opening))
        result: dict[str, int] = {}
        for a in self._attrs:
            policy = self._concession[a]
            if policy[0] == "pin":
                result[a] = int(policy[1])
                continue
            # ("step", floor, signed_step)
            floor = int(policy[1])
            step = int(policy[2])
            last = int(last_vals.get(a, self._opening[a]))
            opp = int(opp_vals[a])
            if step >= 0:
                nxt = min(floor, last + step)
                nxt = min(nxt, max(opp, last))
            else:
                nxt = max(floor, last + step)
                nxt = max(nxt, min(opp, last))
            result[a] = nxt
        return result

    async def on_message(self, ctx: AgentContext, sender: AgentId, payload: bytes) -> None:
        """Record the opponent offer, drive ``respond``, and accept or counter.

        Example::

            await agent.on_message(ctx, AgentId("buyer-0"), b"nego:offer:ab:r0:p30:d1")
        """
        decoded = _decode_n(payload, self._attrs, self._prefix)
        if decoded is None:
            return  # profile / malformed tokens carry no action
        kind, _sid8, rnd, opp_vals = decoded
        neg = ctx.plugins.get("negotiation")
        if neg is None:
            return

        session = self._sessions.get(sender)
        if session is None:
            session = await neg.open(sender, self._terms_from_vals(opp_vals))
            self._sessions[sender] = session
            self._my_vals[sender] = dict(self._opening)
            if not self._profile_sent:
                await ctx.send(sender, self._profile_payload(session.id))
                self._profile_sent = True

        if kind in ("close", "accept"):
            if kind == "accept":
                agreement = await neg.close(session)
                if agreement is not None:
                    self._agreements += 1
            return

        await neg.offer(session, self._terms_from_vals(opp_vals))
        resp = await neg.respond(session)
        next_round = rnd + 1
        if resp.accepted or next_round >= self._max_rounds:
            agreement = await neg.close(session)
            if agreement is not None:
                self._agreements += 1
            await ctx.send(
                sender,
                _encode_n("accept", session.id, next_round, opp_vals, self._attrs, self._prefix),
            )
            return

        ct = resp.counter_terms
        if ct is not None and ct.price is not None:
            counter_vals: dict[str, int] | None = self._vals_from_terms(ct, opp_vals)
        else:
            counter_vals = None
        if counter_vals is not None and counter_vals != dict(opp_vals):
            my_vals = counter_vals
        else:
            my_vals = self._self_concede(sender, opp_vals)

        await neg.offer(session, self._terms_from_vals(my_vals))
        self._my_vals[sender] = my_vals
        await ctx.send(
            sender,
            _encode_n("counter", session.id, next_round, my_vals, self._attrs, self._prefix),
        )


def _accepts_weights(neg_cls: type) -> bool:
    """Return True if ``neg_cls.__init__`` accepts a ``weights`` parameter.

    Example::

        _accepts_weights(ChainAimMultiAttributeNegotiation)  # True
    """
    try:
        params = inspect.signature(neg_cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "weights" in params


def _accepts_attributes(neg_cls: type) -> bool:
    """Return True if ``neg_cls.__init__`` accepts an ``attributes`` parameter.

    Used to pass the scenario's attribute vector only to plugins that understand it
    (the ChainAim multi-attribute plugin), never to the price-only baseline -- so the
    factory stays plugin-agnostic and the baseline construction is unchanged.

    Example::

        _accepts_attributes(ChainAimMultiAttributeNegotiation)  # True
    """
    try:
        params = inspect.signature(neg_cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "attributes" in params


def _accepts_grid_points(neg_cls: type) -> bool:
    """Return True if ``neg_cls.__init__`` accepts a ``grid_points`` parameter.

    Lets the factory pass the scenario's offer-grid resolution only to plugins that
    understand it (the ChainAim multi-attribute plugin), never to the price-only
    baseline -- so the factory stays plugin-agnostic and the baseline construction is
    unchanged.

    Example::

        _accepts_grid_points(ChainAimMultiAttributeNegotiation)  # True
    """
    try:
        params = inspect.signature(neg_cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "grid_points" in params


def _accepts_opponent_model(neg_cls: type) -> bool:
    """Return True if ``neg_cls.__init__`` accepts an ``opponent_model`` parameter.

    Lets the factory pass the scenario's Mechanism-B selector (heuristic/bayesian) plus
    its ``bayes_*`` tuning only to plugins that understand it (the ChainAim
    multi-attribute plugin), never to the price-only baseline -- so the factory stays
    plugin-agnostic and the baseline construction is unchanged.

    Example::

        _accepts_opponent_model(ChainAimMultiAttributeNegotiation)  # True
    """
    try:
        params = inspect.signature(neg_cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "opponent_model" in params


def _resolve_role_plugins(role_plugins: Any) -> dict[str, type]:
    """Resolve an optional ``{role: plugin_name}`` mapping to negotiation classes.

    Returns an empty mapping when ``role_plugins`` is falsy -- the caller then uses
    the single ``layers.negotiation`` class for every role, byte-identical to the
    pre-feature path. Each name is resolved through :class:`nest_core.plugins.PluginRegistry`
    (the same registry the runtime uses), so *any* registered negotiation plugin --
    a ``_BUILTINS`` entry such as ``alternating_offers`` or an entry-point plugin such
    as ``chainaim_neg_multi_pareto`` -- can be named per role, not only the one the
    ``layers.negotiation`` selector chose.

    Args:
        role_plugins: Optional mapping of role name -> registered plugin name, read
            verbatim from ``task.config.role_plugins`` (``None``/empty disables it).

    Returns:
        Mapping of role name -> resolved plugin class (empty when no block is given).

    Example::

        _resolve_role_plugins(
            {"buyer": "chainaim_neg_multi_pareto", "seller": "alternating_offers"}
        )
    """
    if not role_plugins:
        return {}
    from nest_core.plugins import PluginRegistry

    registry = PluginRegistry()
    resolved: dict[str, type] = {}
    for role, name in dict(role_plugins).items():
        cls = registry.resolve("negotiation", str(name))
        if not isinstance(cls, type):
            msg = f"role_plugins[{role!r}]={name!r} did not resolve to a plugin class"
            raise TypeError(msg)
        resolved[str(role)] = cls
    return resolved


def _pair_weights(
    rng: random.Random,
    cfg: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return ``(buyer_wp, buyer_wd, seller_wp, seller_wd)`` for one pair.

    Fixed when ``buyer_w_price`` / ``seller_w_price`` are configured; otherwise
    seeded from *rng* within ranges that keep buyers price-heavy and sellers
    deadline-heavy (so logrolling room always exists). Deadline weight is the
    complement so each agent's weights sum to ``1.0``.
    """
    if "buyer_w_price" in cfg:
        b_wp = float(cfg["buyer_w_price"])
    else:
        lo, hi = cfg.get("buyer_w_price_range", (0.65, 0.85))
        b_wp = rng.uniform(float(lo), float(hi))
    if "seller_w_price" in cfg:
        s_wp = float(cfg["seller_w_price"])
    else:
        lo, hi = cfg.get("seller_w_price_range", (0.15, 0.35))
        s_wp = rng.uniform(float(lo), float(hi))
    return (round(b_wp, 2), round(1.0 - b_wp, 2), round(s_wp, 2), round(1.0 - s_wp, 2))


def _renormalise_weights(
    w_price_base: float,
    extra: Mapping[str, float],
    base_attr: str = "deadline",
) -> dict[str, float]:
    """Combine a price-vs-``base_attr`` split with extra-attribute weights (sums to 1.0).

    ``extra`` weights are taken as-is (rounded to 2 dp); the remaining mass
    ``1 - sum(extra)`` is split between ``price`` and ``base_attr`` by the ratio
    ``w_price_base : 1 - w_price_base``, and ``base_attr`` absorbs the rounding residual
    so the returned map sums to exactly ``1.0`` to 2 dp.

    ``base_attr`` is the non-price base attribute that owns the price-complement mass.
    It defaults to ``"deadline"`` so every existing (price+deadline[+extra]) scenario
    keeps the same keys and values. A scenario whose attribute vector OMITS ``deadline``
    (e.g. price+quantity) passes its present base attribute instead, so no phantom
    ``deadline`` key is injected and the disclosed weights still sum to ``1.0``.

    Example::

        w = _renormalise_weights(0.8, {"quality": 0.2})  # ~{price:.64, quality:.2, deadline:.16}
        w = _renormalise_weights(0.75, {}, "quantity")   # ~{price:.75, quantity:.25}
    """
    rounded_extra = {a: round(v, 2) for a, v in extra.items()}
    remaining = max(0.0, 1.0 - sum(rounded_extra.values()))
    price = round(remaining * w_price_base, 2)
    base = round(1.0 - price - sum(rounded_extra.values()), 2)
    return {"price": price, **rounded_extra, base_attr: base}


def _pair_weights_n(
    rng: random.Random,
    cfg: dict[str, Any],
    attrs: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(buyer_weights, seller_weights)`` per-attribute maps summing to 1.0.

    The price-vs-deadline split is drawn by the unchanged :func:`_pair_weights` (same
    RNG draws, same order), so two-attribute scenarios reproduce today's weights
    byte-for-byte. Any *extra* attribute draws its own buyer/seller weight from a
    configurable ``<attr>_w_*_range`` **after** those base draws -- so adding an
    attribute never perturbs the price/deadline draws -- and price+deadline are then
    renormalised (see :func:`_renormalise_weights`).

    Example::

        b, s = _pair_weights_n(rng, cfg, ("price", "deadline", "quality"))
    """
    b_wp, b_wd, s_wp, s_wd = _pair_weights(rng, cfg)
    non_price = [a for a in attrs if a != "price"]
    # The base attribute owns the price-complement mass: ``deadline`` when present
    # (keeps every existing scenario byte-identical -- same key, same draws), else the
    # first non-price attribute in the vector (e.g. ``quantity`` for price+quantity).
    # This branch only diverges from the prior revision when ``deadline`` is ABSENT,
    # so no existing trace moves.
    if "deadline" in non_price:
        base_attr = "deadline"
    elif non_price:
        base_attr = non_price[0]
    else:
        base_attr = "deadline"
    extras = [a for a in non_price if a != base_attr]
    if not extras:
        # Two-attribute case (price + base_attr). For ("price", "deadline") this is
        # byte-identical to the original 4-tuple return; for ("price", "quantity") the
        # complement goes to quantity (no phantom deadline) and weights still sum to 1.0.
        return (
            {"price": b_wp, base_attr: b_wd},
            {"price": s_wp, base_attr: s_wd},
        )
    b_extra: dict[str, float] = {}
    s_extra: dict[str, float] = {}
    for a in extras:
        blo, bhi = cfg.get(f"buyer_w_{a}_range", (0.10, 0.25))
        slo, shi = cfg.get(f"seller_w_{a}_range", (0.10, 0.25))
        b_extra[a] = rng.uniform(float(blo), float(bhi))
        s_extra[a] = rng.uniform(float(slo), float(shi))
    return (
        _renormalise_weights(b_wp, b_extra, base_attr),
        _renormalise_weights(s_wp, s_extra, base_attr),
    )


def negotiation_multi_factory(
    config: ScenarioConfig,
    plugins: dict[str, Any],
) -> dict[AgentId, StateMachineAgent]:
    """Create paired buyer/seller multi-attribute negotiators.

    All tunables are read from ``config.task.config`` with defaults, so behaviour is
    YAML-driven and nothing is hardcoded:

    - ``price_min`` / ``price_max`` (int): shared price bounds (default ``30`` / ``100``).
    - ``deadline_min`` / ``deadline_max`` (int): shared deadline bounds (default ``1`` / ``30``).
    - ``buyer_open_price`` (int): buyer's opening price (default = ``price_min``).
    - ``buyer_open_deadline`` (int): opening deadline sweetener (default ``deadline_min``).
    - ``seller_anchor`` (int): seller's first-counter anchor price (default ``price_max - 5``).
    - ``buyer_target`` / ``seller_target`` (int): naive price meeting points (``80`` / ``50``).
    - ``naive_deadline`` (int): deadline on self-conceded offers (default mid-range).
    - ``price_step`` (int): naive concession size per round (default ``5``).
    - ``patience`` (float): plugin patience discount (default ``0.9``).
    - ``max_rounds`` (int): per-session safety cap (default ``12``).
    - ``buyer_w_price`` / ``seller_w_price`` (float): fixed weights, else seeded from ranges.
    - ``grid_points`` (int): plugin offer-grid resolution per axis (default ``15`` = plugin
      default; passed only to plugins whose constructor accepts it).
    - ``role_plugins`` (mapping, optional): ``{buyer: <name>, seller: <name>}`` to assign a
      negotiation plugin per role (asymmetric matchup). Absent -> the single
      ``layers.negotiation`` class for every agent (unchanged behaviour).
    - ``attributes`` (list, optional): the ordered attribute vector (default
      ``["price", "deadline"]``). Extra attributes read generic ``<attr>_min`` /
      ``<attr>_max`` / ``<attr>_prefix`` / ``<attr>_dir`` (via ``buyer_dir`` / ``seller_dir``
      overlays) / ``<attr>_open_buyer`` / ``<attr>_open_seller`` / ``<attr>_concede``
      overlays; with the default the build is byte-identical to the 2-attribute factory.

    Pairs are deterministic: ``buyer-i`` negotiates with ``seller-i``.

    Example::

        agents = negotiation_multi_factory(config, plugins)
    """
    cfg = config.task.config
    price_lo = int(cfg.get("price_min", 30))
    price_hi = int(cfg.get("price_max", 100))
    deadline_lo = int(cfg.get("deadline_min", 1))
    deadline_hi = int(cfg.get("deadline_max", 30))
    buyer_open_price = int(cfg.get("buyer_open_price", price_lo))
    buyer_open_deadline = int(cfg.get("buyer_open_deadline", deadline_lo))
    seller_anchor = int(cfg.get("seller_anchor", price_hi - 5))
    buyer_target = int(cfg.get("buyer_target", 80))
    seller_target = int(cfg.get("seller_target", 50))
    naive_deadline = int(cfg.get("naive_deadline", (deadline_lo + deadline_hi) // 2))
    price_step = int(cfg.get("price_step", 5))
    patience = float(cfg.get("patience", 0.9))
    max_rounds = int(cfg.get("max_rounds", 12))
    # Plugin offer-grid resolution: how many points per attribute axis the frontier
    # walk sweeps. Defaults to the plugin's own 15 so 2-attribute scenarios stay
    # byte-identical (passing 15 explicitly == the plugin default). A scenario may
    # raise it (e.g. grid_points: 71 over price [30,100] -> price STEP 1) so the agent
    # can propose the sub-grid points the step-1 frontier validator scores against.
    grid_points = int(cfg.get("grid_points", 15))
    # Mechanism B selector (opponent-weight inference), threaded only to plugins that
    # accept it. Defaults keep the heuristic (scored) model; a scenario may set
    # ``opponent_model: bayesian`` plus optional ``bayes_grid_points`` / ``bayes_beta``
    # / ``bayes_warmup`` tuning.
    opponent_model = str(cfg.get("opponent_model", "heuristic"))
    bayes_grid_points = int(cfg.get("bayes_grid_points", 21))
    bayes_beta = float(cfg.get("bayes_beta", 4.0))
    bayes_warmup = int(cfg.get("bayes_warmup", 0))

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

    pair_count = min(buyer_count, seller_count)
    seller_ids = [AgentId(f"seller-{i}") for i in range(seller_count)]
    buyer_ids = [AgentId(f"buyer-{i}") for i in range(buyer_count)]

    attrs = tuple(cfg.get("attributes", ["price", "deadline"]))
    if not attrs:
        attrs = ("price", "deadline")

    # Per-attribute maps. The two canonical attributes keep their hard-coded
    # bounds/direction/prefix/field-keys/opening/concession so the 2-attribute
    # scenarios are byte-identical; any extra attribute is read generically from
    # ``<attr>_min`` / ``<attr>_max`` / ``<attr>_prefix`` / ``buyer_dir`` / ``seller_dir``
    # / ``<attr>_open_*`` / ``<attr>_concede`` overlays, defaulting sensibly.
    bounds: dict[str, tuple[int, int]] = {
        "price": (price_lo, price_hi),
        "deadline": (deadline_lo, deadline_hi),
    }
    buyer_dir: dict[str, int] = {"price": -1, "deadline": 1}  # buyer: cheap good, long good
    seller_dir: dict[str, int] = {"price": 1, "deadline": -1}  # seller: dear good, short good
    prefix: dict[str, str] = dict(_DEFAULT_ATTR_PREFIX)
    field_keys: dict[str, tuple[str, str, str, str]] = dict(_DEFAULT_PROFILE_FIELD_KEYS)
    buyer_open: dict[str, int] = {
        "price": buyer_open_price,
        "deadline": buyer_open_deadline,
    }
    seller_open: dict[str, int] = {"price": seller_anchor, "deadline": naive_deadline}
    buyer_concession: dict[str, tuple[Any, ...]] = {
        "price": ("step", buyer_target, price_step),
        "deadline": ("pin", naive_deadline),
    }
    seller_concession: dict[str, tuple[Any, ...]] = {
        "price": ("step", seller_target, -price_step),
        "deadline": ("pin", naive_deadline),
    }
    cfg_buyer_dir = dict(cfg.get("buyer_dir", {}))
    cfg_seller_dir = dict(cfg.get("seller_dir", {}))
    for a in attrs:
        if a not in bounds:
            bounds[a] = (int(cfg.get(f"{a}_min", 0)), int(cfg.get(f"{a}_max", 0)))
        if a in cfg_buyer_dir:
            buyer_dir[a] = int(cfg_buyer_dir[a])
        elif a not in buyer_dir:
            buyer_dir[a] = 1
        if a in cfg_seller_dir:
            seller_dir[a] = int(cfg_seller_dir[a])
        elif a not in seller_dir:
            seller_dir[a] = -1
        if a not in prefix or a not in field_keys:
            p = str(cfg.get(f"{a}_prefix", a[0]))
            prefix[a] = p
            field_keys[a] = (f"w{p}", f"{p}min", f"{p}max", f"dir_{p}")
        if a not in buyer_open:
            buyer_open[a] = int(cfg.get(f"{a}_open_buyer", bounds[a][0]))
        if a not in seller_open:
            seller_open[a] = int(cfg.get(f"{a}_open_seller", bounds[a][1]))
        if a not in buyer_concession or a not in seller_concession:
            if str(cfg.get(f"{a}_concede", "pin")) == "step":
                b_floor = int(cfg.get(f"{a}_target_buyer", bounds[a][0]))
                s_floor = int(cfg.get(f"{a}_target_seller", bounds[a][1]))
                a_step = int(cfg.get(f"{a}_step", 1))
                buyer_concession[a] = ("step", b_floor, a_step)
                seller_concession[a] = ("step", s_floor, -a_step)
            else:
                buyer_concession[a] = ("pin", buyer_open[a])
                seller_concession[a] = ("pin", seller_open[a])

    rng = random.Random(config.seed)
    per_pair = [_pair_weights_n(rng, cfg, attrs) for _ in range(pair_count)]

    # Per-role strategy assignment (additive, opt-in). When ``role_plugins`` is
    # present each role is built from its own resolved class -- an asymmetric
    # (cross-brain) matchup; when absent both roles use the single layer-selected
    # class, so this path stays byte-identical to the pre-feature factory.
    layer_cls = plugins.get("negotiation") if plugins else None
    role_cls = _resolve_role_plugins(cfg.get("role_plugins"))
    buyer_cls = role_cls.get("buyer", layer_cls)
    seller_cls = role_cls.get("seller", layer_cls)
    buyer_takes_weights = isinstance(buyer_cls, type) and _accepts_weights(buyer_cls)
    seller_takes_weights = isinstance(seller_cls, type) and _accepts_weights(seller_cls)
    buyer_takes_attributes = isinstance(buyer_cls, type) and _accepts_attributes(buyer_cls)
    seller_takes_attributes = isinstance(seller_cls, type) and _accepts_attributes(seller_cls)
    buyer_takes_grid = isinstance(buyer_cls, type) and _accepts_grid_points(buyer_cls)
    seller_takes_grid = isinstance(seller_cls, type) and _accepts_grid_points(seller_cls)
    buyer_takes_opp = isinstance(buyer_cls, type) and _accepts_opponent_model(buyer_cls)
    seller_takes_opp = isinstance(seller_cls, type) and _accepts_opponent_model(seller_cls)

    agent_plugins: dict[AgentId, dict[str, Any]] = {}
    if plugins:
        agent_plugins = plugins.setdefault("_agent_plugins", {})

    agents: dict[AgentId, StateMachineAgent] = {}

    for i in range(pair_count):
        buyer_weights, seller_weights = per_pair[i]
        buyer = buyer_ids[i]
        seller = seller_ids[i]

        if plugins:
            role_specs = (
                (
                    buyer,
                    buyer_cls,
                    buyer_takes_weights,
                    buyer_takes_attributes,
                    buyer_takes_grid,
                    buyer_takes_opp,
                    buyer_weights,
                    buyer_dir,
                ),
                (
                    seller,
                    seller_cls,
                    seller_takes_weights,
                    seller_takes_attributes,
                    seller_takes_grid,
                    seller_takes_opp,
                    seller_weights,
                    seller_dir,
                ),
            )
            for (
                aid,
                cls,
                takes_weights,
                takes_attrs,
                takes_grid,
                takes_opp,
                weights,
                direction,
            ) in role_specs:
                if not isinstance(cls, type):
                    continue
                if takes_weights:
                    kwargs: dict[str, Any] = {
                        "weights": weights,
                        "bounds": bounds,
                        "direction": direction,
                        "patience": patience,
                    }
                    if takes_attrs:
                        kwargs["attributes"] = attrs
                    if takes_grid:
                        kwargs["grid_points"] = grid_points
                    if takes_opp:
                        kwargs["opponent_model"] = opponent_model
                        kwargs["bayes_grid_points"] = bayes_grid_points
                        kwargs["bayes_beta"] = bayes_beta
                        kwargs["bayes_warmup"] = bayes_warmup
                    instance = cls(aid, **kwargs)
                else:
                    instance = cls(aid, patience=patience)
                agent_plugins.setdefault(aid, {})["negotiation"] = instance

        agents[buyer] = NegotiatorAgent(
            buyer,
            seller,
            is_initiator=True,
            attrs=attrs,
            weights=buyer_weights,
            bounds=bounds,
            direction=buyer_dir,
            opening=buyer_open,
            concession=buyer_concession,
            max_rounds=max_rounds,
            prefix=prefix,
            field_keys=field_keys,
        )
        agents[seller] = NegotiatorAgent(
            seller,
            None,
            is_initiator=False,
            attrs=attrs,
            weights=seller_weights,
            bounds=bounds,
            direction=seller_dir,
            opening=seller_open,
            concession=seller_concession,
            max_rounds=max_rounds,
            prefix=prefix,
            field_keys=field_keys,
        )

    if plugins:
        plugins.pop("negotiation", None)

    return agents
