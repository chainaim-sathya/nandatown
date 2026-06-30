# SPDX-License-Identifier: Apache-2.0
"""Portable "best-deal" certificate for multi-attribute negotiation traces.

Serialises the feasible-frontier verdict already produced by the negotiation validators
into a structured, re-checkable artifact: per agreed pair, the settled deal, both parties'
utilities, and whether the settlement sits on the reconstructed feasible Pareto frontier,
plus the four negotiation verdicts. It is derived entirely from the public validator API
(:func:`nest_core.validators.validate_events` and
:func:`nest_core.validators.negotiation_metrics`), so it adds no new guarantee -- it makes
the existing one portable, and modifies nothing in ``validators.py``.

Generic over N attributes: it reports whatever the trace discloses (demonstrated at 2, 3,
and 4 attributes). Deterministic: floats are those already rounded by the metrics, the
schema is fixed, and there is no wall-clock, so :func:`certificate_json` is byte-reproducible
for a given trace.

Example::

    from nest_core.negotiation_certificate import certificate_json, negotiation_certificate

    cert = negotiation_certificate("traces/multi_attribute_market_4attr_....jsonl")
    Path("cert.json").write_text(certificate_json(cert))
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nest_core.validators import negotiation_metrics, validate_events

_ON_FRONTIER_EPS = 1e-9
_CERTIFICATE_VERSION = 1


def _read_events(trace_path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL trace events (mirrors the validators' own loader)."""
    events: list[dict[str, Any]] = []
    with Path(trace_path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                events.append(json.loads(line))
    return events


def _on_frontier(pareto_distance: float | None, eps: float) -> bool:
    """True when the agreement sits on the feasible frontier (distance ~ 0)."""
    return pareto_distance is not None and pareto_distance <= eps


def _trace_sha256(trace_path: str | Path) -> str:
    """Return the SHA-256 hex digest of the trace file's raw bytes.

    Content-addressed, not path-addressed: identical trace bytes yield an identical
    digest regardless of where the file lives, so the digest is a stable, re-checkable
    identifier for the trace. It doubles as the settlement ``task_id`` when this
    certificate feeds an outcome-verified settlement.

    Example::

        digest = _trace_sha256("traces/run.jsonl")
    """
    return hashlib.sha256(Path(trace_path).read_bytes()).hexdigest()


def negotiation_certificate(
    trace_path: str | Path,
    *,
    eps: float = _ON_FRONTIER_EPS,
    labels: dict[str, str] | None = None,
    scenario: str | None = None,
    grid_step: int = 1,
) -> dict[str, Any]:
    """Build a best-deal certificate for a negotiation trace.

    Reconstructs, per agreed pair: the deal, both parties' utilities, and whether the
    settlement is on the feasible Pareto frontier -- plus the four negotiation verdicts.
    ``labels`` optionally renames the wire attribute keys (e.g. ``{"n": "quantity"}``) for
    presentation; by default the certificate reports the keys as disclosed on the wire
    (full names at 2 attributes, single-char prefixes beyond).

    ``scenario`` optionally labels the run in the ``provenance`` block; when omitted it
    defaults to the trace file's stem. The provenance block also carries
    ``trace_sha256`` -- the SHA-256 of the trace file's raw bytes -- which both
    tamper-evidences the certificate and serves as the settlement ``task_id`` binding.
    Each agreed pair additionally reports its ``nash_point`` (the feasible Nash
    bargaining coordinates) and ``nash_product``, and the top level reports the
    ``on_frontier_eps`` tolerance used to derive each pair's ``on_frontier`` flag.
    ``grid_step`` (default ``1`` = exhaustive) coarsens only the verdict-neutral reporting
    metrics (Nash/Pareto/social-welfare) for speed on fine-grained domains; the four
    negotiation verdicts are always computed exhaustively and are unaffected by it.

    Example::

        cert = negotiation_certificate(trace, labels={"n": "quantity", "q": "quality"})
        cert["verdicts"]["negotiation_frontier_efficient"]["passed"]
    """
    events = _read_events(trace_path)

    verdicts = {
        result.name: {"passed": bool(result.passed), "detail": result.detail}
        for result in validate_events(events, "negotiation")
    }

    metrics = negotiation_metrics(events, grid_step=grid_step)
    label = labels or {}

    def _rename(mapping: dict[str, Any]) -> dict[str, Any]:
        return {label.get(key, key): value for key, value in mapping.items()}

    pairs: list[dict[str, Any]] = []
    on_frontier_count = 0
    for name, pair_metrics in metrics["pairs"].items():
        parties = name.split("<->")
        utilities = pair_metrics.get("utilities") or [None, None]
        pareto_distance = pair_metrics.get("pareto_distance")
        is_on = _on_frontier(pareto_distance, eps)
        on_frontier_count += 1 if is_on else 0
        deal = _rename({k: int(v) for k, v in pair_metrics.get("agreement", {}).items()})
        nash_point_raw = pair_metrics.get("nash_point")
        nash_point = (
            _rename({k: int(v) for k, v in nash_point_raw.items()})
            if nash_point_raw is not None
            else None
        )
        pairs.append(
            {
                "pair": name,
                "parties": parties,
                "deal": deal,
                "utility": {parties[0]: utilities[0], parties[1]: utilities[1]},
                "on_frontier": is_on,
                "pareto_distance": pareto_distance,
                "social_welfare": pair_metrics.get("social_welfare"),
                "nash_distance": pair_metrics.get("nash_distance"),
                "nash_point": nash_point,
                "nash_product": pair_metrics.get("nash_product"),
            }
        )

    attributes = list(pairs[0]["deal"].keys()) if pairs else []
    aggregate = metrics["aggregate"]

    return {
        "kind": "negotiation_best_deal_certificate",
        "version": _CERTIFICATE_VERSION,
        "check": "feasible_frontier_reconstruction",
        "attributes": attributes,
        "on_frontier_eps": eps,
        "provenance": {
            "trace_sha256": _trace_sha256(trace_path),
            "scenario": scenario if scenario is not None else Path(trace_path).stem,
        },
        "verdicts": verdicts,
        "pairs": pairs,
        "summary": {
            "pairs_scored": aggregate["pairs_scored"],
            "on_frontier": on_frontier_count,
            "breakdowns": aggregate["breakdowns"],
            "mean_pareto_distance": aggregate["mean_pareto_distance"],
        },
    }


def certificate_json(certificate: dict[str, Any]) -> str:
    """Serialise a certificate to deterministic, byte-reproducible JSON."""
    return json.dumps(certificate, indent=2, sort_keys=True)
