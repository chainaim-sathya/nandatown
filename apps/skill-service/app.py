# SPDX-License-Identifier: Apache-2.0
"""HTTP service that serves the Prava skill document and runs the real plugin.

Routes only. All payment behaviour lives in ``runner.py``, which calls the
actual ``PravaPayments`` plugin from ``nest_plugins_reference``.

Run locally::

    python -m uvicorn app:app --app-dir apps/skill-service --port 8000

Then::

    curl localhost:8000/health
    curl localhost:8000/skill.md
    curl "localhost:8000/demo/purchase?case=drift"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from runner import CASES, CaseError, run_purchase

SKILL_PATH = Path(__file__).resolve().parent / "skill.md"

app = FastAPI(
    title="ChainAIM Nanda Prava-VISA-Payment",
    description=(
        "An agent pays a merchant on Visa rails through the Prava mandate API, "
        "and refuses the charge if the delivered item is not the one that was "
        "agreed. Demo runs use the plugin's simulated transport: no keys, no "
        "network, no real money."
    ),
    version="1.0",
)


class PurchaseRequest(BaseModel):
    """Body for ``POST /purchase``.

    Every field has a default, so an empty body runs the good case.

    Example::

        PurchaseRequest(case="drift", days_since_delivery=40)
    """

    case: str = Field(default="good", description="good | drift | overcap")
    days_since_delivery: int = Field(default=5, ge=0)
    payment_ref: str = Field(default="prava-demo-0", min_length=1, max_length=255)
    mandate_id: str = Field(default="mdt_simulated_demo", min_length=1)
    cap_minor: int = Field(default=100_000, gt=0)
    fail_closed: bool = Field(default=True)
    mode: str = Field(default="simulated", description="simulated | live")


@app.get("/health", summary="Liveness check")
async def health() -> dict[str, str]:
    """Return a fixed liveness payload.

    Example::

        curl localhost:8000/health
    """
    return {"status": "ok", "service": "chainaim-nanda-prava-visa-payment"}


@app.get("/skill.md", response_class=PlainTextResponse, summary="The skill document")
async def skill_document() -> PlainTextResponse:
    """Serve ``skill.md`` verbatim as markdown.

    Example::

        curl localhost:8000/skill.md

    Raises:
        HTTPException: 404 if the document is missing from the deployment.
    """
    if not SKILL_PATH.is_file():
        raise HTTPException(status_code=404, detail="skill.md is not present in this deployment")
    return PlainTextResponse(
        SKILL_PATH.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


@app.get("/cases", summary="List the demo cases")
async def cases() -> dict[str, Any]:
    """Describe each demo case in plain language.

    Example::

        curl localhost:8000/cases
    """
    return {
        "cases": [
            {"case": spec.name, "summary": spec.summary, "snapshot": spec.snapshot}
            for spec in CASES.values()
        ]
    }


@app.get("/demo/purchase", summary="Run one purchase against the real plugin")
async def demo_purchase(
    case: str = Query(default="good", description="good | drift | overcap"),
    days_since_delivery: int = Query(default=5, ge=0),
) -> dict[str, Any]:
    """Run the full cycle in simulated mode and return what happened.

    Always simulated. This route never accepts a mode, so a public URL cannot
    be made to charge a real card.

    Example::

        curl "localhost:8000/demo/purchase?case=drift"

    Raises:
        HTTPException: 400 for an unknown case name.
    """
    try:
        return await run_purchase(case=case, days_since_delivery=days_since_delivery)
    except CaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/purchase", summary="Run a purchase with your own settings")
async def purchase(request: PurchaseRequest) -> dict[str, Any]:
    """Run the cycle with caller-supplied settings.

    ``mode="live"`` is refused unless the deployment sets ``ALLOW_LIVE=1`` and
    ``PRAVA_API_KEY``.

    Example::

        curl -X POST localhost:8000/purchase -H "Content-Type: application/json" \
             -d '{"case":"good","days_since_delivery":40}'

    Raises:
        HTTPException: 400 for an unknown case, 403 when live mode is refused.
    """
    try:
        return await run_purchase(
            case=request.case,
            mode=request.mode,
            mandate_id=request.mandate_id,
            payment_ref=request.payment_ref,
            cap_minor=request.cap_minor,
            fail_closed=request.fail_closed,
            days_since_delivery=request.days_since_delivery,
        )
    except CaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/", summary="Where to start")
async def root() -> dict[str, Any]:
    """Point a caller at the document and the two demo URLs.

    Example::

        curl localhost:8000/
    """
    return {
        "skill": "/skill.md",
        "health": "/health",
        "cases": "/cases",
        "demo_good": "/demo/purchase?case=good",
        "demo_drift": "/demo/purchase?case=drift",
        "demo_overcap": "/demo/purchase?case=overcap",
        "docs": "/docs",
    }
