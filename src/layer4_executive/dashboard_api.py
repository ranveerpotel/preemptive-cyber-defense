"""
Layer 4 — Executive Intelligence Layer: FastAPI Dashboard

Real-time REST API surfacing governance scores, forecasts, active remediation
workflows, HITL approval queue, and audit trails for security leadership.

Unlike GRC's monthly static reports, this delivers continuous queryable visibility.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.common.models import (
    ActionStatus, GovernanceControl, GMSResult, RemediationAction, RiskForecast
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Preemptive Cyber Defense — Executive Dashboard",
    description="Real-time governance visibility. Paper: Potel (IJEETR 2021).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class GovernanceScoreResponse(BaseModel):
    score: float
    lower_bound: float
    upper_bound: float
    std_dev: float
    confidence: float
    timestamp: str
    trend: str
    below_threshold: bool
    control_contributions: Dict[str, float]


class ForecastResponse(BaseModel):
    current_gms: float
    forecast_7d: List[float]
    trend: str
    top_risk_drivers: List[str]
    generated_at: str


class ActionResponse(BaseModel):
    action_id: str
    action_type: str
    target_entity_id: str
    impact: str
    status: str
    confidence: float
    estimated_risk_reduction: float
    actual_risk_reduction: Optional[float]
    shap_rationale: Optional[str]
    cbf_satisfied: bool
    created_at: str


class HITLApprovalRequest(BaseModel):
    approver: str
    notes: str = ""


class HITLRejectionRequest(BaseModel):
    rejector: str
    reason: str = ""


class AttackGraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    crown_jewel_risk_paths: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Dependency injection — in production, these come from DI container
# ---------------------------------------------------------------------------
# These are set by the application startup process.

_gms_scorer = None
_risk_forecaster = None
_remediation_agent = None
_attack_graph = None
_controls: List[GovernanceControl] = []
_current_gms: Optional[GMSResult] = None
_current_forecast: Optional[RiskForecast] = None


def configure(gms_scorer, risk_forecaster, remediation_agent, attack_graph, controls):
    """Wire up all service dependencies at application startup."""
    global _gms_scorer, _risk_forecaster, _remediation_agent, _attack_graph, _controls
    _gms_scorer = gms_scorer
    _risk_forecaster = risk_forecaster
    _remediation_agent = remediation_agent
    _attack_graph = attack_graph
    _controls = controls


def _require_gms() -> GMSResult:
    if _current_gms is None:
        raise HTTPException(status_code=503, detail="Governance scoring not yet initialized.")
    return _current_gms


# ---------------------------------------------------------------------------
# Endpoints — Layer 4 Executive Intelligence
# ---------------------------------------------------------------------------

@app.get("/governance/score", response_model=GovernanceScoreResponse, tags=["Governance"])
def get_governance_score():
    """
    Current Governance Maturity Score with confidence interval.
    Refreshed every 60 seconds from the Governance Engine.
    """
    gms = _require_gms()
    trend = _gms_scorer.recent_trend() if _gms_scorer else "stable"
    below = _gms_scorer.gms_below_threshold(gms) if _gms_scorer else False
    return GovernanceScoreResponse(
        score=gms.score,
        lower_bound=gms.lower_bound,
        upper_bound=gms.upper_bound,
        std_dev=gms.std_dev,
        confidence=gms.confidence,
        timestamp=gms.timestamp.isoformat(),
        trend=trend,
        below_threshold=below,
        control_contributions=gms.control_contributions,
    )


@app.get("/governance/forecast", response_model=ForecastResponse, tags=["Governance"])
def get_governance_forecast():
    """7-day forward GMS projection with trend classification."""
    global _current_forecast
    if _current_forecast is None and _risk_forecaster and _current_gms:
        _current_forecast = _risk_forecaster.forecast(_controls, _current_gms)
    if _current_forecast is None:
        raise HTTPException(status_code=503, detail="Forecast not yet available.")
    f = _current_forecast
    return ForecastResponse(
        current_gms=f.current_gms,
        forecast_7d=f.forecast_7d,
        trend=f.trend,
        top_risk_drivers=f.top_risk_drivers,
        generated_at=f.generated_at.isoformat(),
    )


@app.get("/risks/active", tags=["Risk Intelligence"])
def get_active_risks():
    """
    Ranked active risk indicators from the Security Knowledge Graph
    and attack graph analysis.
    """
    gms = _require_gms()
    risks = []
    for ctrl_id, contribution in sorted(
        gms.control_contributions.items(), key=lambda x: x[1], reverse=True
    ):
        ctrl = next((c for c in _controls if c.control_id == ctrl_id), None)
        if ctrl:
            risks.append({
                "control_id": ctrl_id,
                "domain": ctrl.domain,
                "effectiveness": ctrl.effectiveness,
                "exposure": ctrl.exposure,
                "risk_contribution": contribution,
                "remediation_priority": 1.0 - ctrl.effectiveness,
            })
    return {"risks": risks, "total": len(risks)}


@app.get("/remediation/active", response_model=List[ActionResponse], tags=["Remediation"])
def get_active_remediation():
    """In-progress and recently completed autonomous remediation actions."""
    if _remediation_agent is None:
        return []
    log = _remediation_agent.audit_log()
    return [
        ActionResponse(
            action_id=entry["action_id"],
            action_type=entry["action_type"],
            target_entity_id=entry.get("target", ""),
            impact=entry["impact"],
            status=entry.get("routed_to", "unknown"),
            confidence=entry.get("confidence", 0.0),
            estimated_risk_reduction=0.0,
            actual_risk_reduction=None,
            shap_rationale=entry.get("shap_rationale"),
            cbf_satisfied=entry.get("cbf_satisfied", False),
            created_at=entry.get("timestamp", datetime.utcnow().isoformat()),
        )
        for entry in log[-50:]  # last 50 actions
    ]


@app.get("/remediation/hitl", tags=["Remediation"])
def get_hitl_queue():
    """Pending Human-in-the-Loop approval queue."""
    if _remediation_agent is None:
        return {"pending": [], "count": 0}
    queue = _remediation_agent.hitl_queue()
    pending = [
        {
            "action_id": a.action_id,
            "action_type": a.action_type,
            "target_entity_id": a.target_entity_id,
            "impact": a.impact.value,
            "confidence": a.confidence,
            "shap_rationale": a.shap_rationale,
            "created_at": a.created_at.isoformat(),
            "elapsed_seconds": (datetime.utcnow() - a.created_at).total_seconds(),
        }
        for a in queue.pending_actions()
    ]
    return {"pending": pending, "count": len(pending)}


@app.post("/remediation/approve/{action_id}", tags=["Remediation"])
def approve_hitl_action(action_id: str, request: HITLApprovalRequest):
    """Human approves a HITL-queued action."""
    if _remediation_agent is None:
        raise HTTPException(status_code=503, detail="Remediation agent not initialized.")
    action = _remediation_agent.hitl_queue().approve(action_id, request.approver, request.notes)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action {action_id} not found in HITL queue.")
    return {"approved": True, "action_id": action_id, "status": action.status.value}


@app.post("/remediation/reject/{action_id}", tags=["Remediation"])
def reject_hitl_action(action_id: str, request: HITLRejectionRequest):
    """Human rejects a HITL-queued action."""
    if _remediation_agent is None:
        raise HTTPException(status_code=503, detail="Remediation agent not initialized.")
    action = _remediation_agent.hitl_queue().reject(action_id, request.rejector, request.reason)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action {action_id} not found in HITL queue.")
    return {"rejected": True, "action_id": action_id, "status": action.status.value}


@app.get("/audit/trail", tags=["Audit"])
def get_audit_trail(limit: int = 100):
    """
    Immutable audit trail of all autonomous and HITL actions with SHAP rationales.
    Supports regulatory reporting and accountability.
    """
    if _remediation_agent is None:
        return {"entries": [], "total": 0}
    log = _remediation_agent.audit_log()
    return {"entries": log[-limit:], "total": len(log)}


@app.get("/attack-graph", response_model=AttackGraphResponse, tags=["Risk Intelligence"])
def get_attack_graph():
    """Current attack graph with asset nodes, edges, and path probabilities."""
    if _attack_graph is None:
        raise HTTPException(status_code=503, detail="Attack graph not initialized.")
    graph_data = _attack_graph.to_dict()
    return AttackGraphResponse(
        nodes=graph_data["nodes"],
        edges=graph_data["edges"],
        crown_jewel_risk_paths=[],  # populated when crown jewel defined
    )


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "gms_initialized": _current_gms is not None,
        "controls_loaded": len(_controls),
        "timestamp": datetime.utcnow().isoformat(),
    }
