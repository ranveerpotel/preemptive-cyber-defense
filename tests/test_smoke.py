"""
Smoke tests — validates core algorithmic components without external dependencies.
Run: python -m pytest tests/test_smoke.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest
from datetime import datetime

from src.common.models import (
    EventSource, GovernanceControl, OCSFEvent, Entity, EntityType,
    Relation, RelationType, RemediationAction, ActionImpact
)
from src.common.config import DEFAULT_CONFIG
from src.layer1_sdf.normalizer import SDFNormalizer, normalize
from src.layer1_sdf.entity_correlator import EntityCorrelator
from src.layer1_sdf.temporal_tracker import TemporalTracker
from src.layer2_governance.gms_scorer import GMSScorer
from src.layer2_governance.risk_forecaster import RiskForecaster
from src.layer3_remediation.cbf import ControlBarrierFunctions
from src.layer3_remediation.hitl_queue import HITLQueue
from src.knowledge_graph.attack_graph import AttackGraph
from src.knowledge_graph.skg import SecurityKnowledgeGraph


# -----------------------------------------------------------------------
# Layer 1: SDF Normalizer
# -----------------------------------------------------------------------

def test_crowdstrike_normalization():
    raw = {
        "ComputerName": "laptop-04",
        "UserName": "alice",
        "LocalIP": "10.0.0.1",
        "RemoteIP": "192.168.1.100",
        "CommandLine": "cmd.exe /c whoami",
        "Severity": "high",
        "timestamp": 1700000000,
    }
    event = normalize(raw, EventSource.CROWDSTRIKE)
    assert event.actor_device == "laptop-04"
    assert event.actor_user == "alice"
    assert event.src_ip == "10.0.0.1"
    assert event.severity == 4
    assert event.category == "endpoint"


def test_okta_normalization():
    raw = {
        "actor": {"alternateId": "alice@corp.com"},
        "client": {"ipAddress": "10.0.0.5"},
        "eventType": "user.session.start",
        "severity": "medium",
    }
    normalizer = SDFNormalizer()
    event = normalizer.process('{"actor": {"alternateId": "alice@corp.com"}, "client": {"ipAddress": "10.0.0.5"}, "eventType": "user.session.start", "severity": "medium"}', "okta")
    assert event is not None
    assert event.actor_user == "alice@corp.com"
    assert event.category == "identity"


def test_normalizer_stats():
    n = SDFNormalizer()
    n.process('{"ComputerName": "x"}', "crowdstrike")
    n.process("not json {{}", "crowdstrike")
    assert n.stats["processed"] == 1
    assert n.stats["errors"] == 1


# -----------------------------------------------------------------------
# Layer 1: Entity Correlator
# -----------------------------------------------------------------------

def test_entity_correlation():
    correlator = EntityCorrelator()
    event = OCSFEvent(
        source=EventSource.CROWDSTRIKE,
        actor_user="alice",
        actor_device="laptop-04",
        src_ip="10.0.0.1",
        target_resource="db-finance",
        severity=4,
    )
    results = correlator.ingest_event(event)
    assert correlator.entity_count() >= 4
    assert correlator.relation_count() >= 3


def test_incident_chain():
    correlator = EntityCorrelator()
    event = OCSFEvent(
        source=EventSource.CROWDSTRIKE,
        actor_user="alice",
        actor_device="laptop-04",
        src_ip="10.0.0.1",
        severity=3,
    )
    correlator.ingest_event(event)
    chain = correlator.find_incident_chain("10.0.0.1")
    assert len(chain) >= 1


# -----------------------------------------------------------------------
# Layer 1: Temporal Tracker
# -----------------------------------------------------------------------

def test_temporal_drift_detection():
    tracker = TemporalTracker()
    event = OCSFEvent(source=EventSource.CROWDSTRIKE, severity=1)
    entity_id = "device:laptop-04"
    # Inject normal samples
    for i in range(65):
        event.timestamp = datetime.utcnow()
        tracker.ingest_event(event, entity_id, risk_score=0.2)
    # Inject spike
    for _ in range(5):
        event.severity = 5
        tracker.ingest_event(event, entity_id, risk_score=0.95)
    trend = tracker.get_risk_trend(entity_id)
    assert trend["current"] > 0.5
    assert trend["slope"] > 0


# -----------------------------------------------------------------------
# Layer 2: GMS Scorer
# -----------------------------------------------------------------------

def test_gms_computation():
    controls = [
        GovernanceControl("c1", "MFA", "identity", weight=0.9, effectiveness=0.7, exposure=0.8, cost=0.2),
        GovernanceControl("c2", "Patching", "endpoint", weight=0.8, effectiveness=0.6, exposure=0.9, cost=0.4),
    ]
    scorer = GMSScorer(DEFAULT_CONFIG)
    result = scorer.compute(controls)
    assert 0.0 <= result.score <= 1.0
    assert result.lower_bound <= result.score <= result.upper_bound
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.control_contributions) == 2


def test_gms_threshold():
    controls = [
        GovernanceControl("c1", "MFA", "identity", weight=1.0, effectiveness=0.1, exposure=0.9, cost=0.2),
    ]
    scorer = GMSScorer(DEFAULT_CONFIG)
    result = scorer.compute(controls)
    assert scorer.gms_below_threshold(result)


# -----------------------------------------------------------------------
# Layer 2: Attack Graph
# -----------------------------------------------------------------------

def test_attack_graph_dijkstra():
    ag = AttackGraph()
    for node in ["internet", "dmz", "appserver", "db"]:
        ag.add_asset(node, "server")
    ag.add_attack_edge("internet", "dmz", 0.6)
    ag.add_attack_edge("dmz", "appserver", 0.5)
    ag.add_attack_edge("appserver", "db", 0.4)

    path = ag.minimum_resistance_path("internet", "db")
    assert path is not None
    assert path.path[0] == "internet"
    assert path.path[-1] == "db"
    # composite probability = 0.6 * 0.5 * 0.4 = 0.12
    assert abs(path.probability - 0.12) < 0.01


def test_no_path_returns_none():
    ag = AttackGraph()
    ag.add_asset("a", "server")
    ag.add_asset("b", "server")
    result = ag.minimum_resistance_path("a", "b")
    assert result is None


# -----------------------------------------------------------------------
# Layer 2: Security Knowledge Graph
# -----------------------------------------------------------------------

def test_skg_knowledge_path():
    skg = SecurityKnowledgeGraph()
    entities = [
        Entity("user:alice", EntityType.USER, "Alice"),
        Entity("db:finance", EntityType.DATA_STORE, "DB-Finance"),
        Entity("server:aws-east", EntityType.CLOUD_SERVICE, "AWS-East-3"),
        Entity("cve:2020", EntityType.APPLICATION, "CVE-2020-XXXX"),
    ]
    for e in entities:
        skg.add_entity(e)
    skg.add_relation(Relation("user:alice", "db:finance", RelationType.HAS_ACCESS_TO))
    skg.add_relation(Relation("db:finance", "server:aws-east", RelationType.RESIDES_ON))
    skg.add_relation(Relation("server:aws-east", "cve:2020", RelationType.IS_VULNERABLE_TO))

    chains = skg.find_risk_chains("user:alice")
    assert any("cve:2020" in chain for chain in chains), "CVE should be reachable from Alice"
    assert skg.relation_count() == 3


# -----------------------------------------------------------------------
# Layer 3: CBF
# -----------------------------------------------------------------------

def test_cbf_blast_radius_violation():
    cbf = ControlBarrierFunctions(DEFAULT_CONFIG, total_asset_count=100)
    action = RemediationAction(
        action_type="isolate_device",
        target_entity_id="laptop-04",
        impact=ActionImpact.HIGH,
        confidence=0.9,
    )
    result = cbf.evaluate(action, affected_asset_count=20, is_reversible=False)
    assert not result.satisfied
    assert any(v.constraint_name == "blast_radius_limit" for v in result.violations)


def test_cbf_low_impact_passes():
    cbf = ControlBarrierFunctions(DEFAULT_CONFIG, total_asset_count=15000)
    action = RemediationAction(
        action_type="create_ticket",
        target_entity_id="user-alice",
        impact=ActionImpact.LOW,
        confidence=0.95,
    )
    result = cbf.evaluate(action, affected_asset_count=1, is_reversible=True)
    assert result.satisfied


# -----------------------------------------------------------------------
# Layer 3: HITL Queue
# -----------------------------------------------------------------------

def test_hitl_approve_reject_cycle():
    queue = HITLQueue()
    action = RemediationAction(
        action_type="suspend_account",
        target_entity_id="user:bob",
        impact=ActionImpact.HIGH,
        confidence=0.55,
    )
    queue.enqueue(action, escalation_reason="Low confidence")
    assert queue.pending_count() == 1

    approved = queue.approve(action.action_id, approver="ciso@corp.com", notes="Verified threat")
    assert approved is not None
    assert queue.pending_count() == 0
    assert len(queue.audit_log()) == 2  # ESCALATED + APPROVED


# -----------------------------------------------------------------------
# Layer 2: Risk Forecaster
# -----------------------------------------------------------------------

def test_risk_forecast_generation():
    controls = [
        GovernanceControl("c1", "MFA", "identity", weight=0.9, effectiveness=0.7, exposure=0.8, cost=0.2),
    ]
    scorer = GMSScorer(DEFAULT_CONFIG)
    gms = scorer.compute(controls)
    forecaster = RiskForecaster()
    forecaster.record_gms(gms)
    forecast = forecaster.forecast(controls, gms)
    assert len(forecast.forecast_7d) == 7
    assert forecast.trend in ("improving", "stable", "declining", "insufficient_data")
