"""
Preemptive Cyber Defense — Main Orchestration

Wires all four architectural layers together and runs the 60-second
governance polling loop described in the paper (Section 15.1).

Startup sequence:
  1. Load governance control catalog
  2. Initialize Security Knowledge Graph + attack graph
  3. Start SDF normalizer (Kafka consumer)
  4. Initialize RL optimizer and train initial policy
  5. Start 60s governance scoring loop
  6. Launch Executive Dashboard API (FastAPI)
"""
from __future__ import annotations
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import List

import numpy as np
import uvicorn

from src.common.config import DEFAULT_CONFIG
from src.common.models import (
    Entity, EntityType, GovernanceControl, OCSFEvent, Relation, RelationType
)
from src.knowledge_graph.attack_graph import AttackGraph
from src.knowledge_graph.skg import SecurityKnowledgeGraph
from src.knowledge_graph.embeddings import SKGEmbeddingTrainer
from src.layer1_sdf.normalizer import SDFNormalizer
from src.layer1_sdf.entity_correlator import EntityCorrelator
from src.layer1_sdf.temporal_tracker import TemporalTracker
from src.layer2_governance.gms_scorer import GMSScorer
from src.layer2_governance.risk_forecaster import RiskForecaster
from src.layer2_governance.rl_optimizer import RLGovernanceOptimizer
from src.layer3_remediation.remediation_agent import RemediationAgent
from src.layer4_executive import dashboard_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Sample control catalog (production: loaded from CMDB / GRC platform)
# ---------------------------------------------------------------------------

def load_default_controls() -> List[GovernanceControl]:
    return [
        GovernanceControl("ctrl-mfa-01", "MFA Enforcement", "identity", weight=0.9, effectiveness=0.7, exposure=0.8, cost=0.2),
        GovernanceControl("ctrl-patch-01", "Endpoint Patching", "endpoint", weight=0.85, effectiveness=0.6, exposure=0.9, cost=0.4),
        GovernanceControl("ctrl-fw-01", "Firewall Rule Review", "network", weight=0.75, effectiveness=0.8, exposure=0.7, cost=0.3),
        GovernanceControl("ctrl-iam-01", "IAM Privilege Review", "identity", weight=0.8, effectiveness=0.65, exposure=0.85, cost=0.35),
        GovernanceControl("ctrl-cloud-01", "Cloud Config Posture", "cloud", weight=0.7, effectiveness=0.55, exposure=0.9, cost=0.5),
        GovernanceControl("ctrl-dlp-01", "Data Loss Prevention", "data", weight=0.6, effectiveness=0.75, exposure=0.6, cost=0.4),
        GovernanceControl("ctrl-edr-01", "EDR Coverage", "endpoint", weight=0.8, effectiveness=0.85, exposure=0.5, cost=0.3),
        GovernanceControl("ctrl-vuln-01", "Vulnerability Scanning", "endpoint", weight=0.75, effectiveness=0.7, exposure=0.8, cost=0.25),
        GovernanceControl("ctrl-sso-01", "SSO Enforcement", "identity", weight=0.65, effectiveness=0.9, exposure=0.4, cost=0.2),
        GovernanceControl("ctrl-monitor-01", "Security Monitoring", "network", weight=0.7, effectiveness=0.8, exposure=0.6, cost=0.35),
    ]


def build_sample_attack_graph(graph: AttackGraph) -> str:
    """Build a small demo attack graph. Returns crown jewel node ID."""
    nodes = [
        ("internet", "external"),
        ("dmz-web-01", "server"),
        ("app-server-01", "server"),
        ("db-finance", "database"),
        ("user-alice", "user_account"),
        ("laptop-04", "device"),
        ("aws-east-3", "cloud_service"),
    ]
    for node_id, asset_type in nodes:
        graph.add_asset(node_id, asset_type)

    edges = [
        ("internet", "dmz-web-01", 0.6),
        ("dmz-web-01", "app-server-01", 0.4),
        ("app-server-01", "db-finance", 0.3),
        ("user-alice", "db-finance", 0.5),
        ("laptop-04", "app-server-01", 0.35),
        ("laptop-04", "aws-east-3", 0.45),
        ("aws-east-3", "db-finance", 0.25),
    ]
    for src, dst, prob in edges:
        graph.add_attack_edge(src, dst, prob)

    return "db-finance"  # crown jewel


def build_sample_skg(skg: SecurityKnowledgeGraph) -> None:
    """Build sample SKG matching paper's example knowledge path."""
    entities = [
        Entity("user:alice", EntityType.USER, "Alice"),
        Entity("db:db-finance", EntityType.DATA_STORE, "DB-Finance"),
        Entity("server:aws-east-3", EntityType.CLOUD_SERVICE, "CloudServer AWS-East-3"),
        Entity("cve:CVE-2020-XXXX", EntityType.APPLICATION, "CVE-2020-XXXX"),
    ]
    for e in entities:
        skg.add_entity(e)

    relations = [
        Relation("user:alice", "db:db-finance", RelationType.HAS_ACCESS_TO),
        Relation("db:db-finance", "server:aws-east-3", RelationType.RESIDES_ON),
        Relation("server:aws-east-3", "cve:CVE-2020-XXXX", RelationType.IS_VULNERABLE_TO),
    ]
    for r in relations:
        skg.add_relation(r)


# ---------------------------------------------------------------------------
# Governance polling loop (paper Section 15.1 algorithm)
# ---------------------------------------------------------------------------

async def governance_loop(
    controls: List[GovernanceControl],
    gms_scorer: GMSScorer,
    risk_forecaster: RiskForecaster,
    remediation_agent: RemediationAgent,
    attack_graph: AttackGraph,
    crown_jewel_id: str,
    poll_interval: int = CONFIG.gms_poll_interval_sec,
) -> None:
    logger.info("Governance polling loop started (interval=%ds)", poll_interval)
    while True:
        try:
            # Step 3: Compute GMS
            gms = gms_scorer.compute(controls)
            dashboard_api._current_gms = gms

            # Step 5: Update risk forecast
            risk_forecaster.record_gms(gms)
            forecast = risk_forecaster.forecast(controls, gms)
            dashboard_api._current_forecast = forecast

            # Attack graph sensitivity analysis (periodic)
            sensitivity = attack_graph.sensitivity_analysis(crown_jewel_id)
            top_edges = list(sensitivity.items())[:3]
            if top_edges:
                logger.info(
                    "Top attack paths to crown jewel: %s",
                    [(f"{e[0][0]}→{e[0][1]}", f"{e[1]:.2%}") for e in top_edges],
                )

            # Step 8: If GMS below threshold → trigger remediation
            if gms_scorer.gms_below_threshold(gms):
                posture_state = np.array([c.effectiveness for c in controls], dtype=np.float32)
                actions = remediation_agent.trigger(gms, controls, posture_state)
                logger.info("Remediation triggered: %d actions", len(actions))
            else:
                logger.info("GMS %.3f above threshold — no remediation needed.", gms.score)

            logger.info(
                "Governance cycle complete: GMS=%.3f [%.3f–%.3f] trend=%s forecast_7d_end=%.3f",
                gms.score, gms.lower_bound, gms.upper_bound,
                gms_scorer.recent_trend(),
                forecast.forecast_7d[-1] if forecast.forecast_7d else gms.score,
            )

        except Exception as exc:
            logger.exception("Governance loop error: %s", exc)

        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

async def main() -> None:
    logger.info("=== Preemptive Cyber Defense System Starting ===")

    # Layer 1: SDF
    normalizer = SDFNormalizer()
    correlator = EntityCorrelator()
    tracker = TemporalTracker()

    # Knowledge Graph
    skg = SecurityKnowledgeGraph()
    build_sample_skg(skg)

    attack_graph = AttackGraph()
    crown_jewel = build_sample_attack_graph(attack_graph)
    logger.info("Attack graph: %d nodes, %d edges. Crown jewel: %s",
                attack_graph.node_count(), attack_graph.edge_count(), crown_jewel)

    # SKG embeddings (train on initial triples)
    embedding_trainer = SKGEmbeddingTrainer(embedding_dim=CONFIG.embedding_dim)
    triples = skg.get_triples()
    if triples:
        embedding_trainer.build_vocabulary(triples)
        logger.info("SKG embedding vocabulary: %d entities, %d relations",
                    len(embedding_trainer._entity_to_idx),
                    len(embedding_trainer._relation_to_idx))

    # Layer 2: Governance Engine
    controls = load_default_controls()
    gms_scorer = GMSScorer(CONFIG)
    risk_forecaster = RiskForecaster()
    rl_optimizer = RLGovernanceOptimizer(CONFIG)
    rl_optimizer.initialize(controls)
    # Quick training in demo mode; production would use 50K+ timesteps
    rl_optimizer.train(total_timesteps=5_000)

    # Layer 3: Remediation Agent
    remediation_agent = RemediationAgent(CONFIG)
    remediation_agent._rl_optimizer = rl_optimizer

    # Layer 4: Executive Dashboard — wire dependencies
    dashboard_api.configure(gms_scorer, risk_forecaster, remediation_agent, attack_graph, controls)

    # Launch governance loop as background task
    governance_task = asyncio.create_task(
        governance_loop(controls, gms_scorer, risk_forecaster, remediation_agent, attack_graph, crown_jewel)
    )

    # Launch FastAPI server
    server_config = uvicorn.Config(
        app=dashboard_api.app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    logger.info("Executive Dashboard available at http://localhost:8000/docs")

    try:
        await server.serve()
    finally:
        governance_task.cancel()
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
