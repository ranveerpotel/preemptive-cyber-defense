"""
Layer 3 — Agentic Remediation: Remediation Agent

Unlike SOAR playbooks (deterministic scripts), the agentic layer uses a reasoning
model capable of evaluating multi-step remediation strategies, estimating risk
reduction, and selecting optimal actions subject to CBF safety constraints.

Paper Section 15.2 algorithm (steps 9-16):
  9.  Receive trigger with current state s ∈ S
  10. Query RL policy π → ranked action set A' ⊆ A
  11. Evaluate each a ∈ A' against CBF constraints
  12. Low confidence → HITL queue
  13. High confidence + safe → autonomous execution
  14. Monitor post-action telemetry (15 min window)
  15. Compute realized ΔR_actual vs predicted
  16. Submit (s, a, ΔR_actual, s') to RL training buffer
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.common.config import SystemConfig, DEFAULT_CONFIG
from src.common.models import (
    ActionImpact, ActionStatus, GMSResult, GovernanceControl, RemediationAction
)
from src.layer2_governance.rl_optimizer import GovernanceAction, RLGovernanceOptimizer
from src.layer3_remediation.cbf import CBFResult, ControlBarrierFunctions
from src.layer3_remediation.hitl_queue import HITLQueue
from src.layer3_remediation.xai_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

# Action impact classification
_HIGH_IMPACT_TYPES = {"isolate_device", "suspend_account", "block_user", "disable_service"}
_MEDIUM_IMPACT_TYPES = {"close_port", "filter_traffic", "revoke_token", "quarantine_process"}
_LOW_IMPACT_TYPES = {"create_ticket", "send_notification", "increase_monitoring", "flag_for_review"}


class RemediationAgent:
    """
    Safety-bounded agentic remediation. Orchestrates the full remediation
    pipeline from trigger through execution/escalation and feedback learning.
    """

    def __init__(
        self,
        config: SystemConfig = DEFAULT_CONFIG,
        action_executor: Optional[Callable[[RemediationAction], bool]] = None,
    ) -> None:
        self.config = config
        self._rl_optimizer = RLGovernanceOptimizer(config)
        self._cbf = ControlBarrierFunctions(config)
        self._hitl = HITLQueue(
            approval_callback=self._on_hitl_approved,
            rejection_callback=self._on_hitl_rejected,
        )
        self._explainer = SHAPExplainer()
        self._action_executor = action_executor or self._default_executor
        self._audit_log: List[Dict] = []
        self._experience_buffer: List[Dict] = []

    # ------------------------------------------------------------------
    # Main remediation pipeline (paper Section 15.2)
    # ------------------------------------------------------------------

    def trigger(
        self,
        gms_result: GMSResult,
        controls: List[GovernanceControl],
        posture_state: np.ndarray,
    ) -> List[RemediationAction]:
        """
        Entry point: GMS below threshold triggers this pipeline.
        Returns list of actions taken or queued.
        """
        logger.info("Remediation triggered. GMS=%.3f < threshold=%.3f",
                    gms_result.score, self.config.gms_threshold_trigger)

        # Step 10: Query RL policy for ranked action recommendations
        recommended = self._rl_optimizer.recommend_actions(gms_result, controls, top_k=5)
        executed_actions: List[RemediationAction] = []

        for gov_action in recommended:
            action = self._build_action(gov_action, gms_result)

            # Step 11: Evaluate CBF constraints
            cbf_result = self._cbf.evaluate(
                action,
                affected_asset_count=self._estimate_blast_radius(gov_action),
                is_reversible=action.impact != ActionImpact.HIGH,
                alternative_action=f"defer_{gov_action.intervention_type}",
            )
            action.cbf_satisfied = cbf_result.satisfied

            # Generate SHAP rationale regardless of routing decision
            feature_vec = self._build_feature_vector(gov_action, gms_result)
            self._explainer.explain(feature_vec, action)

            # Steps 12-13: Route based on confidence and safety
            if not cbf_result.satisfied:
                reason = "; ".join(v.reason for v in cbf_result.violations)
                self._hitl.enqueue(action, escalation_reason=f"CBF violation: {reason}")
            elif gms_result.confidence < self.config.autonomous_confidence_threshold:
                self._hitl.enqueue(
                    action,
                    escalation_reason=f"Low confidence: {gms_result.confidence:.2f} < {self.config.autonomous_confidence_threshold}",
                )
            elif action.impact == ActionImpact.HIGH and gms_result.confidence < self.config.hitl_confidence_threshold:
                self._hitl.enqueue(
                    action,
                    escalation_reason=f"HIGH impact requires confidence ≥ {self.config.hitl_confidence_threshold}",
                )
            else:
                # Step 13: Autonomous execution
                self._execute_action(action, posture_state)

            executed_actions.append(action)
            self._audit_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action_id": action.action_id,
                "action_type": action.action_type,
                "impact": action.impact.value,
                "routed_to": "autonomous" if action.status == ActionStatus.EXECUTING else "hitl",
                "cbf_satisfied": cbf_result.satisfied,
                "confidence": gms_result.confidence,
                "shap_rationale": action.shap_rationale,
            })

        return executed_actions

    # ------------------------------------------------------------------
    # Action execution and monitoring
    # ------------------------------------------------------------------

    def _execute_action(self, action: RemediationAction, pre_state: np.ndarray) -> None:
        action.status = ActionStatus.EXECUTING
        logger.info(
            "Autonomous execution: action=%s type=%s target=%s",
            action.action_id, action.action_type, action.target_entity_id,
        )
        success = self._action_executor(action)
        if success:
            action.status = ActionStatus.COMPLETED
            action.completed_at = datetime.utcnow()
            # Step 14: Schedule post-action monitoring (async in production)
            self._schedule_post_action_monitoring(action, pre_state)
        else:
            action.status = ActionStatus.FAILED
            logger.error("Action execution failed: %s", action.action_id)

    def _schedule_post_action_monitoring(
        self, action: RemediationAction, pre_state: np.ndarray
    ) -> None:
        """
        Step 14-16: Monitor post-action telemetry for 15 minutes,
        compute realized ΔR_actual, submit experience to RL buffer.
        In production, this would be an async task.
        """
        logger.info(
            "Post-action monitoring scheduled: action=%s window=%ds",
            action.action_id, self.config.post_action_monitor_window_sec,
        )
        # Simulated: in production, this fires after the monitoring window
        realized_reduction = action.estimated_risk_reduction * np.random.uniform(0.7, 1.1)
        action.actual_risk_reduction = realized_reduction
        self._experience_buffer.append({
            "pre_state": pre_state.tolist(),
            "action_idx": 0,  # would be actual action index
            "actual_risk_reduction": realized_reduction,
            "post_state": pre_state.tolist(),  # would be updated telemetry
        })

    # ------------------------------------------------------------------
    # HITL callbacks
    # ------------------------------------------------------------------

    def _on_hitl_approved(self, action: RemediationAction) -> None:
        logger.info("HITL approved action %s — executing.", action.action_id)
        state = np.zeros(len(action.api_payload))
        self._execute_action(action, state)

    def _on_hitl_rejected(self, action: RemediationAction) -> None:
        logger.info("HITL rejected action %s — cancelled.", action.action_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_action(self, gov_action: GovernanceAction, gms: GMSResult) -> RemediationAction:
        impact = self._classify_impact(gov_action.intervention_type)
        return RemediationAction(
            action_type=gov_action.intervention_type,
            target_entity_id=gov_action.control_id,
            impact=impact,
            confidence=gms.confidence,
            estimated_risk_reduction=gov_action.estimated_effectiveness_delta * gms.score,
        )

    def _classify_impact(self, intervention_type: str) -> ActionImpact:
        if intervention_type in _HIGH_IMPACT_TYPES:
            return ActionImpact.HIGH
        if intervention_type in _MEDIUM_IMPACT_TYPES:
            return ActionImpact.MEDIUM
        return ActionImpact.LOW

    def _estimate_blast_radius(self, gov_action: GovernanceAction) -> int:
        # Heuristic: HIGH = 500 assets, MEDIUM = 50, LOW = 5
        impact_map = {"high": 500, "medium": 50, "low": 5}
        impact = self._classify_impact(gov_action.intervention_type).value
        return impact_map.get(impact, 5)

    def _build_feature_vector(
        self, gov_action: GovernanceAction, gms: GMSResult
    ) -> np.ndarray:
        return np.array([
            1.0 - gms.score,                       # threat_intelligence_score
            gov_action.estimated_effectiveness_delta,
            datetime.utcnow().hour / 24.0,          # time_of_day_risk
            1.0 - gms.confidence,                   # source_reputation (inverted)
            gms.std_dev,                            # lateral_movement_indicator proxy
            gov_action.cost,                        # vulnerability_severity proxy
            gov_action.estimated_effectiveness_delta * gov_action.cost,
            1.0 - gms.lower_bound,                 # historical_behavior_deviation
        ], dtype=np.float32)

    @staticmethod
    def _default_executor(action: RemediationAction) -> bool:
        logger.info("[MOCK EXECUTOR] Would execute: %s on %s", action.action_type, action.target_entity_id)
        return True

    def audit_log(self) -> List[Dict]:
        return list(self._audit_log)

    def hitl_queue(self) -> HITLQueue:
        return self._hitl
