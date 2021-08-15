"""
Layer 3 — Agentic Remediation: Control Barrier Functions (CBF)

CBF h: S → ℝ defines safe set C = {s ∈ S | h(s) ≥ 0}.
Any autonomous action must maintain h(s') ≥ 0.

Paper Section X safety constraints:
  1. Max tolerable service downtime
  2. Change window enforcement
  3. Blast radius limitation
  4. Reversibility preference
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, List, Optional

from src.common.config import SystemConfig, DEFAULT_CONFIG
from src.common.models import ActionImpact, RemediationAction

logger = logging.getLogger(__name__)


@dataclass
class CBFViolation:
    constraint_name: str
    reason: str
    suggested_alternative: Optional[str] = None


@dataclass
class CBFResult:
    satisfied: bool
    violations: List[CBFViolation]
    safe_action: Optional[str] = None   # alternative if violated


class ControlBarrierFunctions:
    """
    Enforces structural safety constraints on autonomous remediation actions.
    Safety is structurally enforced rather than relying on model self-restraint.
    """

    def __init__(
        self,
        config: SystemConfig = DEFAULT_CONFIG,
        change_windows: Optional[List[Dict]] = None,
        total_asset_count: int = 15_000,
    ) -> None:
        self.config = config
        self.total_asset_count = total_asset_count
        # Default change windows: weeknights 10pm-6am + weekends
        self.change_windows = change_windows or [
            {"start": time(22, 0), "end": time(6, 0)},
        ]
        self._downtime_tracker: Dict[str, float] = {}  # entity_id → downtime fraction
        self._active_incident: bool = False

    def evaluate(
        self,
        action: RemediationAction,
        affected_asset_count: int,
        is_reversible: bool,
        alternative_action: Optional[str] = None,
    ) -> CBFResult:
        """
        Evaluate all CBF constraints for a proposed action.
        Returns CBFResult with satisfied=True only if ALL constraints pass.
        Paper Section X: h(s') ≥ 0 must hold for all constraints simultaneously.
        """
        violations: List[CBFViolation] = []

        # Constraint 1: Max tolerable service downtime
        v = self._check_downtime(action)
        if v:
            violations.append(v)

        # Constraint 2: Change window enforcement
        v = self._check_change_window(action)
        if v:
            violations.append(v)

        # Constraint 3: Blast radius limitation
        v = self._check_blast_radius(action, affected_asset_count)
        if v:
            violations.append(v)

        # Constraint 4: Reversibility preference (soft constraint)
        if not is_reversible and action.impact == ActionImpact.HIGH:
            violations.append(CBFViolation(
                constraint_name="reversibility_preference",
                reason="High-impact irreversible action requires HITL approval.",
                suggested_alternative=alternative_action,
            ))

        satisfied = len(violations) == 0
        safe_action = alternative_action if violations else None

        if not satisfied:
            logger.warning(
                "CBF violated for action %s: %s",
                action.action_id,
                [v.constraint_name for v in violations],
            )

        return CBFResult(satisfied=satisfied, violations=violations, safe_action=safe_action)

    def mark_active_incident(self, active: bool) -> None:
        """Critical incidents override change window constraint (paper Section X)."""
        self._active_incident = active

    def record_service_impact(self, entity_id: str, downtime_fraction: float) -> None:
        """Track cumulative downtime for blast-radius accounting."""
        self._downtime_tracker[entity_id] = max(
            self._downtime_tracker.get(entity_id, 0.0), downtime_fraction
        )

    # ------------------------------------------------------------------
    # Individual constraint checks — h(s) ≥ 0 formulations
    # ------------------------------------------------------------------

    def _check_downtime(self, action: RemediationAction) -> Optional[CBFViolation]:
        entity_downtime = self._downtime_tracker.get(action.target_entity_id, 0.0)
        if entity_downtime > self.config.max_service_downtime_pct:
            return CBFViolation(
                constraint_name="max_service_downtime",
                reason=f"Estimated downtime {entity_downtime:.1%} exceeds threshold "
                       f"{self.config.max_service_downtime_pct:.1%}.",
                suggested_alternative="schedule_maintenance_window",
            )
        return None

    def _check_change_window(self, action: RemediationAction) -> Optional[CBFViolation]:
        if self._active_incident:
            return None  # active incident overrides change window
        if action.impact == ActionImpact.LOW:
            return None  # low-impact actions exempt from change window

        now = datetime.utcnow().time()
        in_window = False
        for window in self.change_windows:
            start, end = window["start"], window["end"]
            if start <= end:
                in_window = start <= now <= end
            else:
                # Crosses midnight
                in_window = now >= start or now <= end
            if in_window:
                break

        if not in_window:
            return CBFViolation(
                constraint_name="change_window_enforcement",
                reason=f"High-impact action attempted outside change window at {now}.",
                suggested_alternative="defer_to_next_change_window",
            )
        return None

    def _check_blast_radius(
        self, action: RemediationAction, affected_asset_count: int
    ) -> Optional[CBFViolation]:
        blast_radius_pct = affected_asset_count / max(1, self.total_asset_count)
        if blast_radius_pct > self.config.blast_radius_threshold:
            return CBFViolation(
                constraint_name="blast_radius_limit",
                reason=f"Action affects {blast_radius_pct:.1%} of assets "
                       f"(threshold: {self.config.blast_radius_threshold:.1%}). "
                       "Mandatory HITL required.",
                suggested_alternative="scope_reduction",
            )
        return None
