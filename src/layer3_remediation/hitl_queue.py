"""
Layer 3 — Agentic Remediation: Human-in-the-Loop (HITL) Queue

High-stakes actions are routed through the HITL approval gate.
Conditions for HITL escalation (paper Sections 3.3, 15.2):
  - posterior_probability < confidence_threshold (regardless of impact)
  - ActionImpact.HIGH (device isolation, account suspension)
  - CBF violation (blast radius, change window)

"Human judgment remains authoritative for consequential decisions."
"""
from __future__ import annotations
import logging
import uuid
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

from src.common.models import ActionImpact, ActionStatus, RemediationAction

logger = logging.getLogger(__name__)


class HITLQueue:
    """
    Thread-safe queue for actions requiring human approval.
    Supports approval, rejection, and timeout escalation.
    """

    def __init__(
        self,
        approval_callback: Optional[Callable[[RemediationAction], None]] = None,
        rejection_callback: Optional[Callable[[RemediationAction], None]] = None,
        timeout_seconds: int = 3600,   # 1 hour default SLA
    ) -> None:
        self._queue: Deque[RemediationAction] = deque()
        self._pending: Dict[str, RemediationAction] = {}
        self._audit_log: List[Dict] = []
        self._approval_callback = approval_callback
        self._rejection_callback = rejection_callback
        self.timeout_seconds = timeout_seconds

    def enqueue(self, action: RemediationAction, escalation_reason: str) -> str:
        """
        Add an action to the HITL queue. Returns queue position ID.
        Paper Section 15.2 step 12.
        """
        action.status = ActionStatus.AWAITING_HITL
        self._pending[action.action_id] = action
        self._queue.append(action)
        self._audit("ESCALATED_TO_HITL", action, {"reason": escalation_reason})
        logger.info(
            "HITL escalation: action=%s impact=%s reason='%s'",
            action.action_id, action.impact.value, escalation_reason,
        )
        return action.action_id

    def approve(self, action_id: str, approver: str, notes: str = "") -> Optional[RemediationAction]:
        """
        Human approves an action. Triggers execution callback.
        """
        action = self._pending.pop(action_id, None)
        if not action:
            logger.warning("Approval for unknown action_id=%s", action_id)
            return None

        action.status = ActionStatus.APPROVED
        self._remove_from_queue(action_id)
        self._audit("APPROVED", action, {"approver": approver, "notes": notes})
        logger.info("HITL approved: action=%s by %s", action_id, approver)

        if self._approval_callback:
            self._approval_callback(action)
        return action

    def reject(self, action_id: str, rejector: str, reason: str = "") -> Optional[RemediationAction]:
        """
        Human rejects an action. Action is cancelled.
        """
        action = self._pending.pop(action_id, None)
        if not action:
            logger.warning("Rejection for unknown action_id=%s", action_id)
            return None

        action.status = ActionStatus.REJECTED
        self._remove_from_queue(action_id)
        self._audit("REJECTED", action, {"rejector": rejector, "reason": reason})
        logger.info("HITL rejected: action=%s by %s: %s", action_id, rejector, reason)

        if self._rejection_callback:
            self._rejection_callback(action)
        return action

    def pending_count(self) -> int:
        return len(self._pending)

    def pending_actions(self) -> List[RemediationAction]:
        return list(self._queue)

    def check_timeouts(self) -> List[RemediationAction]:
        """
        Returns actions that have exceeded the SLA timeout.
        Organizations with defined SLAs should call this periodically.
        """
        now = datetime.utcnow()
        timed_out = []
        for action in list(self._pending.values()):
            elapsed = (now - action.created_at).total_seconds()
            if elapsed > self.timeout_seconds:
                timed_out.append(action)
                logger.warning(
                    "HITL SLA breach: action=%s elapsed=%.0fs", action.action_id, elapsed
                )
        return timed_out

    def audit_log(self) -> List[Dict]:
        return list(self._audit_log)

    def _remove_from_queue(self, action_id: str) -> None:
        self._queue = deque(a for a in self._queue if a.action_id != action_id)

    def _audit(self, event_type: str, action: RemediationAction, extra: Dict) -> None:
        self._audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "action_id": action.action_id,
            "action_type": action.action_type,
            "impact": action.impact.value,
            "target": action.target_entity_id,
            "shap_rationale": action.shap_rationale,
            **extra,
        })
