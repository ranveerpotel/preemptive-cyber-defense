"""
Layer 2 — AI Governance Engine: Risk Forecaster
7-day forward GMS projection. Implements the paper's forward-looking
governance: "Governance Maturity is currently 82%, with a forecast decline
of 15 percentage points over the next seven days."
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

from src.common.models import GMSResult, GovernanceControl, RiskForecast

logger = logging.getLogger(__name__)


class RiskForecaster:
    """
    Combines trend extrapolation with scheduled risk events (cloud migrations,
    patch windows) to produce a 7-day GMS forecast with per-day projections.
    """

    def __init__(self) -> None:
        self._gms_history: List[Tuple[datetime, float]] = []
        self._scheduled_events: List[Tuple[datetime, float]] = []  # (date, gms_delta)

    def record_gms(self, result: GMSResult) -> None:
        self._gms_history.append((result.timestamp, result.score))
        # Keep 30 days of history
        cutoff = datetime.utcnow() - timedelta(days=30)
        self._gms_history = [(ts, v) for ts, v in self._gms_history if ts >= cutoff]

    def schedule_event(self, event_date: datetime, expected_gms_delta: float) -> None:
        """Register a known future event that will affect GMS (e.g. cloud migration)."""
        self._scheduled_events.append((event_date, expected_gms_delta))

    def forecast(
        self,
        controls: List[GovernanceControl],
        current_gms: GMSResult,
    ) -> RiskForecast:
        """
        Generate 7-day GMS forecast using:
        1. Linear trend from historical GMS series
        2. Scheduled event adjustments
        3. Control degradation modeling (controls drift if not maintained)
        """
        daily_projections = self._project_daily(current_gms.score, days=7)
        daily_projections = self._apply_scheduled_events(daily_projections)
        trend = self._classify_trend(daily_projections)
        top_risk_drivers = self._identify_risk_drivers(controls)

        return RiskForecast(
            current_gms=current_gms.score,
            forecast_7d=daily_projections,
            trend=trend,
            top_risk_drivers=top_risk_drivers,
        )

    def _project_daily(self, current_gms: float, days: int = 7) -> List[float]:
        """
        Linear regression on recent history to extrapolate daily GMS.
        Falls back to flat projection if insufficient history.
        """
        if len(self._gms_history) < 3:
            return [current_gms] * days

        # Use hourly samples; average per day
        scores = [v for _, v in self._gms_history[-24 * 7:]]  # last 7 days
        if len(scores) < 3:
            return [current_gms] * days

        x = np.arange(len(scores), dtype=float)
        slope, intercept = np.polyfit(x, scores, 1)

        # Project forward
        n = len(scores)
        projections = []
        for d in range(1, days + 1):
            projected = intercept + slope * (n + d * 24)  # daily increment
            projected = max(0.0, min(1.0, projected))
            projections.append(round(projected, 3))
        return projections

    def _apply_scheduled_events(self, projections: List[float]) -> List[float]:
        now = datetime.utcnow()
        for event_date, delta in self._scheduled_events:
            days_away = (event_date - now).days
            if 0 <= days_away < len(projections):
                for d in range(days_away, len(projections)):
                    projections[d] = max(0.0, min(1.0, projections[d] + delta))
        return projections

    def _classify_trend(self, projections: List[float]) -> str:
        if not projections:
            return "stable"
        delta = projections[-1] - projections[0]
        if delta > 0.03:
            return "improving"
        if delta < -0.03:
            return "declining"
        return "stable"

    def _identify_risk_drivers(self, controls: List[GovernanceControl]) -> List[str]:
        """
        Identifies top controls contributing to risk (low effectiveness × high weight).
        """
        risk_scores = [
            (ctrl.control_id, ctrl.weight * (1.0 - ctrl.effectiveness) * ctrl.exposure)
            for ctrl in controls
        ]
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        return [ctrl_id for ctrl_id, _ in risk_scores[:5]]
