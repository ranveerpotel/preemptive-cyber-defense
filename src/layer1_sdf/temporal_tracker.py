"""
Layer 1 — Security Data Fabric: Temporal Tracker
Time-series analysis of risk indicators per entity. Detects gradual drift
(misconfiguration growing from minor deviation to critical exposure) before
it crosses an exploitable threshold.
"""
from __future__ import annotations
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.common.models import OCSFEvent

logger = logging.getLogger(__name__)

# Named window sizes for drift detection
_SHORT_WINDOW = 5    # 5-minute micro window
_LONG_WINDOW = 60    # 60-minute macro window
_ANOMALY_Z_THRESHOLD = 2.5  # standard deviations for anomaly flag


class RiskSample:
    __slots__ = ("timestamp", "risk_score", "severity", "event_count")

    def __init__(self, timestamp: datetime, risk_score: float, severity: int, event_count: int = 1) -> None:
        self.timestamp = timestamp
        self.risk_score = risk_score
        self.severity = severity
        self.event_count = event_count


class EntityTimeSeries:
    """Sliding-window time-series for a single entity."""

    def __init__(self, entity_id: str, max_samples: int = 1440) -> None:
        self.entity_id = entity_id
        self._samples: Deque[RiskSample] = deque(maxlen=max_samples)

    def add_sample(self, timestamp: datetime, risk_score: float, severity: int) -> None:
        self._samples.append(RiskSample(timestamp, risk_score, severity))

    def rolling_mean(self, window: int) -> float:
        recent = list(self._samples)[-window:]
        if not recent:
            return 0.0
        return float(np.mean([s.risk_score for s in recent]))

    def rolling_std(self, window: int) -> float:
        recent = list(self._samples)[-window:]
        if len(recent) < 2:
            return 0.0
        return float(np.std([s.risk_score for s in recent]))

    def is_anomalous(self) -> bool:
        """Z-score anomaly: current value vs long-window baseline."""
        if len(self._samples) < _SHORT_WINDOW:
            return False
        current = self.rolling_mean(_SHORT_WINDOW)
        baseline_mean = self.rolling_mean(_LONG_WINDOW)
        baseline_std = self.rolling_std(_LONG_WINDOW)
        if baseline_std < 1e-6:
            return False
        z_score = (current - baseline_mean) / baseline_std
        return z_score > _ANOMALY_Z_THRESHOLD

    def drift_slope(self) -> float:
        """Linear regression slope over all samples — positive = worsening."""
        if len(self._samples) < 3:
            return 0.0
        scores = [s.risk_score for s in self._samples]
        x = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0])
        return slope

    def current_risk(self) -> float:
        if not self._samples:
            return 0.0
        return self._samples[-1].risk_score

    def sample_count(self) -> int:
        return len(self._samples)


class TemporalTracker:
    """
    Aggregates time-series data across all entities. Implements the paper's
    temporal tracking: monitors evolution of risk indicators, detecting gradual
    drift before it crosses an exploitable threshold.
    """

    def __init__(self) -> None:
        self._series: Dict[str, EntityTimeSeries] = {}
        self._anomaly_callbacks: List = []

    def ingest_event(self, event: OCSFEvent, entity_id: str, risk_score: float) -> None:
        if entity_id not in self._series:
            self._series[entity_id] = EntityTimeSeries(entity_id)
        ts = self._series[entity_id]
        ts.add_sample(event.timestamp, risk_score, event.severity)

        if ts.is_anomalous():
            logger.warning("Temporal anomaly detected for entity %s (z > %.1f)", entity_id, _ANOMALY_Z_THRESHOLD)
            for cb in self._anomaly_callbacks:
                cb(entity_id, ts)

    def get_risk_trend(self, entity_id: str) -> Dict[str, float]:
        ts = self._series.get(entity_id)
        if not ts or ts.sample_count() == 0:
            return {"current": 0.0, "slope": 0.0, "anomalous": False}
        return {
            "current": ts.current_risk(),
            "slope": ts.drift_slope(),
            "anomalous": ts.is_anomalous(),
            "short_mean": ts.rolling_mean(_SHORT_WINDOW),
            "long_mean": ts.rolling_mean(_LONG_WINDOW),
        }

    def top_drifting_entities(self, n: int = 10) -> List[Tuple[str, float]]:
        """Returns entity_ids sorted by worsening drift slope (descending)."""
        slopes = [
            (eid, series.drift_slope())
            for eid, series in self._series.items()
            if series.sample_count() >= 3
        ]
        return sorted(slopes, key=lambda x: x[1], reverse=True)[:n]

    def anomalous_entities(self) -> List[str]:
        return [eid for eid, ts in self._series.items() if ts.is_anomalous()]

    def register_anomaly_callback(self, callback) -> None:
        self._anomaly_callbacks.append(callback)

    def entity_count(self) -> int:
        return len(self._series)
