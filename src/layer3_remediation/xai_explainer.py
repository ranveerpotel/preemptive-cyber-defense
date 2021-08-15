"""
Layer 3 — Agentic Remediation: SHAP-Based Explainable AI (XAI)

Each remediation action is accompanied by a machine-generated Rationale Report
using SHAP (SHapley Additive exPlanations) values.

Paper Section VIII: "This connection was blocked because it (1) employed an
atypical protocol [SHAP: 0.42], (2) originated from an elevated threat intelligence
score [SHAP: 0.31], and (3) occurred during heightened administrative activity [SHAP: 0.27]."
"""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.common.models import RemediationAction

logger = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Using synthetic rationale generation.")


# Feature names for the governance scoring model input
DEFAULT_FEATURE_NAMES = [
    "threat_intelligence_score",
    "protocol_anomaly_score",
    "time_of_day_risk",
    "source_reputation",
    "lateral_movement_indicator",
    "vulnerability_severity",
    "asset_criticality",
    "historical_behavior_deviation",
]


class SHAPExplainer:
    """
    Generates SHAP-based Rationale Reports for every autonomous action.
    Implements paper Section VIII: φᵢ = weighted avg marginal contribution
    of feature i across all possible feature subsets.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self._model = model
        self._feature_names = feature_names or DEFAULT_FEATURE_NAMES
        self._explainer: Optional[Any] = None
        self._background_data: Optional[np.ndarray] = None

    def fit(self, background_data: np.ndarray) -> None:
        """Initialize SHAP explainer with background dataset."""
        self._background_data = background_data
        if _SHAP_AVAILABLE and self._model is not None:
            try:
                self._explainer = shap.KernelExplainer(self._model, background_data)
                logger.info("SHAP KernelExplainer initialized.")
            except Exception as exc:
                logger.warning("SHAP explainer init failed: %s", exc)
                self._explainer = None

    def explain(
        self,
        feature_vector: np.ndarray,
        action: RemediationAction,
    ) -> str:
        """
        Generate a machine-readable Rationale Report for the action.
        Returns formatted rationale string matching paper's format.
        """
        shap_values = self._compute_shap_values(feature_vector)
        rationale = self._format_rationale(action, shap_values, feature_vector)
        action.shap_rationale = rationale
        return rationale

    def _compute_shap_values(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """
        Compute SHAP φᵢ values for each feature.
        Falls back to synthetic values proportional to feature magnitude if SHAP unavailable.
        """
        if _SHAP_AVAILABLE and self._explainer is not None:
            try:
                values = self._explainer.shap_values(feature_vector.reshape(1, -1))
                if isinstance(values, list):
                    values = values[0]
                shap_arr = np.array(values).flatten()
                return {
                    name: float(shap_arr[i]) if i < len(shap_arr) else 0.0
                    for i, name in enumerate(self._feature_names)
                }
            except Exception as exc:
                logger.warning("SHAP computation failed: %s", exc)

        # Synthetic fallback: normalize feature values as proxy SHAP contributions
        total = float(np.sum(np.abs(feature_vector)) + 1e-9)
        return {
            name: float(feature_vector[i] / total) if i < len(feature_vector) else 0.0
            for i, name in enumerate(self._feature_names)
        }

    def _format_rationale(
        self,
        action: RemediationAction,
        shap_values: Dict[str, float],
        feature_vector: np.ndarray,
    ) -> str:
        """
        Format SHAP values into the paper's structured Rationale Report.
        Sorted by absolute contribution; top 3 most influential features shown.
        """
        sorted_features = sorted(
            shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        reasons = []
        for i, (feature_name, shap_val) in enumerate(sorted_features, start=1):
            human_name = feature_name.replace("_", " ")
            direction = "elevated" if shap_val > 0 else "reduced"
            reasons.append(f"({i}) {direction} {human_name} [SHAP contribution: {abs(shap_val):.2f}]")

        rationale = (
            f"Action '{action.action_type}' on entity '{action.target_entity_id}' "
            f"was {'executed autonomously' if action.confidence >= 0.6 else 'escalated to HITL'} "
            f"because: {', '.join(reasons)}."
        )
        return rationale

    def top_features(self, shap_values: Dict[str, float], n: int = 5) -> List[Tuple[str, float]]:
        return sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
