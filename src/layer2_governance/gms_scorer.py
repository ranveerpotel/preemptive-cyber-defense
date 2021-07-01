"""
Layer 2 — AI Governance Engine: Governance Maturity Score (GMS)

Paper formulation:
  GMS = Σ(wᵢ · eᵢ) / Σ(wᵢ · exposureᵢ)
  GMS_robust = GMS_point ± k · σ(GMS)
  σ(GMS) estimated via Monte Carlo dropout (N=100 forward passes).

Complexity: O(k·n) time, O(k) space. Real-time (<1s).
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.common.config import SystemConfig, DEFAULT_CONFIG
from src.common.models import GovernanceControl, GMSResult

logger = logging.getLogger(__name__)


class GovernanceScoringNetwork(nn.Module):
    """
    Neural scoring model with MC-Dropout for uncertainty estimation.
    Input: concatenated [effectiveness, exposure, weight] per control.
    Output: scalar GMS adjustment factor.
    """

    def __init__(self, num_controls: int, dropout_rate: float = 0.2) -> None:
        super().__init__()
        input_dim = num_controls * 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GMSScorer:
    """
    Computes Governance Maturity Score with Monte Carlo uncertainty bounds.
    Implements the paper's formulation from Sections 3.2, 9, and 15.1.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._neural_model: Optional[GovernanceScoringNetwork] = None
        self._history: List[GMSResult] = []

    def compute(self, controls: List[GovernanceControl]) -> GMSResult:
        """
        Main GMS computation with uncertainty bounds.

        Steps (paper Section 15.1):
          1. Compute weighted GMS_point = Σ(wᵢ·eᵢ) / Σ(wᵢ·exposureᵢ)
          2. Run MC-Dropout N times to estimate σ(GMS)
          3. Apply confidence interval: GMS_robust = GMS_point ± k·σ
        """
        if not controls:
            return GMSResult(score=0.0, lower_bound=0.0, upper_bound=0.0, std_dev=0.0, confidence=0.0)

        # Step 1: analytical GMS point estimate
        gms_point, contributions = self._compute_analytical_gms(controls)

        # Step 2: MC-Dropout uncertainty estimation
        std_dev = self._mc_dropout_uncertainty(controls)

        # Step 3: robust bounds
        k = self.config.confidence_multiplier_k
        lower = max(0.0, gms_point - k * std_dev)
        upper = min(1.0, gms_point + k * std_dev)

        # Posterior confidence: inversely proportional to relative uncertainty
        confidence = 1.0 - min(1.0, std_dev / (gms_point + 1e-9))

        result = GMSResult(
            score=gms_point,
            lower_bound=lower,
            upper_bound=upper,
            std_dev=std_dev,
            confidence=confidence,
            control_contributions=contributions,
        )
        self._history.append(result)
        logger.info(
            "GMS: %.3f [%.3f, %.3f] σ=%.3f conf=%.2f",
            gms_point, lower, upper, std_dev, confidence,
        )
        return result

    def _compute_analytical_gms(
        self, controls: List[GovernanceControl]
    ) -> Tuple[float, Dict[str, float]]:
        """
        GMS = Σ(wᵢ · eᵢ) / Σ(wᵢ · exposureᵢ)
        Returns (gms_score, per_control_contributions).
        """
        numerator = 0.0
        denominator = 0.0
        contributions: Dict[str, float] = {}

        for ctrl in controls:
            weighted_effectiveness = ctrl.weight * ctrl.effectiveness
            weighted_exposure = ctrl.weight * ctrl.exposure
            numerator += weighted_effectiveness
            denominator += weighted_exposure
            contributions[ctrl.control_id] = weighted_effectiveness

        gms = numerator / (denominator + 1e-9)
        gms = max(0.0, min(1.0, gms))

        # Normalize contributions to show relative impact
        total = sum(contributions.values()) + 1e-9
        contributions = {k: v / total for k, v in contributions.items()}
        return gms, contributions

    def _mc_dropout_uncertainty(self, controls: List[GovernanceControl]) -> float:
        """
        Monte Carlo Dropout: run N forward passes with dropout active to
        estimate the standard deviation of the GMS distribution.
        Paper reference: Section IX, σ(GMS) via MC dropout.
        """
        if self._neural_model is None:
            self._neural_model = GovernanceScoringNetwork(len(controls))

        # Build feature vector: [effectiveness, exposure, weight] per control
        features = []
        for ctrl in controls:
            features.extend([ctrl.effectiveness, ctrl.exposure, ctrl.weight])
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Ensure consistent size with model (rebuild if controls count changed)
        if x.shape[1] != len(controls) * 3:
            self._neural_model = GovernanceScoringNetwork(len(controls))

        # Enable dropout at inference time for MC estimation
        self._neural_model.train()
        samples: List[float] = []
        with torch.no_grad():
            for _ in range(self.config.mc_dropout_passes):
                out = self._neural_model(x)
                samples.append(float(out.item()))

        return float(np.std(samples))

    def gms_below_threshold(self, result: GMSResult) -> bool:
        return result.score < self.config.gms_threshold_trigger

    def recent_trend(self, window: int = 10) -> str:
        if len(self._history) < 3:
            return "insufficient_data"
        recent = [r.score for r in self._history[-window:]]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.005:
            return "improving"
        if slope < -0.005:
            return "declining"
        return "stable"
