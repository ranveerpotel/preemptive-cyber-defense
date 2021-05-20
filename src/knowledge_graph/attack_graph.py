"""
Attack Graph: directed graph G = (V, E, P) where P: E → [0,1] gives
the probability of successful traversal along each edge.

Path probability = product of edge probabilities along minimum-resistance path.
Uses modified Dijkstra on log-transformed probabilities (multiplicative → additive).
Complexity: O(|V| + |E|). Target: 10K+ nodes.
"""
from __future__ import annotations
import math
import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.common.models import AttackPath, Entity

logger = logging.getLogger(__name__)

_LOG_BASE = 10.0
_EPSILON = 1e-9   # prevent log(0)


class AttackGraph:
    """
    Represents organizational assets as nodes and potential attack traversals
    as directed edges with traversal probability weights.

    Paper formulation:
      G = (V, E, P)
      V = servers, user accounts, databases, network segments, cloud resources
      E = directed attack paths derived from topology, trust, vulnerability adjacency
      P: E → [0,1] = probability of successful edge traversal
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_asset(self, node_id: str, asset_type: str, **attributes) -> None:
        self._graph.add_node(node_id, asset_type=asset_type, **attributes)

    def add_attack_edge(self, src: str, dst: str, traversal_probability: float) -> None:
        """
        Add or update a directed attack path edge.
        traversal_probability ∈ [0,1]: higher = easier for attacker.
        """
        p = max(_EPSILON, min(1.0 - _EPSILON, traversal_probability))
        # Store log-transformed weight for Dijkstra (negate for min-path = max-prob)
        log_weight = -math.log(p, _LOG_BASE)
        self._graph.add_edge(src, dst, probability=p, log_weight=log_weight)

    def update_edge_probability(self, src: str, dst: str, new_probability: float) -> None:
        if self._graph.has_edge(src, dst):
            self.add_attack_edge(src, dst, new_probability)

    def remove_asset(self, node_id: str) -> None:
        self._graph.remove_node(node_id)

    # ------------------------------------------------------------------
    # Path analysis (paper Section IV)
    # ------------------------------------------------------------------

    def minimum_resistance_path(self, src: str, dst: str) -> Optional[AttackPath]:
        """
        Compute the most likely attack path from src to dst using modified Dijkstra
        on log-transformed probabilities (multiplicative → additive).
        Returns None if no path exists.
        """
        if src not in self._graph or dst not in self._graph:
            return None
        try:
            path_nodes = nx.dijkstra_path(
                self._graph, src, dst, weight="log_weight"
            )
        except nx.NetworkXNoPath:
            return None

        edges = []
        composite_log_weight = 0.0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            p = self._graph[u][v]["probability"]
            edges.append((u, v, p))
            composite_log_weight += self._graph[u][v]["log_weight"]

        # composite probability = product of edge probabilities
        composite_probability = _LOG_BASE ** (-composite_log_weight)
        return AttackPath(
            path=path_nodes,
            probability=composite_probability,
            edges=edges,
        )

    def highest_risk_paths(self, crown_jewel_id: str, top_n: int = 10) -> List[AttackPath]:
        """
        Find the top-N highest-probability attack paths leading to the crown jewel
        from all reachable source nodes. Implements the paper's composite path
        probability analysis for lateral movement assessment.
        """
        paths: List[AttackPath] = []
        for src in self._graph.nodes:
            if src == crown_jewel_id:
                continue
            path = self.minimum_resistance_path(src, crown_jewel_id)
            if path and path.probability > 0:
                paths.append(path)
        return sorted(paths, key=lambda p: p.probability, reverse=True)[:top_n]

    def sensitivity_analysis(self, crown_jewel_id: str) -> Dict[Tuple[str, str], float]:
        """
        For each edge, compute the marginal reduction in attack path probability
        to the crown jewel if that edge were removed (patched/mitigated).
        Guides remediation prioritization per paper Section IV.

        Returns: {(src, dst): marginal_reduction_pct}
        """
        baseline_path = self.minimum_resistance_path(
            self._find_highest_risk_source(crown_jewel_id), crown_jewel_id
        )
        if not baseline_path:
            return {}
        baseline_prob = baseline_path.probability

        marginal_reductions: Dict[Tuple[str, str], float] = {}
        for u, v in list(self._graph.edges()):
            original_prob = self._graph[u][v]["probability"]
            # Simulate edge removal (set probability to near-zero)
            self.update_edge_probability(u, v, _EPSILON)
            counterfactual = self.minimum_resistance_path(
                self._find_highest_risk_source(crown_jewel_id), crown_jewel_id
            )
            if counterfactual:
                reduction = max(0.0, baseline_prob - counterfactual.probability)
            else:
                reduction = baseline_prob  # path eliminated entirely
            marginal_reductions[(u, v)] = reduction / (baseline_prob + _EPSILON)
            # Restore
            self.update_edge_probability(u, v, original_prob)

        return dict(sorted(marginal_reductions.items(), key=lambda x: x[1], reverse=True))

    def _find_highest_risk_source(self, crown_jewel_id: str) -> str:
        """Returns the external node with the highest-probability path to target."""
        best_src, best_prob = crown_jewel_id, 0.0
        for src in self._graph.nodes:
            if src == crown_jewel_id:
                continue
            path = self.minimum_resistance_path(src, crown_jewel_id)
            if path and path.probability > best_prob:
                best_prob = path.probability
                best_src = src
        return best_src

    # ------------------------------------------------------------------
    # Graph stats
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def to_dict(self) -> Dict:
        return {
            "nodes": [
                {"id": n, **self._graph.nodes[n]}
                for n in self._graph.nodes
            ],
            "edges": [
                {"src": u, "dst": v, "probability": self._graph[u][v]["probability"]}
                for u, v in self._graph.edges()
            ],
        }
