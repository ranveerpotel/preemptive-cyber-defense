"""
Security Knowledge Graph (SKG): heterogeneous property graph SKG = (E, R, A).
  E = entity set (users, devices, apps, data stores, network segments, cloud services)
  R = relation set (typed relationships)
  A = attribute set with historical distributions

Serves as the organizational memory of the security program, enabling cross-domain
risk chain detection invisible to single-domain tools.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.common.models import Entity, EntityType, Relation, RelationType

logger = logging.getLogger(__name__)


class SecurityKnowledgeGraph:
    """
    Heterogeneous property graph implementing the paper's SKG specification.

    Example knowledge path the paper describes:
      "User Alice has_access_to Database DB-Finance, which resides_on
       CloudServer AWS-East-3, which is_vulnerable_to CVE-2020-XXXX."
    """

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entities: Dict[str, Entity] = {}
        self._snapshots: List[Dict] = []   # historical snapshots for embedding training

    # ------------------------------------------------------------------
    # Entity / Relation CRUD
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.entity_id] = entity
        self._graph.add_node(
            entity.entity_id,
            entity_type=entity.entity_type.value,
            name=entity.name,
            risk_score=entity.risk_score,
            **entity.attributes,
        )

    def update_entity_risk(self, entity_id: str, risk_score: float) -> None:
        if entity_id in self._entities:
            self._entities[entity_id].risk_score = risk_score
            self._graph.nodes[entity_id]["risk_score"] = risk_score

    def add_relation(self, relation: Relation) -> None:
        self._graph.add_edge(
            relation.src_entity_id,
            relation.dst_entity_id,
            key=relation.relation_type.value,
            relation_type=relation.relation_type.value,
            weight=relation.weight,
            **relation.attributes,
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def neighbors(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> List[Entity]:
        result = []
        for _, dst, data in self._graph.out_edges(entity_id, data=True, keys=False):
            if relation_type is None or data.get("relation_type") == relation_type.value:
                entity = self._entities.get(dst)
                if entity:
                    result.append(entity)
        return result

    # ------------------------------------------------------------------
    # Risk chain traversal
    # ------------------------------------------------------------------

    def find_risk_chains(self, start_entity_id: str, max_depth: int = 5) -> List[List[str]]:
        """
        BFS/DFS to find all risk propagation chains from a starting entity.
        Used to surface cross-domain risk invisible to single-domain tools.
        Returns list of entity_id chains.
        """
        chains: List[List[str]] = []
        queue: List[Tuple[str, List[str]]] = [(start_entity_id, [start_entity_id])]
        visited_paths: Set[str] = set()

        while queue:
            current_id, path = queue.pop(0)
            if len(path) >= max_depth:
                chains.append(path)
                continue
            successors = list(self._graph.successors(current_id))
            if not successors:
                chains.append(path)
                continue
            for nbr_id in successors:
                chain_key = "->".join(path + [nbr_id])
                if chain_key not in visited_paths and nbr_id not in path:
                    visited_paths.add(chain_key)
                    queue.append((nbr_id, path + [nbr_id]))

        return [c for c in chains if len(c) > 1]

    def vulnerability_exposure_chains(self, cve_entity_id: str) -> List[List[str]]:
        """
        Given a CVE entity, find all entities exposed through the vulnerability.
        Traverses IS_VULNERABLE_TO relations inbound to cve_entity_id.
        """
        exposed: List[List[str]] = []
        for src_id, _, data in self._graph.in_edges(cve_entity_id, data=True):
            if data.get("relation_type") == RelationType.IS_VULNERABLE_TO.value:
                # follow upward access chains from the vulnerable device
                chains = self.find_risk_chains(src_id)
                exposed.extend(chains)
        return exposed

    # ------------------------------------------------------------------
    # Snapshot management (for TransE embedding training)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict:
        """Capture current graph state as a training snapshot for embeddings."""
        snap = {
            "timestamp": datetime.utcnow().isoformat(),
            "triples": [
                (u, data.get("relation_type", "unknown"), v)
                for u, v, data in self._graph.edges(data=True)
            ],
            "entity_risks": {eid: e.risk_score for eid, e in self._entities.items()},
        }
        self._snapshots.append(snap)
        return snap

    def get_triples(self) -> List[Tuple[str, str, str]]:
        """Returns all (head, relation, tail) triples for embedding training."""
        return [
            (u, data.get("relation_type", "unknown"), v)
            for u, v, data in self._graph.edges(data=True)
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def entity_count(self) -> int:
        return len(self._entities)

    def relation_count(self) -> int:
        return self._graph.number_of_edges()

    def high_risk_entities(self, threshold: float = 0.7) -> List[Entity]:
        return [e for e in self._entities.values() if e.risk_score >= threshold]
