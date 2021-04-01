"""
Layer 1 — Security Data Fabric: Entity Correlator
Graph-based entity resolution: links IP ↔ user ↔ device ↔ cloud resource
into a unified entity model. Complexity: O(n log n), O(n+e).
"""
from __future__ import annotations
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from src.common.models import Entity, EntityType, OCSFEvent, Relation, RelationType

logger = logging.getLogger(__name__)


class EntityCorrelator:
    """
    Maintains an in-memory entity graph and performs entity resolution across
    security events. Each node is a canonical Entity; edges are typed Relations.

    Entity resolution rules:
      - Same IP observed with same user in <5min window → link
      - Same device name resolves to same Entity regardless of IP rotation
      - CVE on device → is_vulnerable_to relation to CVE entity
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._alias_index: Dict[str, str] = {}   # alias → canonical entity_id
        self._entities: Dict[str, Entity] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_event(self, event: OCSFEvent) -> List[Tuple[Entity, Optional[Relation]]]:
        """
        Process one normalized event. Returns list of (entity, relation) pairs
        that were created or updated. Relation is None for standalone entities.
        """
        created: List[Tuple[Entity, Optional[Relation]]] = []

        user_entity = device_entity = ip_entity = target_entity = None

        if event.actor_user:
            user_entity = self._upsert(event.actor_user, EntityType.USER, event)
            created.append((user_entity, None))

        if event.actor_device:
            device_entity = self._upsert(event.actor_device, EntityType.DEVICE, event)
            created.append((device_entity, None))

        if event.src_ip:
            ip_entity = self._upsert(event.src_ip, EntityType.IP_ADDRESS, event)
            created.append((ip_entity, None))

        if event.target_resource:
            target_entity = self._upsert(event.target_resource, EntityType.DATA_STORE, event)
            created.append((target_entity, None))

        # Create relations
        if user_entity and device_entity:
            rel = self._upsert_relation(user_entity, device_entity, RelationType.RESIDES_ON)
            created.append((user_entity, rel))

        if user_entity and ip_entity:
            rel = self._upsert_relation(user_entity, ip_entity, RelationType.COMMUNICATES_WITH)
            created.append((user_entity, rel))

        if user_entity and target_entity:
            rel = self._upsert_relation(user_entity, target_entity, RelationType.HAS_ACCESS_TO)
            created.append((user_entity, rel))

        if device_entity and target_entity:
            rel = self._upsert_relation(device_entity, target_entity, RelationType.COMMUNICATES_WITH)
            created.append((device_entity, rel))

        # CVE vulnerability relations
        for cve_id in event.cve_ids:
            cve_entity = self._upsert(cve_id, EntityType.APPLICATION, event)
            cve_entity.attributes["is_cve"] = True
            if device_entity:
                rel = self._upsert_relation(
                    device_entity, cve_entity, RelationType.IS_VULNERABLE_TO, weight=0.9
                )
                created.append((device_entity, rel))

        return created

    def get_entity(self, name_or_id: str) -> Optional[Entity]:
        canonical_id = self._alias_index.get(name_or_id, name_or_id)
        return self._entities.get(canonical_id)

    def get_neighbors(self, entity_id: str, relation_type: Optional[RelationType] = None) -> List[Entity]:
        neighbors = []
        for _, neighbor_id, data in self._graph.out_edges(entity_id, data=True):
            if relation_type is None or data.get("relation_type") == relation_type:
                entity = self._entities.get(neighbor_id)
                if entity:
                    neighbors.append(entity)
        return neighbors

    def entity_count(self) -> int:
        return len(self._entities)

    def relation_count(self) -> int:
        return self._graph.number_of_edges()

    def find_incident_chain(self, ip: str) -> List[Entity]:
        """
        Returns all entities linked to a given IP in the incident chain.
        Implements the paper's example: IP 10.0.0.1 → User_Admin → Laptop_04.
        Traverses both outbound and inbound edges so IP nodes (which are
        targets of user→IP edges) are properly discovered.
        """
        # Try both raw name and prefixed canonical ID
        ip_entity = self.get_entity(ip) or self.get_entity(f"ip_address:{ip}")
        if not ip_entity:
            return []
        chain: List[Entity] = [ip_entity]
        visited: Set[str] = {ip_entity.entity_id}
        queue = [ip_entity.entity_id]
        while queue:
            current_id = queue.pop(0)
            # Traverse outbound edges
            for _, nbr_id, _ in self._graph.out_edges(current_id, data=True):
                if nbr_id not in visited:
                    visited.add(nbr_id)
                    entity = self._entities.get(nbr_id)
                    if entity:
                        chain.append(entity)
                        queue.append(nbr_id)
            # Traverse inbound edges (IP is a destination of user→IP edges)
            for src_id, _, _ in self._graph.in_edges(current_id, data=True):
                if src_id not in visited:
                    visited.add(src_id)
                    entity = self._entities.get(src_id)
                    if entity:
                        chain.append(entity)
                        queue.append(src_id)
        return chain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_entity_id(self, name: str, entity_type: EntityType) -> str:
        return f"{entity_type.value}:{name}"

    def _upsert(self, name: str, entity_type: EntityType, event: OCSFEvent) -> Entity:
        entity_id = self._make_entity_id(name, entity_type)
        canonical_id = self._alias_index.get(entity_id, entity_id)

        if canonical_id not in self._entities:
            entity = Entity(
                entity_id=canonical_id,
                entity_type=entity_type,
                name=name,
                last_seen=event.timestamp,
            )
            self._entities[canonical_id] = entity
            self._graph.add_node(canonical_id, entity_type=entity_type.value)
            self._alias_index[entity_id] = canonical_id
        else:
            entity = self._entities[canonical_id]
            entity.last_seen = event.timestamp
            # propagate severity into risk score (simple EMA)
            severity_norm = (event.severity - 1) / 4.0  # normalize to [0,1]
            entity.risk_score = 0.9 * entity.risk_score + 0.1 * severity_norm

        return entity

    def _upsert_relation(
        self,
        src: Entity,
        dst: Entity,
        relation_type: RelationType,
        weight: float = 1.0,
    ) -> Relation:
        if not self._graph.has_edge(src.entity_id, dst.entity_id):
            self._graph.add_edge(
                src.entity_id,
                dst.entity_id,
                relation_type=relation_type,
                weight=weight,
            )
        else:
            # strengthen existing edge
            current_weight = self._graph[src.entity_id][dst.entity_id].get("weight", 1.0)
            self._graph[src.entity_id][dst.entity_id]["weight"] = min(1.0, current_weight + 0.05)

        return Relation(
            src_entity_id=src.entity_id,
            dst_entity_id=dst.entity_id,
            relation_type=relation_type,
            weight=weight,
        )
