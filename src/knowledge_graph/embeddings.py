"""
TransE-variant graph embeddings for the Security Knowledge Graph.
Trained on historical SKG snapshots to detect subtle behavioral anomalies
that violate no explicit rule but deviate from learned baselines.

Paper reference: Section V — embedding learning applied to SKG for anomaly
detection. Uses TransE scoring: ||h + r - t||₂ < margin for valid triples.
Complexity: O(d · |R|) inference, O(|E| · d) space.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class TransEModel(nn.Module):
    """
    TransE knowledge graph embedding model.
    For valid triples (h, r, t): h + r ≈ t
    Scoring: f(h, r, t) = ||h + r - t||₂ (lower = more valid)
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 128, margin: float = 1.0) -> None:
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.embedding_dim = embedding_dim
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        # Normalize entity embeddings to unit sphere
        with torch.no_grad():
            self.entity_embeddings.weight.data = nn.functional.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )

    def score(self, h_idx: torch.Tensor, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Lower score = more valid triple."""
        h = self.entity_embeddings(h_idx)
        r = self.relation_embeddings(r_idx)
        t = self.entity_embeddings(t_idx)
        return torch.norm(h + r - t, p=2, dim=1)

    def forward(
        self,
        pos_h: torch.Tensor,
        pos_r: torch.Tensor,
        pos_t: torch.Tensor,
        neg_h: torch.Tensor,
        neg_r: torch.Tensor,
        neg_t: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-based ranking loss."""
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        loss = torch.clamp(self.margin + pos_score - neg_score, min=0.0).mean()
        return loss


class SKGEmbeddingTrainer:
    """
    Manages TransE training on SKG triples and provides anomaly scoring.
    Implements the paper's behavioral baseline learning for detecting subtle
    access anomalies (user accessing DB at unusual hour from unfamiliar segment).
    """

    def __init__(self, embedding_dim: int = 128, margin: float = 1.0, lr: float = 1e-3) -> None:
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.lr = lr
        self._model: Optional[TransEModel] = None
        self._entity_to_idx: Dict[str, int] = {}
        self._relation_to_idx: Dict[str, int] = {}
        self._idx_to_entity: Dict[int, str] = {}
        self._trained = False

    def build_vocabulary(self, triples: List[Tuple[str, str, str]]) -> None:
        entities: set = set()
        relations: set = set()
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        self._entity_to_idx = {e: i for i, e in enumerate(sorted(entities))}
        self._idx_to_entity = {i: e for e, i in self._entity_to_idx.items()}
        self._relation_to_idx = {r: i for i, r in enumerate(sorted(relations))}
        self._model = TransEModel(
            num_entities=len(entities),
            num_relations=len(relations),
            embedding_dim=self.embedding_dim,
            margin=self.margin,
        )

    def train(self, triples: List[Tuple[str, str, str]], epochs: int = 100, batch_size: int = 256) -> List[float]:
        if self._model is None:
            self.build_vocabulary(triples)
        assert self._model is not None

        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        indexed = self._index_triples(triples)
        loss_history: List[float] = []

        self._model.train()
        for epoch in range(epochs):
            np.random.shuffle(indexed)
            epoch_loss = 0.0
            for batch_start in range(0, len(indexed), batch_size):
                batch = indexed[batch_start: batch_start + batch_size]
                pos_h = torch.tensor([t[0] for t in batch], dtype=torch.long)
                pos_r = torch.tensor([t[1] for t in batch], dtype=torch.long)
                pos_t = torch.tensor([t[2] for t in batch], dtype=torch.long)
                # Negative sampling: corrupt head or tail randomly
                neg_h, neg_r, neg_t = self._corrupt(pos_h, pos_r, pos_t)

                optimizer.zero_grad()
                loss = self._model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
                loss.backward()
                optimizer.step()
                # Renormalize entity embeddings
                with torch.no_grad():
                    self._model.entity_embeddings.weight.data = nn.functional.normalize(
                        self._model.entity_embeddings.weight.data, p=2, dim=1
                    )
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(1, len(indexed) // batch_size)
            loss_history.append(avg_loss)
            if epoch % 10 == 0:
                logger.info("TransE epoch %d/%d — loss: %.4f", epoch, epochs, avg_loss)

        self._trained = True
        return loss_history

    def anomaly_score(self, head: str, relation: str, tail: str) -> float:
        """
        Returns TransE score for a triple. Higher = more anomalous.
        Implements paper's behavioral distance from learned centroid.
        """
        if not self._trained or self._model is None:
            return 0.0
        h_idx = self._entity_to_idx.get(head)
        r_idx = self._relation_to_idx.get(relation)
        t_idx = self._entity_to_idx.get(tail)
        if any(idx is None for idx in [h_idx, r_idx, t_idx]):
            return float("inf")  # unknown entity = maximally anomalous
        self._model.eval()
        with torch.no_grad():
            score = self._model.score(
                torch.tensor([h_idx]),
                torch.tensor([r_idx]),
                torch.tensor([t_idx]),
            )
        return float(score.item())

    def entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        if not self._trained or self._model is None:
            return None
        idx = self._entity_to_idx.get(entity_id)
        if idx is None:
            return None
        self._model.eval()
        with torch.no_grad():
            emb = self._model.entity_embeddings(torch.tensor([idx]))
        return emb.numpy()[0]

    def _index_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[int, int, int]]:
        result = []
        for h, r, t in triples:
            h_idx = self._entity_to_idx.get(h)
            r_idx = self._relation_to_idx.get(r)
            t_idx = self._entity_to_idx.get(t)
            if all(idx is not None for idx in [h_idx, r_idx, t_idx]):
                result.append((h_idx, r_idx, t_idx))
        return result

    def _corrupt(
        self,
        pos_h: torch.Tensor,
        pos_r: torch.Tensor,
        pos_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform negative sampling: corrupt head or tail."""
        n = pos_h.size(0)
        num_entities = len(self._entity_to_idx)
        corrupt_head = torch.rand(n) < 0.5
        random_entities = torch.randint(0, num_entities, (n,))
        neg_h = torch.where(corrupt_head, random_entities, pos_h)
        neg_t = torch.where(~corrupt_head, random_entities, pos_t)
        return neg_h, pos_r.clone(), neg_t
