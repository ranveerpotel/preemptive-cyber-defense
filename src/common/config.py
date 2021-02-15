"""System-wide configuration."""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SystemConfig:
    # Governance scoring
    gms_poll_interval_sec: int = 60
    gms_threshold_trigger: float = 0.65      # below this → trigger remediation
    mc_dropout_passes: int = 100             # Monte Carlo uncertainty estimation
    confidence_multiplier_k: float = 1.96   # 95% CI

    # Action confidence thresholds
    autonomous_confidence_threshold: float = 0.60
    hitl_confidence_threshold: float = 0.80  # below this for HIGH impact → HITL

    # Control Barrier Function params
    max_service_downtime_pct: float = 0.05   # 5% max production downtime
    blast_radius_threshold: float = 0.10    # >10% assets → mandatory HITL
    post_action_monitor_window_sec: int = 900  # 15 minutes

    # RL / PPO
    rl_discount_factor: float = 0.99
    rl_learning_rate: float = 3e-4
    rl_training_buffer_size: int = 10_000

    # TransE embedding
    embedding_dim: int = 128
    embedding_margin: float = 1.0

    # Attack graph
    attack_graph_dijkstra_log_base: float = 10.0

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_raw: str = "raw.events"
    kafka_topic_normalized: str = "normalized.events"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # TimescaleDB
    timescale_dsn: str = "postgresql://postgres:password@localhost:5432/cyberdefense"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333


DEFAULT_CONFIG = SystemConfig()
