"""
OCSF-aligned domain models and shared types for the Preemptive Cyber Defense system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid


class EventSource(str, Enum):
    CROWDSTRIKE = "crowdstrike"
    OKTA = "okta"
    AWS_CLOUDTRAIL = "aws_cloudtrail"
    CISCO = "cisco"
    TENABLE = "tenable"
    GENERIC = "generic"


class EntityType(str, Enum):
    USER = "user"
    DEVICE = "device"
    APPLICATION = "application"
    DATA_STORE = "data_store"
    NETWORK_SEGMENT = "network_segment"
    CLOUD_SERVICE = "cloud_service"
    IP_ADDRESS = "ip_address"


class RelationType(str, Enum):
    HAS_ACCESS_TO = "has_access_to"
    RESIDES_ON = "resides_on"
    COMMUNICATES_WITH = "communicates_with"
    IS_VULNERABLE_TO = "is_vulnerable_to"
    IS_MEMBER_OF = "is_member_of"


class ActionImpact(str, Enum):
    LOW = "low"        # ticket, notification — autonomous
    MEDIUM = "medium"  # port close, traffic filter — autonomous if confident
    HIGH = "high"      # device isolate, account suspend — HITL required


class ActionStatus(str, Enum):
    PENDING = "pending"
    AWAITING_HITL = "awaiting_hitl"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OCSFEvent:
    """Open Cybersecurity Schema Framework normalized event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: EventSource = EventSource.GENERIC
    category: str = ""
    activity_name: str = ""
    severity: int = 1          # 1=info, 2=low, 3=medium, 4=high, 5=critical
    actor_user: Optional[str] = None
    actor_device: Optional[str] = None
    target_resource: Optional[str] = None
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    cve_ids: List[str] = field(default_factory=list)
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    entity_id: str
    entity_type: EntityType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    last_seen: Optional[datetime] = None


@dataclass
class Relation:
    src_entity_id: str
    dst_entity_id: str
    relation_type: RelationType
    attributes: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class GovernanceControl:
    control_id: str
    name: str
    domain: str           # identity, network, endpoint, cloud, data
    weight: float         # criticality weight
    effectiveness: float  # current effectiveness score ∈ [0,1]
    exposure: float       # current exposure surface ∈ [0,1]
    cost: float           # implementation cost (normalized)
    evidence: List[str] = field(default_factory=list)


@dataclass
class GMSResult:
    score: float           # GMS_point
    lower_bound: float     # GMS_point - k·σ
    upper_bound: float     # GMS_point + k·σ
    std_dev: float         # σ(GMS) from MC dropout
    confidence: float      # posterior probability
    timestamp: datetime = field(default_factory=datetime.utcnow)
    control_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class RemediationAction:
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = ""
    target_entity_id: str = ""
    impact: ActionImpact = ActionImpact.LOW
    status: ActionStatus = ActionStatus.PENDING
    confidence: float = 0.0
    estimated_risk_reduction: float = 0.0
    actual_risk_reduction: Optional[float] = None
    shap_rationale: Optional[str] = None
    cbf_satisfied: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    api_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackPath:
    path: List[str]          # ordered node IDs
    probability: float       # composite path probability
    edges: List[Tuple[str, str, float]]  # (src, dst, edge_prob)


@dataclass
class RiskForecast:
    current_gms: float
    forecast_7d: List[float]    # daily GMS projections
    trend: str                   # "improving", "stable", "declining"
    top_risk_drivers: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)
