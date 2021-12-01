# Technical Design Document
## Preemptive Cyber Defense Architecture
### Based on: "A Data-Driven Architecture for Preemptive Cyber Defense Using AI-Based Governance and Autonomous Remediation" (Potel, IJEETR 2021)

---

## 1. Executive Summary

This document specifies the implementation design for a four-layer Preemptive Cyber Defense system that transforms raw security telemetry into governed, predictive, and autonomously remediable insight. The architecture replaces reactive SIEM/SOAR/GRC tooling with a continuously optimizing AI system.

**Target outcomes (from paper's experimental results):**
- 85% reduction in monthly reporting time (200h → 30h)
- 79% reduction in mean time to remediation (14 days → 3 days)
- 40% increase in hidden risk discovery
- 61% reduction in false positive rate
- Real-time governance scoring (vs. monthly manual cycles)

---

## 2. Tech Stack Decision

### Evaluated Options

| Criteria | Option A: Python Microservices | Option B: Go + Python Hybrid | Option C: Cloud-Native (AWS) |
|---|---|---|---|
| ML ecosystem | Excellent (PyTorch, SB3, SHAP) | Poor (Go has no RL libraries) | Moderate (SageMaker limits) |
| Graph processing | Excellent (NetworkX, Neo4j) | Moderate | Moderate (Neptune) |
| Streaming throughput | Good (Kafka + Faust) | Excellent | Excellent (Kinesis) |
| Dev velocity | Fast | Slow (two languages) | Medium (lock-in) |
| RL/PPO support | Native (Stable-Baselines3) | None | None |
| SHAP/XAI | Native library | None | None |
| Operational complexity | Low | High | Medium |
| Cost at scale | Low | Medium | High |

### Decision: **Option A — Python Microservices**

**Rationale:** The paper's core algorithms (PPO reinforcement learning, TransE graph embeddings, SHAP explainability, Monte Carlo dropout uncertainty, Control Barrier Functions) all have mature Python libraries. Go's throughput advantage at the ingestion layer does not outweigh the cost of maintaining two ML runtimes. A Python async stack with Kafka for streaming meets the paper's stated scale (10M+ events/day) within acceptable latency bounds (<1s governance scoring).

### Selected Stack

```
Runtime:        Python 3.11
API Framework:  FastAPI + Uvicorn (async, <5ms p99 overhead)
Event Stream:   Apache Kafka (10M+ events/day, partitioned by source)
Graph DB:       Neo4j 5.x (native property graph, Cypher queries)
Time-series DB: TimescaleDB (PostgreSQL extension, temporal tracking)
Cache:          Redis 7 (governance scores, hot entity state)
Vector Store:   Qdrant (TransE embedding similarity search)
ML Framework:   PyTorch 2.x (TransE embeddings, neural scoring)
RL Engine:      Stable-Baselines3 (PPO governance optimizer)
XAI:            SHAP 0.44 (TreeExplainer + KernelExplainer)
Graph Analysis: NetworkX (attack graph, Dijkstra)
Uncertainty:    MC-Dropout via PyTorch (GMS confidence bounds)
Containerization: Docker + Docker Compose (dev), Kubernetes (prod)
Monitoring:     Prometheus + Grafana
```

---

## 3. Four-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: Executive Intelligence Layer                          │
│  FastAPI dashboard  │  Audit trails  │  Board-level reporting   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ GMS scores, forecasts, actions
┌──────────────────────────────▼──────────────────────────────────┐
│  LAYER 3: Agentic Remediation Layer                             │
│  RL Policy (PPO)  │  CBF safety  │  HITL queue  │  SHAP XAI    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ posture state, triggers
┌──────────────────────────────▼──────────────────────────────────┐
│  LAYER 2: AI Governance Engine                                  │
│  GMS scorer  │  Risk forecaster  │  Attack graph  │  SKG embed  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ normalized OCSF events
┌──────────────────────────────▼──────────────────────────────────┐
│  LAYER 1: Security Data Fabric (SDF)                            │
│  Schema normalizer  │  Entity correlator  │  Temporal tracker   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ raw telemetry
        ┌──────────────────────┴───────────────────┐
        │  Sources: SIEM / EDR / IAM / Cloud / Vuln │
        └──────────────────────────────────────────┘
```

---

## 4. Component Specifications

### 4.1 Security Data Fabric (Layer 1)

**Schema Normalization**
- Input: vendor-specific JSON (CrowdStrike, Okta, AWS CloudTrail, Cisco, Tenable)
- Output: OCSF v1.0 canonical event objects
- Complexity: O(n) time, O(n) space — 10M+ events/day
- Implementation: field-mapping registry per vendor, async Kafka consumer

**Entity Correlation**
- Graph-based entity resolution linking IP ↔ user ↔ device ↔ cloud resource
- Complexity: O(n log n) time, O(n + e) space — 100K+ entities
- Implementation: NetworkX for in-memory graph, Neo4j for persistence

**Temporal Tracking**
- Time-series storage of risk indicators per entity in TimescaleDB
- Sliding window anomaly detection for drift before threshold breach
- Implementation: TimescaleDB hypertables, 60s polling interval

### 4.2 AI Governance Engine (Layer 2)

**Governance Maturity Score (GMS)**

```
GMS = Σ(wᵢ · eᵢ) / Σ(wᵢ · exposureᵢ)

where:
  eᵢ = control effectiveness score ∈ [0,1]
  wᵢ = control weight (criticality)
  exposureᵢ = current exposure surface for control domain
```

**Robust GMS with Uncertainty**
```
GMS_robust = GMS_point ± k · σ(GMS)

where:
  GMS_point = point estimate
  σ(GMS)    = std dev via Monte Carlo dropout (N=100 forward passes)
  k         = confidence multiplier (default: 1.96 for 95% CI)
```

**Attack Graph**
- Directed graph G = (V, E, P): assets as nodes, attack paths as edges
- Edge probability P: E → [0,1] = likelihood of successful traversal
- Path probability = product of edge probabilities (min-resistance path)
- Modified Dijkstra on log-transformed probabilities (multiplicative → additive)

**Security Knowledge Graph (SKG)**
- Heterogeneous property graph SKG = (E, R, A)
- E: entities (users, devices, apps, data stores, network segments, cloud services)
- R: relations (has_access_to, resides_on, communicates_with, is_vulnerable_to, is_member_of)
- A: attribute set with historical distributions
- TransE-variant embeddings trained on historical SKG snapshots (embedding dim d=128)

**Reinforcement Learning Governance Optimizer**
- MDP: M = (S, A, T, R, γ)
- S: posture state (control effectiveness scores, exposure levels, vulnerability counts)
- A: governance interventions (enable MFA, patch, close port, increase monitoring)
- Algorithm: Proximal Policy Optimization (PPO) with decaying learning rate
- Convergence: E[GMS(t)] → GMS* at rate O(1/√t)

**Budget-Constrained Control Optimization**
```
Maximize: Σ ΔR(cᵢ) · xᵢ
Subject to: Σ Cost(cᵢ) · xᵢ ≤ B, xᵢ ∈ [0,1]

where:
  ΔR(cᵢ) = marginal risk reduction of control cᵢ
  xᵢ     = implementation level (continuous relaxation)
  B      = available security budget
```

### 4.3 Agentic Remediation Layer (Layer 3)

**Action Taxonomy**
| Impact | Actions | Auth Required |
|---|---|---|
| Low | Ticket creation, notification | Autonomous |
| Medium | Port closure, traffic filtering | Autonomous (high confidence) |
| High | Device isolation, account suspension | HITL approval |

**Control Barrier Functions (CBF)**
```
h: S → ℝ defines safe set C = {s ∈ S | h(s) ≥ 0}
Constraint: any action must maintain h(s') ≥ 0

Safety constraints:
  1. Max tolerable service downtime threshold
  2. Change window enforcement (outside critical incidents)
  3. Blast radius limit: >threshold% assets → mandatory HITL
  4. Reversibility preference
```

**Confidence Thresholds**
- posterior_probability < 0.6 → escalate to HITL regardless of impact
- posterior_probability ≥ 0.6 AND CBF satisfied → autonomous execution
- All actions: SHAP rationale report generated

**SHAP Explainability**
- Each action accompanied by machine-generated Rationale Report
- SHAP values φᵢ = weighted avg marginal contribution of feature i across all subsets
- Format: "Action taken because [feature] [SHAP: x.xx], [feature] [SHAP: x.xx], ..."

### 4.4 Executive Intelligence Layer (Layer 4)

**API Endpoints**
- `GET /governance/score` — current GMS with confidence interval
- `GET /governance/forecast` — 7-day forward projection
- `GET /risks/active` — ranked active risk indicators
- `GET /remediation/active` — in-progress autonomous actions
- `GET /remediation/hitl` — pending human approval queue
- `POST /remediation/approve/{action_id}` — HITL approval
- `GET /audit/trail` — immutable action audit log
- `GET /attack-graph` — current attack graph with path probabilities

---

## 5. Data Flow

```
1. Telemetry sources → Kafka topic: raw.events
2. SDF normalizer (consumer) → OCSF events → Kafka: normalized.events
3. Entity correlator (consumer) → updates Neo4j + Redis entity cache
4. Temporal tracker → writes risk indicators to TimescaleDB
5. Governance Engine (60s poll):
   a. Queries Neo4j + TimescaleDB for current state
   b. Computes GMS with MC-Dropout uncertainty
   c. Updates attack graph edge probabilities
   d. Runs SKG embedding update (incremental)
   e. Calls RL policy for governance recommendations
   f. Publishes GMS state to Redis + Executive Dashboard
6. If GMS < threshold → trigger Remediation Agent
7. Remediation Agent:
   a. Queries RL policy π for ranked action set A'
   b. Evaluates each action against CBF constraints
   c. Low confidence → HITL queue
   d. High confidence + safe → autonomous execution via security APIs
   e. Generates SHAP rationale report
   f. Monitors post-action telemetry (15min window)
   g. Submits (s, a, ΔR_actual, s') to RL training buffer
8. All actions → immutable audit log
9. Executive Dashboard → real-time SSE stream of GMS, forecasts, actions
```

---

## 6. Scalability Analysis

| Component | Time Complexity | Space Complexity | Target Scale |
|---|---|---|---|
| Schema Normalization | O(n) | O(n) | 10M+ events/day |
| Entity Correlation | O(n log n) | O(n + e) | 100K+ entities |
| SKG Embedding | O(d · \|R\|) | O(\|E\| · d) | 1M+ relations |
| Governance Scoring | O(k · n) | O(k) | Real-time (<1s) |
| Attack Graph Analysis | O(\|V\| + \|E\|) | O(\|V\|) | 10K+ nodes |
| RL Policy Inference | O(d²) | O(d) | Sub-millisecond |

---

## 7. Directory Structure

```
preemptiveCyberDefence/
├── docs/
│   └── TECHNICAL_DESIGN.md        (this document)
├── src/
│   ├── common/
│   │   ├── models.py               OCSF event models + domain types
│   │   └── config.py               system configuration
│   ├── layer1_sdf/
│   │   ├── normalizer.py           vendor → OCSF normalization
│   │   ├── entity_correlator.py    graph-based entity resolution
│   │   └── temporal_tracker.py     time-series risk tracking
│   ├── layer2_governance/
│   │   ├── gms_scorer.py           GMS computation + MC uncertainty
│   │   ├── risk_forecaster.py      7-day forward projection
│   │   ├── attack_graph.py         directed graph + modified Dijkstra
│   │   └── rl_optimizer.py         PPO MDP governance optimizer
│   ├── layer3_remediation/
│   │   ├── remediation_agent.py    agentic action selection
│   │   ├── cbf.py                  Control Barrier Functions
│   │   ├── hitl_queue.py           Human-in-the-Loop escalation
│   │   └── xai_explainer.py        SHAP rationale generation
│   ├── layer4_executive/
│   │   └── dashboard_api.py        FastAPI executive dashboard
│   └── knowledge_graph/
│       ├── skg.py                  Security Knowledge Graph
│       ├── attack_graph.py         attack path computation
│       └── embeddings.py           TransE graph embeddings
├── requirements.txt
└── docker-compose.yml
```

---

## 8. Deployment Architecture

```
┌──────────────────────────────────────────────────────┐
│  Kubernetes Cluster (prod) / Docker Compose (dev)    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Kafka   │  │  Neo4j   │  │  TimescaleDB     │  │
│  │ (stream) │  │  (graph) │  │  (time-series)   │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Redis   │  │  Qdrant  │  │  Prometheus      │  │
│  │  (cache) │  │ (vectors)│  │  + Grafana       │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Application Services (Python)                  │ │
│  │  sdf-service | governance-service | remediation │ │
│  │  executive-api | skg-service                    │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```
