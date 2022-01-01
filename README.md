# Preemptive Cyber Defense Architecture

> **Python implementation** of the peer-reviewed research paper:
>
> Potel, R. (2021). *A Data-Driven Architecture for Preemptive Cyber Defense Using AI-Based Governance and Autonomous Remediation.* International Journal of Engineering & Extended Technologies Research (IJEETR), Vol. 3, Issue 6, pp. 4053–4062.
> **DOI:** [10.15662/IJEETR.2021.0306010](https://doi.org/10.15662/IJEETR.2021.0306010)

---

## Overview

Modern cybersecurity programs remain largely reactive despite extensive investment in detection and response technologies. Fragmented tooling, delayed reporting cycles, and limited executive visibility prevent organizations from managing cybersecurity as a predictive and governable system.

This project implements a **data-driven preemptive cyber defense architecture** that:

- Integrates heterogeneous security telemetry into a **unified Security Data Fabric**
- Applies **AI-driven governance analytics** for continuous, scored risk visibility
- Executes **bounded, auditable remediation actions** safely and autonomously

---

## Four-Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│  Layer 4: Executive Intelligence                     │
│  Real-time dashboard · Audit trails · Board reports  │
├──────────────────────────────────────────────────────┤
│  Layer 3: Agentic Remediation                        │
│  PPO policy · CBF safety · HITL queue · SHAP XAI    │
├──────────────────────────────────────────────────────┤
│  Layer 2: AI Governance Engine                       │
│  GMS scoring · Risk forecasting · Attack graph · RL  │
├──────────────────────────────────────────────────────┤
│  Layer 1: Security Data Fabric                       │
│  OCSF normalization · Entity correlation · Drift     │
└──────────────────────────────────────────────────────┘
```

| Layer | Component | Primary Function |
|---|---|---|
| 1 | Security Data Fabric | Normalize and correlate heterogeneous telemetry |
| 2 | AI Governance Engine | Continuous posture scoring and risk forecasting |
| 3 | Agentic Remediation | Safety-bounded autonomous action execution |
| 4 | Executive Intelligence | Real-time dashboard and audit reporting |

---

## Key Algorithms

### Governance Maturity Score (GMS)
```
GMS       = Σ(wᵢ · eᵢ) / Σ(wᵢ · exposureᵢ)
GMS_robust = GMS_point ± k · σ(GMS)
```
Uncertainty bounds estimated via Monte Carlo Dropout (N=100 forward passes).

### Attack Graph (Section IV)
Directed graph G = (V, E, P) with modified Dijkstra on log-transformed edge probabilities (multiplicative → additive). Sensitivity analysis identifies the minimum control set that collapses highest-risk attack paths.

### Security Knowledge Graph — TransE Embeddings (Section V)
Heterogeneous property graph SKG = (E, R, A) with TransE-variant embeddings trained on historical snapshots to detect subtle behavioral anomalies invisible to single-domain tools.

### RL Governance Optimizer (Section VI)
MDP M = (S, A, T, R, γ) solved with **Proximal Policy Optimization (PPO)**. Convergence theorem: E[GMS(t)] → GMS* at rate O(1/√t).

### Budget-Constrained Control Optimization (Section VII)
```
Maximize:  Σ ΔR(cᵢ) · xᵢ
Subject to: Σ Cost(cᵢ) · xᵢ ≤ B,  xᵢ ∈ [0, 1]
```

### Control Barrier Functions (Section X)
Structural safety constraints prevent autonomous remediation from causing unintended operational harm — blast radius limits, change window enforcement, reversibility preference, and max service downtime thresholds.

---

## Experimental Results

| Metric | Traditional SOC | This Architecture | Improvement |
|---|---|---|---|
| Monthly Reporting Time | 200 hours | 30 hours | **85% reduction** |
| Mean Time to Remediation | 14 days | 3 days | **79% reduction** |
| Hidden Risk Discovery | Baseline | +40% | **+40% more risks found** |
| Governance Score Accuracy | N/A (manual) | 94.2% | **Quantified for first time** |
| False Positive Rate | 31% | 12% | **61% reduction** |
| Executive Report Latency | Monthly | Real-time | **Continuous visibility** |

---

## Project Structure

```
preemptive-cyber-defense/
├── docs/
│   └── TECHNICAL_DESIGN.md        Technical design document
├── src/
│   ├── common/
│   │   ├── models.py               OCSF event models and domain types
│   │   └── config.py               System-wide configuration
│   ├── knowledge_graph/
│   │   ├── attack_graph.py         Directed attack graph + modified Dijkstra
│   │   ├── skg.py                  Security Knowledge Graph (SKG)
│   │   └── embeddings.py           TransE graph embeddings
│   ├── layer1_sdf/
│   │   ├── normalizer.py           Vendor → OCSF schema normalization
│   │   ├── entity_correlator.py    Graph-based entity resolution
│   │   └── temporal_tracker.py     Time-series drift detection
│   ├── layer2_governance/
│   │   ├── gms_scorer.py           GMS + Monte Carlo uncertainty
│   │   ├── rl_optimizer.py         PPO reinforcement learning optimizer
│   │   └── risk_forecaster.py      7-day forward projection
│   ├── layer3_remediation/
│   │   ├── remediation_agent.py    Agentic action pipeline (Section 15.2)
│   │   ├── cbf.py                  Control Barrier Functions
│   │   ├── hitl_queue.py           Human-in-the-Loop escalation queue
│   │   └── xai_explainer.py        SHAP rationale reports
│   ├── layer4_executive/
│   │   └── dashboard_api.py        FastAPI executive dashboard
│   └── main.py                     Orchestration entry point
├── tests/
│   └── test_smoke.py               Core logic tests (10 tests, all passing)
├── docker-compose.yml              Full infrastructure stack
└── requirements.txt                Python dependencies
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Event Streaming | Apache Kafka |
| Graph Database | Neo4j 5.x |
| Time-Series DB | TimescaleDB (PostgreSQL) |
| Cache | Redis 7 |
| Vector Store | Qdrant |
| ML Framework | PyTorch 2.x |
| RL Engine | Stable-Baselines3 (PPO) |
| XAI | SHAP |
| Graph Analysis | NetworkX |
| Containerization | Docker + Docker Compose |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker + Docker Compose

### Run with Docker
```bash
docker-compose up
```

### Run locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

Executive Dashboard available at: `http://localhost:8000/docs`

### Run tests
```bash
pytest tests/test_smoke.py -v
```

---

## API Endpoints (Layer 4)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/governance/score` | Current GMS with confidence interval |
| GET | `/governance/forecast` | 7-day forward GMS projection |
| GET | `/risks/active` | Ranked active risk indicators |
| GET | `/remediation/active` | In-progress autonomous actions |
| GET | `/remediation/hitl` | Pending human approval queue |
| POST | `/remediation/approve/{id}` | HITL action approval |
| POST | `/remediation/reject/{id}` | HITL action rejection |
| GET | `/audit/trail` | Immutable action audit log |
| GET | `/attack-graph` | Attack graph with path probabilities |

---

## Reference

**Journal Paper:**
Potel, R. (2021). A Data-Driven Architecture for Preemptive Cyber Defense Using AI-Based Governance and Autonomous Remediation. *International Journal of Engineering & Extended Technologies Research (IJEETR)*, 3(6), 4053–4062.
[https://doi.org/10.15662/IJEETR.2021.0306010](https://doi.org/10.15662/IJEETR.2021.0306010)

**ISSN:** 2322-0163 | Published: November–December 2021

---

## License

This implementation is provided for research and educational purposes in alignment with the published paper.
