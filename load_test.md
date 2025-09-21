# Performance Baseline

## Purpose
Establish an initial latency/throughput baseline for the classification API to detect regressions over time.

## Tooling
Suggested: Locust (Python) or k6 (JavaScript). Locust skeleton provided below.

## Targets
| Endpoint | Goal | Notes |
|----------|------|-------|
| `GET /health/ready` | p95 < 50ms | Simple readiness check |
| `POST /classify` | p95 < 250ms | With warm model and small payload |
| `GET /version` | p95 < 60ms | Metadata retrieval |

## Test Data Strategy
Use a small rotating list of representative ticket titles/descriptions (mix of urgent, billing, sales) to avoid cache bias.

## Locust Quick Start
Install:
```
pip install locust
```
Run:
```
locust -f locustfile.py --headless -u 50 -r 5 -t 2m --host=http://localhost:8000
```

## Metrics to Capture
- p50 / p95 / p99 latency per endpoint
- RPS sustained
- Error rate (4xx/5xx)
- CPU & memory (external: Docker stats or orchestrator metrics)

## Interpreting Results
Create a baseline JSON (commit once) then compare subsequent runs in CI performance stage (future work).

---
Update this file as baselines evolve or SLOs are formalized.
