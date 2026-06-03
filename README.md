# Discrete-Event Simulator (DES)

A fast, accurate simulator for queueing networks and LLM inference scheduling written in Rust.

## Overview

The simulator implements a core discrete-event loop with a priority queue (min-heap) of future events. This enables detailed modeling of:

- **Queueing networks**: M/M/1, M/G/1, M/M/k, finite-buffer systems, Jackson networks
- **LLM inference**: two-phase request lifecycle (prefill + per-token decode) with KV-cache constraints and pluggable scheduling policies

Both domains share the same underlying engine; the LLM work is a domain-specific layer on top of the queueing core.

## Quick Start

Run all validations:
```bash
cargo run
```

Run a specific validation:
```bash
cargo run -- mm1                  # M/M/1 queue validation
cargo run -- pk                   # P-K formula: effect of service variance
cargo run -- srpt                 # SRPT vs FCFS on Pareto workload
cargo run -- mmk                  # M/M/k multi-server queue
cargo run -- mmk-finite           # M/M/1/K finite buffer
cargo run -- jackson              # Jackson open network (product-form)
cargo run -- littles-law          # Little's Law (L = λT)
cargo run -- utilization          # Utilization law (ρ = λS)
cargo run -- response-time-law    # Response time law (R = S + W)
cargo run -- scaling-with-servers # Multi-server throughput scaling
cargo run -- llm                  # LLM inference scheduler comparison
cargo run -- llm-ttft             # TTFT collapse at high load
cargo run -- llm-prompt-length    # Effect of prompt length distribution
cargo run -- llm-batch-tradeoff   # Batch size vs latency tradeoffs
cargo run -- llm-kv-bottleneck    # KV-cache utilization analysis
cargo run -- llm-fairness         # Tail latency fairness (short vs long prompts)
cargo run -- llm-chunked-prefill  # Chunked prefill vs whole-prompt on P99 TTFT
cargo run -- llm-decode-stall     # Decode freeze: Orca vs Sarathi-Serve (P99 step latency)
```

## Validation Results

### Operational Laws: Simulation vs Theory

| Scenario | Max Error | Theory |
|----------|-----------|--------|
| M/M/1 (ρ: 0.5→0.99) | 0.04% | `E[T] = 1/(μ - λ)`, `E[N] = ρ/(1-ρ)` |
| P-K Formula (Erlang, Exp, Hyperexp) | 0.05% | `E[T] = E[S] + λE[S²]/(2(1-ρ))` |
| SRPT vs FCFS (Pareto, ρ=0.5→0.9) | <0.1% | SRPT provably optimal for M/G/1 |
| M/M/k (k=1,2,4,8) | 0.02% | Erlang-C formula |
| Jackson Network (open, 3 nodes) | 0.08% | Product-form solution |
| Response Time Law (all systems) | 0% | R = S + W is an identity |

### LLM Scheduling: Request-Level vs Iteration-Level / Orca (λ=2.0 reqs/sec)

Shared-GPU model: prefill and decode run sequentially in GPU iterations.

| Metric | Request-Level | Iteration-Level (Orca) |
|--------|---------------|------------------------|
| TTFT (ms) | 33,822 | 123 |
| E2E Latency (ms) | 33,823 | 1,914 |
| Throughput (req/sec) | 0.650 | 1.999 |
| KV-cache utilization | 6.2% | 22.7% |
| Avg batch size | 1.00 | 3.65 |

### Decode Stall: Orca vs Sarathi-Serve / Chunked Prefill (λ=1.5 reqs/sec)

| Chunk size | Throughput | Mean step | P99 step | P99/mean |
|------------|------------|-----------|----------|----------|
| none (Orca) | 1.499 | 25.7ms | 167.7ms | 6.5× |
| 16 | 1.499 | 25.5ms | 29.1ms | 1.1× |
| 64 | 1.499 | 25.6ms | 52.9ms | 2.1× |
| 256 | 1.499 | 25.7ms | 148.4ms | 5.8× |

## Testing

```bash
cargo test
cargo clippy -- -D warnings
```
