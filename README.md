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
cargo run -- mm1          # M/M/1 queue validation
cargo run -- pk           # P-K formula: effect of service variance
cargo run -- srpt         # SRPT vs FCFS on Pareto workload
cargo run -- mmk          # M/M/k multi-server queue
cargo run -- mmk-finite   # M/M/1/K finite buffer
cargo run -- jackson      # Jackson open network (product-form)
cargo run -- llm          # LLM inference scheduler
```

## Testing

```bash
cargo test
cargo clippy -- -D warnings
```
