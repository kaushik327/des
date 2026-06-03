//! LLM inference scheduler simulator.
//!
//! Uses a single shared-GPU model: prefill and decode are processed sequentially
//! in GPU "iterations", matching real serving systems.  Two regimes:
//!
//! - **`chunk_size == 0` (Orca)**: each iteration is either a full-prompt prefill
//!   OR a decode step.  A long prompt monopolises the GPU for its entire prefill
//!   duration, stalling every in-flight decode request for that period.
//!
//! - **`chunk_size > 0` (Sarathi-Serve)**: each iteration processes one fixed-size
//!   chunk of the prompt then a decode step for the active batch.  The active
//!   request continues its chunks without yielding to new arrivals — identical to
//!   the reference system.  Decode stall per iteration is bounded by
//!   `chunk_size × t_prefill_per_token`.
//!
//! Two admission policies control when a prefilled request enters the decode batch:
//!
//! - **`RequestLevel`**: the next request is admitted only when the batch drains
//!   completely (one-at-a-time serialisation).
//! - **`IterationLevel`** (Orca): new requests are admitted at every iteration
//!   boundary as soon as KV-cache space is available.

use std::collections::{HashMap, VecDeque};

use rand::{SeedableRng, rngs::SmallRng};

use crate::calendar::{Event, EventCalendar};
use crate::clock::SimClock;
use crate::distributions::{Distribution, Exponential};
use crate::stats::{EmpiricalCdf, Welford};

#[derive(Debug, Clone)]
enum LlmEventKind {
    Arrival {
        req_id: u64,
    },
    /// One GPU iteration: one prefill chunk (if any), then one decode step (if not blocked).
    Iteration,
}

struct Request {
    arrival_time: f64,
    prompt_len: usize,
    output_len: usize,
    tokens_generated: usize,
    tokens_prefilled: usize,
    /// KV-cache slots held: `prompt_len` at admission, +1 per decode step.
    kv_slots: usize,
    first_token_time: Option<f64>,
}

/// Stylized GPU hardware parameters.
#[derive(Debug, Clone, Copy)]
pub struct HardwareConfig {
    /// Total KV-cache capacity in tokens across all concurrently active requests.
    pub kv_capacity: usize,
    /// GPU time to process one prompt token during prefill.
    pub t_prefill_per_token: f64,
    /// Base decode step latency (memory-bandwidth bound, independent of batch size).
    pub t_decode_base: f64,
    /// Additional per-request overhead per decode step (KV-cache attention cost).
    pub t_decode_per_req: f64,
    /// Hard cap on the number of requests in the decode batch simultaneously.
    pub max_batch_size: usize,
    /// Tokens prefilled per GPU iteration.
    ///
    /// `0` = Orca-style: the full prompt is prefilled atomically, blocking decode
    /// for the entire prefill duration (`prompt_len × t_prefill_per_token`).
    ///
    /// `> 0` = Sarathi-Serve: one chunk per iteration, then a decode step runs
    /// in the same iteration.  Decode stall is bounded by
    /// `chunk_size × t_prefill_per_token` per iteration.
    pub chunk_size: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            kv_capacity: 2048,
            t_prefill_per_token: 0.001,
            t_decode_base: 0.020,
            t_decode_per_req: 0.001,
            max_batch_size: 16,
            chunk_size: 0,
        }
    }
}

/// Admission policy: when can a prefilled request enter the decode batch?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmPolicy {
    /// Admit only when the decode batch is completely empty (one request at a time).
    RequestLevel,
    /// Admit at every iteration boundary as long as KV-cache and batch limits allow.
    IterationLevel,
}

impl std::fmt::Display for LlmPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmPolicy::RequestLevel => write!(f, "request-level"),
            LlmPolicy::IterationLevel => write!(f, "iter-level "),
        }
    }
}

/// LLM inference scheduler simulator.
///
/// The GPU is modelled as a sequential resource.  A single `Iteration` event
/// drives both prefill and decode: prefill chunk first (if any), then a decode
/// step (unless an unchunked prefill blocked the GPU).
pub struct InferenceScheduler {
    pub clock: SimClock,
    hw: HardwareConfig,
    policy: LlmPolicy,

    calendar: EventCalendar<Event<LlmEventKind>>,
    requests: HashMap<u64, Request>,
    next_req_id: u64,

    prefill_queue: VecDeque<u64>,
    /// The request currently being chunked across iterations; `None` when no prefill is active.
    being_prefilled: Option<u64>,
    kv_wait: VecDeque<u64>,
    decode_batch: Vec<u64>,
    /// Prevents double-scheduling of `Iteration` events.
    iteration_scheduled: bool,
    kv_used: usize,

    rng: SmallRng,
    arrival_dist: Exponential,
    prompt_dist: Box<dyn Distribution>,
    output_dist: Box<dyn Distribution>,

    pub completed: u64,
    ttft_stats: Welford,
    ttft_cdf: EmpiricalCdf,
    e2e_stats: Welford,
    /// Wall-clock interval between consecutive decode steps.
    /// Spikes during unchunked prefills (Orca); bounded by chunk_size×t_prefill (Sarathi-Serve).
    step_lat_stats: Welford,
    step_lat_cdf: EmpiricalCdf,
    last_decode_time: f64,
    kv_area: f64,
    batch_area: f64,
    last_event_time: f64,
}

impl InferenceScheduler {
    pub fn new(
        policy: LlmPolicy,
        hw: HardwareConfig,
        lambda: f64,
        prompt_dist: Box<dyn Distribution>,
        output_dist: Box<dyn Distribution>,
        seed: u64,
    ) -> Self {
        let mut sched = Self {
            clock: SimClock::new(),
            hw,
            policy,
            calendar: EventCalendar::new(),
            requests: HashMap::new(),
            next_req_id: 0,
            prefill_queue: VecDeque::new(),
            being_prefilled: None,
            kv_wait: VecDeque::new(),
            decode_batch: Vec::new(),
            iteration_scheduled: false,
            kv_used: 0,
            rng: SmallRng::seed_from_u64(seed),
            arrival_dist: Exponential::new(lambda),
            prompt_dist,
            output_dist,
            completed: 0,
            ttft_stats: Welford::new(),
            ttft_cdf: EmpiricalCdf::new(),
            e2e_stats: Welford::new(),
            step_lat_stats: Welford::new(),
            step_lat_cdf: EmpiricalCdf::new(),
            last_decode_time: 0.0,
            kv_area: 0.0,
            batch_area: 0.0,
            last_event_time: 0.0,
        };
        sched.schedule_next_arrival();
        sched
    }

    pub fn run_until(&mut self, end_time: f64) {
        while let Some(ev) = self.calendar.pop_next() {
            if ev.timestamp > end_time {
                self.calendar.push(ev);
                break;
            }
            self.clock.advance_to(ev.timestamp);
            self.dispatch(ev.kind);
        }
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    pub fn mean_ttft(&self) -> Option<f64> {
        self.ttft_stats.mean()
    }

    pub fn mean_e2e(&self) -> Option<f64> {
        self.e2e_stats.mean()
    }

    pub fn p99_ttft(&mut self) -> Option<f64> {
        self.ttft_stats.mean()?;
        Some(self.ttft_cdf.percentile(0.99))
    }

    pub fn throughput(&self) -> f64 {
        if self.clock.time <= 0.0 {
            return 0.0;
        }
        self.completed as f64 / self.clock.time
    }

    pub fn mean_kv_util(&self) -> f64 {
        if self.clock.time <= 0.0 {
            return 0.0;
        }
        self.kv_area / (self.clock.time * self.hw.kv_capacity as f64)
    }

    pub fn mean_batch_size(&self) -> f64 {
        if self.clock.time <= 0.0 {
            return 0.0;
        }
        self.batch_area / self.clock.time
    }

    pub fn mean_step_latency(&self) -> Option<f64> {
        self.step_lat_stats.mean()
    }

    /// P99 wall-clock gap between consecutive decode steps.
    /// With `chunk_size == 0` this equals the P99 prefill duration (worst-case freeze).
    pub fn p99_step_latency(&mut self) -> Option<f64> {
        self.step_lat_stats.mean()?;
        Some(self.step_lat_cdf.percentile(0.99))
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn update_time_stats(&mut self) {
        let dt = self.clock.time - self.last_event_time;
        if dt > 0.0 {
            self.kv_area += self.kv_used as f64 * dt;
            self.batch_area += self.decode_batch.len() as f64 * dt;
        }
        self.last_event_time = self.clock.time;
    }

    fn decode_step_time(&self) -> f64 {
        self.hw.t_decode_base + self.decode_batch.len() as f64 * self.hw.t_decode_per_req
    }

    fn sample_len(dist: &dyn Distribution, rng: &mut SmallRng) -> usize {
        (dist.sample(rng).round() as usize).max(1)
    }

    fn schedule_next_arrival(&mut self) {
        let dt = self.arrival_dist.sample(&mut self.rng);
        let req_id = self.next_req_id;
        self.next_req_id += 1;
        self.calendar.push(Event::new(
            self.clock.time + dt,
            LlmEventKind::Arrival { req_id },
        ));
    }

    fn try_start_prefill(&mut self) {
        if self.being_prefilled.is_some() {
            return;
        }
        while let Some(req_id) = self.prefill_queue.pop_front() {
            if self.requests.contains_key(&req_id) {
                self.being_prefilled = Some(req_id);
                break;
            }
        }
    }

    fn next_iteration_duration(&self) -> f64 {
        let prefill_time = if let Some(req_id) = self.being_prefilled {
            let req = &self.requests[&req_id];
            let remaining = req.prompt_len - req.tokens_prefilled;
            let chunk = if self.hw.chunk_size > 0 {
                self.hw.chunk_size.min(remaining)
            } else {
                remaining
            };
            chunk as f64 * self.hw.t_prefill_per_token
        } else {
            0.0
        };
        let decode_blocked = prefill_time > 0.0 && self.hw.chunk_size == 0;
        let decode_time = if !self.decode_batch.is_empty() && !decode_blocked {
            self.decode_step_time()
        } else {
            0.0
        };
        prefill_time + decode_time
    }

    fn ensure_iteration_scheduled(&mut self) {
        if self.iteration_scheduled {
            return;
        }
        let dur = self.next_iteration_duration();
        if dur > 0.0 {
            self.calendar
                .push(Event::new(self.clock.time + dur, LlmEventKind::Iteration));
            self.iteration_scheduled = true;
        }
    }

    fn can_admit(&self, prompt_len: usize) -> bool {
        self.kv_used + prompt_len <= self.hw.kv_capacity
            && self.decode_batch.len() < self.hw.max_batch_size
            && (self.policy == LlmPolicy::IterationLevel || self.decode_batch.is_empty())
    }

    fn try_admit(&mut self, req_id: u64) {
        let prompt_len = self.requests[&req_id].prompt_len;
        if self.can_admit(prompt_len) {
            self.kv_used += prompt_len;
            self.requests.get_mut(&req_id).unwrap().kv_slots = prompt_len;
            self.decode_batch.push(req_id);
        } else {
            self.kv_wait.push_back(req_id);
        }
    }

    fn admit_waiting(&mut self) {
        while let Some(&req_id) = self.kv_wait.front() {
            let prompt_len = self.requests[&req_id].prompt_len;
            if self.can_admit(prompt_len) {
                self.kv_wait.pop_front();
                self.kv_used += prompt_len;
                self.requests.get_mut(&req_id).unwrap().kv_slots = prompt_len;
                self.decode_batch.push(req_id);
            } else {
                break;
            }
        }
    }

    fn do_decode_step(&mut self, now: f64) {
        if self.last_decode_time > 0.0 {
            let lat = now - self.last_decode_time;
            self.step_lat_stats.update(lat);
            self.step_lat_cdf.push(lat);
        }
        self.last_decode_time = now;

        self.kv_used += self.decode_batch.len();

        let mut ttft_updates: Vec<f64> = Vec::new();
        let mut completions: Vec<u64> = Vec::new();

        for &req_id in &self.decode_batch {
            let req = self.requests.get_mut(&req_id).unwrap();
            req.tokens_generated += 1;
            req.kv_slots += 1;
            if req.first_token_time.is_none() {
                req.first_token_time = Some(now);
                ttft_updates.push(now - req.arrival_time);
            }
            if req.tokens_generated >= req.output_len {
                completions.push(req_id);
            }
        }

        for ttft in ttft_updates {
            self.ttft_stats.update(ttft);
            self.ttft_cdf.push(ttft);
        }
        for req_id in &completions {
            let req = self.requests.remove(req_id).unwrap();
            self.kv_used -= req.kv_slots;
            self.e2e_stats.update(now - req.arrival_time);
            self.completed += 1;
        }
        self.decode_batch.retain(|id| !completions.contains(id));

        self.admit_waiting();
    }

    // ── Event handlers ────────────────────────────────────────────────────────

    fn dispatch(&mut self, kind: LlmEventKind) {
        match kind {
            LlmEventKind::Arrival { req_id } => self.on_arrival(req_id),
            LlmEventKind::Iteration => self.on_iteration(),
        }
    }

    fn on_arrival(&mut self, req_id: u64) {
        self.update_time_stats();
        let prompt_len =
            Self::sample_len(self.prompt_dist.as_ref(), &mut self.rng).min(self.hw.kv_capacity);
        let output_len = Self::sample_len(self.output_dist.as_ref(), &mut self.rng);
        self.requests.insert(
            req_id,
            Request {
                arrival_time: self.clock.time,
                prompt_len,
                output_len,
                tokens_generated: 0,
                tokens_prefilled: 0,
                kv_slots: 0,
                first_token_time: None,
            },
        );
        self.prefill_queue.push_back(req_id);
        // Only claim the GPU when idle: the scheduled iteration's duration is already
        // committed; starting a prefill now would make on_iteration process more work
        // than that duration accounts for, skewing the clock.
        if !self.iteration_scheduled {
            self.try_start_prefill();
        }
        self.ensure_iteration_scheduled();
        self.schedule_next_arrival();
    }

    fn on_iteration(&mut self) {
        self.update_time_stats();
        self.iteration_scheduled = false;
        let now = self.clock.time;

        let prefill_was_running = self.being_prefilled.is_some();

        if let Some(req_id) = self.being_prefilled {
            let prefill_done = {
                let req = self.requests.get_mut(&req_id).unwrap();
                let remaining = req.prompt_len - req.tokens_prefilled;
                let chunk = if self.hw.chunk_size > 0 {
                    self.hw.chunk_size.min(remaining)
                } else {
                    remaining
                };
                req.tokens_prefilled += chunk;
                req.tokens_prefilled >= req.prompt_len
            };

            if prefill_done {
                self.being_prefilled = None;
                self.try_admit(req_id);
            }
        }

        let decode_blocked = prefill_was_running && self.hw.chunk_size == 0;
        if !self.decode_batch.is_empty() && !decode_blocked {
            self.do_decode_step(now);
        }

        if self.being_prefilled.is_none() {
            self.try_start_prefill();
        }

        self.ensure_iteration_scheduled();
    }
}
