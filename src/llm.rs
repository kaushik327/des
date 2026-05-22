//! LLM inference scheduler simulator.
//!
//! Models the two-phase request lifecycle (prefill → per-token decode) with a
//! shared KV-cache capacity constraint.  Two scheduling policies are compared:
//!
//! - **Request-level**: one request occupies the decode batch at a time; the
//!   next is admitted only after the current finishes all output tokens.
//! - **Iteration-level (Orca)**: after every decode step, any waiting requests
//!   that fit in remaining KV-cache are admitted to the batch immediately.
//!
//! Hardware parameters are stylized rather than tied to a specific GPU; results
//! should be read as relative comparisons between policies.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

use rand::{SeedableRng, rngs::SmallRng};

use crate::calendar::EventCalendar;
use crate::clock::SimClock;
use crate::distributions::{Distribution, Exponential};
use crate::stats::Welford;

// ── Events ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum LlmEventKind {
    /// A new inference request arrives at the system.
    Arrival { req_id: u64 },
    /// Prefill phase finishes for `req_id`; KV-cache is now allocated.
    PrefillComplete { req_id: u64 },
    /// One decode step fires for the entire active batch simultaneously.
    DecodeStep,
}

#[derive(Debug, Clone)]
struct LlmEvent {
    timestamp: f64,
    kind: LlmEventKind,
}

impl LlmEvent {
    fn new(timestamp: f64, kind: LlmEventKind) -> Self {
        Self { timestamp, kind }
    }
}

impl PartialEq for LlmEvent {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
    }
}
impl Eq for LlmEvent {}
impl PartialOrd for LlmEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for LlmEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp
            .partial_cmp(&other.timestamp)
            .unwrap_or(Ordering::Equal)
    }
}

// ── Request ───────────────────────────────────────────────────────────────────

struct Request {
    arrival_time: f64,
    prompt_len: usize,
    output_len: usize,
    tokens_generated: usize,
    /// Slots currently held in the KV cache: `prompt_len` at admission, +1 per decode step.
    kv_slots: usize,
    first_token_time: Option<f64>,
}

// ── Public configuration ──────────────────────────────────────────────────────

/// Stylized GPU hardware parameters.
#[derive(Debug, Clone, Copy)]
pub struct HardwareConfig {
    /// Total KV-cache capacity in tokens across all concurrently active requests.
    pub kv_capacity: usize,
    /// Compute time to prefill a single prompt token.
    pub t_prefill_per_token: f64,
    /// Base decode step latency (memory-bandwidth bound, independent of batch size).
    pub t_decode_base: f64,
    /// Additional per-request overhead per decode step (attention over KV-cache).
    pub t_decode_per_req: f64,
    /// Hard cap on the number of requests in the decode batch simultaneously.
    pub max_batch_size: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            kv_capacity: 2048,
            t_prefill_per_token: 0.001,
            t_decode_base: 0.020,
            t_decode_per_req: 0.001,
            max_batch_size: 16,
        }
    }
}

/// Scheduling policy for the decode batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmPolicy {
    /// One request at a time; next admitted only when the current finishes all decode.
    RequestLevel,
    /// Orca-style: at every decode step admit all waiting requests that fit in KV-cache.
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

// ── Scheduler ────────────────────────────────────────────────────────────────

/// LLM inference scheduler simulator.
///
/// Arrival → prefill queue → prefill (single-threaded) → KV-wait queue →
/// decode batch → completion.  Prefill and decode run concurrently on separate
/// resources, matching real serving systems.
pub struct InferenceScheduler {
    pub clock: SimClock,
    hw: HardwareConfig,
    policy: LlmPolicy,

    calendar: EventCalendar<LlmEvent>,
    requests: HashMap<u64, Request>,
    next_req_id: u64,

    /// Requests waiting to be prefilled, in arrival order.
    prefill_queue: VecDeque<u64>,
    /// The request currently being prefilled, if any.
    being_prefilled: Option<u64>,

    /// Requests whose prefill is done but that cannot yet enter the decode batch.
    kv_wait: VecDeque<u64>,
    /// Requests currently in the decode batch.
    decode_batch: Vec<u64>,
    /// Whether a `DecodeStep` event is already queued in the calendar.
    decode_step_scheduled: bool,
    /// Total KV-cache slots currently in use.
    kv_used: usize,

    rng: SmallRng,
    arrival_dist: Exponential,
    prompt_dist: Box<dyn Distribution>,
    output_dist: Box<dyn Distribution>,

    pub completed: u64,
    ttft_stats: Welford,
    e2e_stats: Welford,
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
            decode_step_scheduled: false,
            kv_used: 0,
            rng: SmallRng::seed_from_u64(seed),
            arrival_dist: Exponential::new(lambda),
            prompt_dist,
            output_dist,
            completed: 0,
            ttft_stats: Welford::new(),
            e2e_stats: Welford::new(),
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

    /// Mean time-to-first-token across completed requests.
    pub fn mean_ttft(&self) -> Option<f64> {
        self.ttft_stats.mean()
    }

    /// Mean end-to-end latency (arrival → last output token) across completed requests.
    pub fn mean_e2e(&self) -> Option<f64> {
        self.e2e_stats.mean()
    }

    /// Completed requests per unit of simulated time.
    pub fn throughput(&self) -> f64 {
        if self.clock.time <= 0.0 {
            return 0.0;
        }
        self.completed as f64 / self.clock.time
    }

    /// Time-averaged fraction of KV-cache capacity occupied.
    pub fn mean_kv_util(&self) -> f64 {
        if self.clock.time <= 0.0 {
            return 0.0;
        }
        self.kv_area / (self.clock.time * self.hw.kv_capacity as f64)
    }

    /// Time-averaged number of requests concurrently decoding.
    pub fn mean_batch_size(&self) -> f64 {
        if self.clock.time <= 0.0 {
            return 0.0;
        }
        self.batch_area / self.clock.time
    }

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
        self.calendar.push(LlmEvent::new(
            self.clock.time + dt,
            LlmEventKind::Arrival { req_id },
        ));
    }

    fn try_start_prefill(&mut self) {
        if self.being_prefilled.is_some() {
            return;
        }
        if let Some(req_id) = self.prefill_queue.pop_front() {
            let prompt_len = self.requests[&req_id].prompt_len;
            let done_at = self.clock.time + prompt_len as f64 * self.hw.t_prefill_per_token;
            self.being_prefilled = Some(req_id);
            self.calendar.push(LlmEvent::new(
                done_at,
                LlmEventKind::PrefillComplete { req_id },
            ));
        }
    }

    /// Try to move `req_id` (prefill just completed) directly into the decode batch,
    /// or park it in `kv_wait` if constraints prevent immediate admission.
    fn try_admit(&mut self, req_id: u64) {
        let prompt_len = self.requests[&req_id].prompt_len;
        let kv_ok = self.kv_used + prompt_len <= self.hw.kv_capacity;
        let batch_ok = self.decode_batch.len() < self.hw.max_batch_size;
        let policy_ok = self.policy == LlmPolicy::IterationLevel || self.decode_batch.is_empty();

        if kv_ok && batch_ok && policy_ok {
            self.kv_used += prompt_len;
            self.requests.get_mut(&req_id).unwrap().kv_slots = prompt_len;
            self.decode_batch.push(req_id);
            self.ensure_decode_scheduled();
        } else {
            self.kv_wait.push_back(req_id);
        }
    }

    /// Drain `kv_wait` into the decode batch as long as KV-cache and batch-size
    /// constraints (and policy) allow.  FCFS order is preserved within the queue.
    fn admit_waiting(&mut self) {
        while let Some(&req_id) = self.kv_wait.front() {
            let prompt_len = self.requests[&req_id].prompt_len;
            let kv_ok = self.kv_used + prompt_len <= self.hw.kv_capacity;
            let batch_ok = self.decode_batch.len() < self.hw.max_batch_size;
            // RequestLevel: only admit when the batch has just drained to empty.
            let policy_ok =
                self.policy == LlmPolicy::IterationLevel || self.decode_batch.is_empty();

            if kv_ok && batch_ok && policy_ok {
                self.kv_wait.pop_front();
                self.kv_used += prompt_len;
                self.requests.get_mut(&req_id).unwrap().kv_slots = prompt_len;
                self.decode_batch.push(req_id);
            } else {
                break;
            }
        }
        self.ensure_decode_scheduled();
    }

    fn ensure_decode_scheduled(&mut self) {
        if !self.decode_batch.is_empty() && !self.decode_step_scheduled {
            let t = self.clock.time + self.decode_step_time();
            self.calendar
                .push(LlmEvent::new(t, LlmEventKind::DecodeStep));
            self.decode_step_scheduled = true;
        }
    }

    fn dispatch(&mut self, kind: LlmEventKind) {
        match kind {
            LlmEventKind::Arrival { req_id } => self.on_arrival(req_id),
            LlmEventKind::PrefillComplete { req_id } => self.on_prefill_complete(req_id),
            LlmEventKind::DecodeStep => self.on_decode_step(),
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
                kv_slots: 0,
                first_token_time: None,
            },
        );
        self.prefill_queue.push_back(req_id);
        self.try_start_prefill();
        self.schedule_next_arrival();
    }

    fn on_prefill_complete(&mut self, req_id: u64) {
        self.update_time_stats();
        self.being_prefilled = None;
        self.try_admit(req_id);
        self.try_start_prefill();
    }

    fn on_decode_step(&mut self) {
        self.update_time_stats();
        self.decode_step_scheduled = false;

        let now = self.clock.time;

        // Each request in the batch generates one token, consuming one more KV slot.
        self.kv_used += self.decode_batch.len();

        // Collect outcomes before mutably borrowing stats accumulators.
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
        }

        for req_id in &completions {
            let req = self.requests.remove(req_id).unwrap();
            self.kv_used -= req.kv_slots;
            self.e2e_stats.update(now - req.arrival_time);
            self.completed += 1;
        }
        self.decode_batch.retain(|id| !completions.contains(id));

        self.admit_waiting();
        self.ensure_decode_scheduled();
    }
}
