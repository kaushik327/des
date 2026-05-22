//! Open Jackson queueing network.
//!
//! Each node is an independent M/M/1 queue. External Poisson arrivals feed the
//! network; at departure a job routes to another node with probabilities given by
//! a routing matrix, or exits the network.
//!
//! Product-form result: in steady state each node behaves as an independent M/M/1
//! with effective arrival rate λ_i solved from the traffic equations
//!   λ_i = γ_i + Σ_j λ_j · R[j][i]
//! so E[N_i] = ρ_i/(1−ρ_i) where ρ_i = λ_i/μ_i.

use std::collections::{HashMap, VecDeque};

use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::clock::SimClock;
use crate::distributions::{Distribution, Exponential};

#[derive(Debug, Clone)]
enum NetEvent {
    Arrival { node: usize, job_id: u64 },
    Departure { node: usize, job_id: u64 },
}

#[derive(Debug, Clone)]
struct TimedEvent {
    time: f64,
    kind: NetEvent,
}

#[derive(Debug)]
struct Node {
    service_dist: Box<dyn Distribution>,
    queue: VecDeque<u64>,
    busy: bool,
    system_size: u64,
    area_n: f64,
    last_event_time: f64,
}

impl Node {
    fn new(service_dist: Box<dyn Distribution>) -> Self {
        Self {
            service_dist,
            queue: VecDeque::new(),
            busy: false,
            system_size: 0,
            area_n: 0.0,
            last_event_time: 0.0,
        }
    }

    fn mean_system_size(&self, now: f64) -> f64 {
        if now <= 0.0 {
            return 0.0;
        }
        (self.area_n + self.system_size as f64 * (now - self.last_event_time)) / now
    }
}

/// Open Jackson network: a directed graph of M/M/1 nodes with probabilistic routing.
pub struct JacksonNetwork {
    pub clock: SimClock,
    nodes: Vec<Node>,
    /// routing[i][j] = probability job routes from node i to node j; row sum ≤ 1.
    routing: Vec<Vec<f64>>,
    external_rates: Vec<f64>,
    arrival_dists: Vec<Option<Exponential>>,
    calendar: Vec<TimedEvent>,
    rng: SmallRng,
    next_job_id: u64,
    arrival_times: HashMap<u64, f64>,
    pub jobs_completed: u64,
    pub total_sojourn: f64,
}

impl std::fmt::Debug for JacksonNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JacksonNetwork")
            .field("clock", &self.clock)
            .field("jobs_completed", &self.jobs_completed)
            .finish()
    }
}

impl JacksonNetwork {
    /// Build a network with `n` nodes.
    ///
    /// - `service_dists`: one distribution per node.
    /// - `routing`: n×n matrix; `routing[i][j]` = p(i→j), row sum ≤ 1.
    /// - `external_rates`: Poisson rate of external arrivals at each node.
    pub fn new(
        service_dists: Vec<Box<dyn Distribution>>,
        routing: Vec<Vec<f64>>,
        external_rates: Vec<f64>,
        seed: u64,
    ) -> Self {
        let n = service_dists.len();
        assert_eq!(routing.len(), n);
        assert_eq!(external_rates.len(), n);
        let nodes: Vec<Node> = service_dists.into_iter().map(Node::new).collect();
        let arrival_dists: Vec<Option<Exponential>> = external_rates
            .iter()
            .map(|&r| {
                if r > 0.0 {
                    Some(Exponential::new(r))
                } else {
                    None
                }
            })
            .collect();
        let mut net = Self {
            clock: SimClock::new(),
            nodes,
            routing,
            external_rates,
            arrival_dists,
            calendar: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            next_job_id: 0,
            arrival_times: HashMap::new(),
            jobs_completed: 0,
            total_sojourn: 0.0,
        };
        for i in 0..n {
            net.schedule_next_external(i);
        }
        net
    }

    pub fn run_until(&mut self, end_time: f64) {
        loop {
            let next_idx = self
                .calendar
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.time.total_cmp(&b.1.time))
                .map(|(i, _)| i);
            let Some(idx) = next_idx else { break };
            if self.calendar[idx].time > end_time {
                break;
            }
            let ev = self.calendar.swap_remove(idx);
            self.clock.advance_to(ev.time);
            self.dispatch(ev.kind);
        }
    }

    /// Time-averaged mean number of jobs at node `i`.
    pub fn mean_system_size(&self, node: usize) -> f64 {
        self.nodes[node].mean_system_size(self.clock.time)
    }

    /// Mean end-to-end sojourn time across all jobs that exited the network.
    pub fn mean_sojourn(&self) -> Option<f64> {
        (self.jobs_completed > 0).then(|| self.total_sojourn / self.jobs_completed as f64)
    }

    /// Solve the traffic equations λ_i = γ_i + Σ_j λ_j·R[j][i] by Gauss-Seidel.
    pub fn effective_arrival_rates(&self) -> Vec<f64> {
        let n = self.nodes.len();
        let mut lam = self.external_rates.clone();
        for _ in 0..1000 {
            let prev = lam.clone();
            for (i, lam_i) in lam.iter_mut().enumerate().take(n) {
                *lam_i = self.external_rates[i]
                    + (0..n).map(|j| prev[j] * self.routing[j][i]).sum::<f64>();
            }
            let max_err = lam
                .iter()
                .zip(&prev)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_err < 1e-10 {
                break;
            }
        }
        lam
    }

    fn push_event(&mut self, time: f64, kind: NetEvent) {
        self.calendar.push(TimedEvent { time, kind });
    }

    fn schedule_next_external(&mut self, node: usize) {
        if let Some(dist) = self.arrival_dists[node] {
            let dt = dist.sample(&mut self.rng);
            let job_id = self.next_job_id;
            self.next_job_id += 1;
            self.push_event(self.clock.time + dt, NetEvent::Arrival { node, job_id });
        }
    }

    fn schedule_departure(&mut self, node: usize, job_id: u64) {
        let dt = self.nodes[node].service_dist.sample(&mut self.rng);
        self.push_event(self.clock.time + dt, NetEvent::Departure { node, job_id });
    }

    fn update_node_stats(&mut self, node: usize) {
        let n = &mut self.nodes[node];
        let dt = self.clock.time - n.last_event_time;
        if dt > 0.0 {
            n.area_n += n.system_size as f64 * dt;
        }
        n.last_event_time = self.clock.time;
    }

    fn dispatch(&mut self, kind: NetEvent) {
        match kind {
            NetEvent::Arrival { node, job_id } => self.on_arrival(node, job_id),
            NetEvent::Departure { node, job_id } => self.on_departure(node, job_id),
        }
    }

    fn on_arrival(&mut self, node: usize, job_id: u64) {
        self.update_node_stats(node);
        self.nodes[node].system_size += 1;
        self.arrival_times.entry(job_id).or_insert(self.clock.time);

        if !self.nodes[node].busy {
            self.nodes[node].busy = true;
            self.schedule_departure(node, job_id);
        } else {
            self.nodes[node].queue.push_back(job_id);
        }
        self.schedule_next_external(node);
    }

    fn on_departure(&mut self, node: usize, job_id: u64) {
        self.update_node_stats(node);
        self.nodes[node].system_size -= 1;
        self.nodes[node].busy = false;

        let u: f64 = self.rng.random();
        let mut cum = 0.0;
        let mut routed = false;
        for (j, &p) in self.routing[node].iter().enumerate() {
            cum += p;
            if u < cum {
                let next_node = j;
                self.update_node_stats(next_node);
                self.nodes[next_node].system_size += 1;
                if !self.nodes[next_node].busy {
                    self.nodes[next_node].busy = true;
                    self.schedule_departure(next_node, job_id);
                } else {
                    self.nodes[next_node].queue.push_back(job_id);
                }
                routed = true;
                break;
            }
        }

        if !routed && let Some(t_enter) = self.arrival_times.remove(&job_id) {
            self.total_sojourn += self.clock.time - t_enter;
            self.jobs_completed += 1;
        }

        if let Some(next_job) = self.nodes[node].queue.pop_front() {
            self.nodes[node].busy = true;
            self.schedule_departure(node, next_job);
        }
    }
}
