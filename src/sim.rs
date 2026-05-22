use std::collections::HashMap;

use rand::{SeedableRng, rngs::SmallRng};

use crate::calendar::EventCalendar;
use crate::clock::SimClock;
use crate::distributions::{Distribution, Exponential};
use crate::event::{Event, EventKind};
use crate::queue::SimQueue;
use crate::server::Server;
use crate::stats::{EmpiricalCdf, Welford};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Policy {
    #[default]
    Fcfs,
    Srpt,
}

#[derive(Debug)]
pub struct Simulation {
    pub clock: SimClock,
    pub arrivals_processed: u64,
    pub drops: u64,
    calendar: EventCalendar,
    rng: SmallRng,
    arrival_dist: Option<Exponential>,
    service_dist: Option<Box<dyn Distribution>>,
    policy: Policy,
    server: Server,
    queue: SimQueue,
    capacity: Option<usize>, // finite buffer (None = ∞)
    job_remaining: HashMap<u64, f64>,
    arrival_times: HashMap<u64, f64>,
    response_time: Welford,
    response_time_cdf: EmpiricalCdf,
    system_size: u64,
    area_n: f64,
    busy_area: f64,
    last_event_time: f64,
    warmup_until: f64,
    next_job_id: u64,
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}

impl Simulation {
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            clock: SimClock::new(),
            arrivals_processed: 0,
            drops: 0,
            calendar: EventCalendar::new(),
            rng: SmallRng::seed_from_u64(seed),
            arrival_dist: None,
            service_dist: None,
            policy: Policy::Fcfs,
            server: Server::new(0),
            queue: SimQueue::fcfs(),
            capacity: None,
            job_remaining: HashMap::new(),
            arrival_times: HashMap::new(),
            response_time: Welford::new(),
            response_time_cdf: EmpiricalCdf::new(),
            system_size: 0,
            area_n: 0.0,
            busy_area: 0.0,
            last_event_time: 0.0,
            warmup_until: 0.0,
            next_job_id: 0,
        }
    }

    pub fn start_arrivals(&mut self, lambda: f64) {
        self.arrival_dist = Some(Exponential::new(lambda));
        self.schedule_next_arrival();
    }

    pub fn start_service(&mut self, dist: impl Distribution + 'static) {
        self.service_dist = Some(Box::new(dist));
    }

    pub fn set_policy(&mut self, policy: Policy) {
        self.policy = policy;
        self.queue = match policy {
            Policy::Fcfs => SimQueue::fcfs(),
            Policy::Srpt => SimQueue::srpt(),
        };
    }

    #[allow(dead_code)]
    pub fn set_capacity(&mut self, cap: usize) {
        self.capacity = Some(cap);
    }

    #[allow(dead_code)]
    pub fn set_warmup(&mut self, until: f64) {
        self.warmup_until = until;
    }

    pub fn mean_response_time(&self) -> Option<f64> {
        self.response_time.mean()
    }
    pub fn response_time_std_dev(&self) -> Option<f64> {
        self.response_time.std_dev()
    }

    pub fn mean_system_size(&self) -> f64 {
        let post = self.clock.time - self.warmup_until;
        if post <= 0.0 { 0.0 } else { self.area_n / post }
    }

    pub fn server_utilization(&self) -> f64 {
        let post = self.clock.time - self.warmup_until;
        if post <= 0.0 {
            0.0
        } else {
            self.busy_area / post
        }
    }

    pub fn tail_prob(&mut self, threshold: f64) -> f64 {
        self.response_time_cdf.finish();
        self.response_time_cdf.tail_prob(threshold)
    }

    #[allow(dead_code)]
    pub fn schedule(&mut self, event: Event) {
        self.calendar.push(event);
    }

    pub fn run_until(&mut self, end_time: f64) {
        while let Some(event) = self.calendar.pop_next() {
            if event.timestamp > end_time {
                self.calendar.push(event);
                break;
            }
            self.clock.advance_to(event.timestamp);
            self.dispatch(event);
        }
    }

    // ── internals ──────────────────────────────────────────────────────────

    fn update_time_stats(&mut self) {
        let t_from = self.last_event_time.max(self.warmup_until);
        let t_now = self.clock.time;
        if t_now > t_from {
            let dt = t_now - t_from;
            self.area_n += self.system_size as f64 * dt;
            if !self.server.is_idle() {
                self.busy_area += dt;
            }
        }
        self.last_event_time = t_now;
    }

    fn sample_service_time(&mut self) -> Option<f64> {
        self.service_dist.as_ref().map(|d| d.sample(&mut self.rng))
    }

    fn begin_service(&mut self, job_id: u64) {
        let remaining = self.job_remaining.get(&job_id).copied().unwrap_or(0.0);
        if remaining > 0.0 {
            let departure_at = self.clock.time + remaining;
            let epoch = self.server.begin(job_id, departure_at);
            self.calendar.push(Event::new(
                departure_at,
                EventKind::Departure {
                    job_id,
                    server_id: self.server.id(),
                    epoch,
                },
            ));
        } else {
            self.server.start_bare(job_id);
        }
    }

    fn schedule_next_arrival(&mut self) {
        if let Some(dist) = self.arrival_dist {
            let dt = dist.sample(&mut self.rng);
            let job_id = self.next_job_id;
            self.next_job_id += 1;
            self.calendar.push(Event::new(
                self.clock.time + dt,
                EventKind::Arrival { job_id },
            ));
        }
    }

    fn dispatch(&mut self, event: Event) {
        match event.kind {
            EventKind::Arrival { job_id } => self.on_arrival(job_id),
            EventKind::Departure { job_id, epoch, .. } => {
                if self.server.is_current_epoch(epoch) {
                    self.on_departure(job_id);
                }
                // else: stale event from a preempted job — discard silently
            }
        }
    }

    fn on_arrival(&mut self, job_id: u64) {
        self.update_time_stats();

        // Finite-buffer drop check (before incrementing system_size)
        if let Some(cap) = self.capacity
            && self.queue.len() >= cap
        {
            self.drops += 1;
            self.schedule_next_arrival();
            return;
        }

        self.system_size += 1;
        self.arrivals_processed += 1;
        self.arrival_times.insert(job_id, self.clock.time);

        // Sample service time. For SRPT this must happen at arrival so we know
        // the job size for preemption decisions. For FCFS it happens here too
        // (sampling at arrival vs. service-start is equivalent in distribution).
        if let Some(size) = self.sample_service_time() {
            self.job_remaining.insert(job_id, size);
        }

        if self.server.is_idle() {
            self.begin_service(job_id);
        } else if self.policy == Policy::Srpt {
            let new_remaining = self
                .job_remaining
                .get(&job_id)
                .copied()
                .unwrap_or(f64::INFINITY);
            let current_remaining = self.server.departure_time - self.clock.time;
            if new_remaining < current_remaining {
                // Preempt: put current job back with updated remaining time
                let current_job = self.server.current_job().unwrap();
                self.server.finish();
                self.job_remaining.insert(current_job, current_remaining);
                self.queue.push(current_job, current_remaining);
                self.begin_service(job_id);
            } else {
                self.queue.push(job_id, new_remaining);
            }
        } else {
            let remaining = self.job_remaining.get(&job_id).copied().unwrap_or(0.0);
            self.queue.push(job_id, remaining);
        }

        self.schedule_next_arrival();
    }

    fn on_departure(&mut self, job_id: u64) {
        self.update_time_stats();
        self.system_size -= 1;
        self.server.finish();
        self.job_remaining.remove(&job_id);

        if let Some(t_arrive) = self.arrival_times.remove(&job_id)
            && t_arrive >= self.warmup_until
        {
            let sojourn = self.clock.time - t_arrive;
            self.response_time.update(sojourn);
            self.response_time_cdf.push(sojourn);
        }

        if let Some((next_job, _remaining)) = self.queue.pop() {
            self.begin_service(next_job);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Exponential;
    use crate::event::EventKind;

    fn arrival(t: f64, id: u64) -> Event {
        Event::new(t, EventKind::Arrival { job_id: id })
    }

    fn departure(t: f64, job: u64, server: usize) -> Event {
        Event::new(
            t,
            EventKind::Departure {
                job_id: job,
                server_id: server,
                epoch: 0,
            },
        )
    }

    #[test]
    fn processes_events_in_order() {
        let mut sim = Simulation::new();
        sim.schedule(arrival(3.0, 3));
        sim.schedule(arrival(1.0, 1));
        sim.schedule(departure(2.5, 1, 0));
        sim.schedule(arrival(1.5, 2));
        sim.run_until(f64::INFINITY);
        assert_eq!(sim.clock.time, 3.0);
    }

    #[test]
    fn stops_before_end_time() {
        let mut sim = Simulation::new();
        sim.schedule(arrival(1.0, 1));
        sim.schedule(arrival(5.0, 2));
        sim.run_until(3.0);
        assert_eq!(sim.clock.time, 1.0);
    }

    #[test]
    fn empty_simulation_runs_cleanly() {
        let mut sim = Simulation::new();
        sim.run_until(100.0);
        assert_eq!(sim.clock.time, 0.0);
    }

    #[test]
    fn poisson_arrival_rate_within_1_percent() {
        let lambda = 3.0;
        let end_time = 50_000.0;
        let mut sim = Simulation::with_seed(42);
        sim.start_arrivals(lambda);
        sim.run_until(end_time);
        let empirical_rate = sim.arrivals_processed as f64 / end_time;
        let err = (empirical_rate - lambda).abs() / lambda;
        assert!(
            err < 0.01,
            "rate error {err:.4} > 0.01 (got {empirical_rate:.4}, expected {lambda})"
        );
    }

    #[test]
    fn server_utilization_matches_rho() {
        let lambda = 0.7_f64;
        let mu = 1.0_f64;
        let rho = lambda / mu;
        let mut sim = Simulation::with_seed(7);
        sim.start_arrivals(lambda);
        sim.start_service(Exponential::new(mu));
        sim.run_until(500_000.0);
        let util = sim.server_utilization();
        let err = (util - rho).abs() / rho;
        assert!(
            err < 0.01,
            "utilization error {err:.4} (got {util:.4}, expected {rho:.4})"
        );
    }

    #[test]
    fn srpt_beats_fcfs_on_pareto() {
        use crate::distributions::Pareto;
        let lambda = 0.8_f64;
        let mu = 1.0_f64;
        let end_time = 500_000.0;

        let mut fcfs = Simulation::with_seed(42);
        fcfs.start_arrivals(lambda);
        fcfs.start_service(Pareto::with_mean(2.5, 1.0 / mu));
        fcfs.run_until(end_time);

        let mut srpt = Simulation::with_seed(42);
        srpt.start_arrivals(lambda);
        srpt.start_service(Pareto::with_mean(2.5, 1.0 / mu));
        srpt.set_policy(Policy::Srpt);
        srpt.run_until(end_time);

        let et_fcfs = fcfs.mean_response_time().unwrap();
        let et_srpt = srpt.mean_response_time().unwrap();
        assert!(
            et_srpt < et_fcfs,
            "SRPT ({et_srpt:.2}) should beat FCFS ({et_fcfs:.2}) on Pareto workload"
        );
    }
}
