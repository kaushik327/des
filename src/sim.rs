use std::collections::HashMap;

use rand::{SeedableRng, rngs::SmallRng};

use crate::calendar::EventCalendar;
use crate::clock::SimClock;
use crate::distributions::{Distribution, Exponential};
use crate::event::{Event, EventKind};
use crate::queue::FcfsQueue;
use crate::server::Server;
use crate::stats::{EmpiricalCdf, Welford};

#[derive(Debug)]
pub struct Simulation {
    pub clock: SimClock,
    pub arrivals_processed: u64,
    calendar: EventCalendar,
    rng: SmallRng,
    arrival_dist: Option<Exponential>,
    service_dist: Option<Exponential>,
    server: Server,
    queue: FcfsQueue,
    // per-job arrival timestamps for response-time accounting
    arrival_times: HashMap<u64, f64>,
    // post-warmup sample stats for response time
    response_time: Welford,
    response_time_cdf: EmpiricalCdf,
    // time-average of system size N (queue + server), post-warmup
    system_size: u64,
    area_n: f64,
    // time-average of server busy fraction, post-warmup
    busy_area: f64,
    last_event_time: f64,
    // discard observations before this simulated time (0 = no warmup)
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
            calendar: EventCalendar::new(),
            rng: SmallRng::seed_from_u64(seed),
            arrival_dist: None,
            service_dist: None,
            server: Server::new(0),
            queue: FcfsQueue::new(),
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

    /// Begin a Poisson arrival process at rate `lambda`.
    pub fn start_arrivals(&mut self, lambda: f64) {
        self.arrival_dist = Some(Exponential::new(lambda));
        self.schedule_next_arrival();
    }

    /// Set exponential service rate `mu` for the single server.
    pub fn start_service(&mut self, mu: f64) {
        self.service_dist = Some(Exponential::new(mu));
    }

    /// Discard all observations before `until` simulated time.
    /// A good default is 5–10× the expected mixing time (≈ 1/(√μ−√λ)² for M/M/1).
    #[allow(dead_code)]
    pub fn set_warmup(&mut self, until: f64) {
        self.warmup_until = until;
    }

    /// Mean response time (sojourn time) across all post-warmup completed jobs.
    pub fn mean_response_time(&self) -> Option<f64> {
        self.response_time.mean()
    }

    /// Standard deviation of response time across all post-warmup completed jobs.
    pub fn response_time_std_dev(&self) -> Option<f64> {
        self.response_time.std_dev()
    }

    /// Time-averaged mean number of jobs in the system (queue + server).
    pub fn mean_system_size(&self) -> f64 {
        let post_warmup_time = self.clock.time - self.warmup_until;
        if post_warmup_time <= 0.0 {
            return 0.0;
        }
        self.area_n / post_warmup_time
    }

    /// Time-averaged server utilization (fraction of time server is busy).
    pub fn server_utilization(&self) -> f64 {
        let post_warmup_time = self.clock.time - self.warmup_until;
        if post_warmup_time <= 0.0 {
            return 0.0;
        }
        self.busy_area / post_warmup_time
    }

    /// P{T > threshold} from the empirical response-time distribution.
    pub fn tail_prob(&mut self, threshold: f64) -> f64 {
        self.response_time_cdf.finish();
        self.response_time_cdf.tail_prob(threshold)
    }

    #[allow(dead_code)] // used by tests and for manually seeding non-Poisson events
    pub fn schedule(&mut self, event: Event) {
        self.calendar.push(event);
    }

    /// Run until no events remain or the next event exceeds `end_time`.
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

    // ── internal helpers ────────────────────────────────────────────────────

    fn update_time_stats(&mut self) {
        // Only integrate the portion of the interval that falls after warmup.
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

    fn schedule_departure(&mut self, job_id: u64) {
        if let Some(dist) = self.service_dist {
            let dt = dist.sample(&mut self.rng);
            self.calendar.push(Event::new(
                self.clock.time + dt,
                EventKind::Departure {
                    job_id,
                    server_id: self.server.id(),
                },
            ));
        }
    }

    fn dispatch(&mut self, event: Event) {
        match event.kind {
            EventKind::Arrival { job_id } => self.on_arrival(job_id),
            EventKind::Departure {
                job_id,
                server_id: _,
            } => self.on_departure(job_id),
        }
    }

    fn on_arrival(&mut self, job_id: u64) {
        self.update_time_stats();
        self.system_size += 1;
        self.arrivals_processed += 1;
        self.arrival_times.insert(job_id, self.clock.time);

        if self.server.is_idle() {
            self.server.start(job_id);
            self.schedule_departure(job_id);
        } else {
            self.queue.push(job_id);
        }

        self.schedule_next_arrival();
    }

    fn on_departure(&mut self, job_id: u64) {
        self.update_time_stats();
        self.system_size -= 1;

        self.server.finish();

        // Count only jobs that arrived after warmup, so their full sojourn
        // occurred in steady state.
        if let Some(t_arrive) = self.arrival_times.remove(&job_id)
            && t_arrive >= self.warmup_until
        {
            let sojourn = self.clock.time - t_arrive;
            self.response_time.update(sojourn);
            self.response_time_cdf.push(sojourn);
        }

        if let Some(next_job) = self.queue.pop() {
            self.server.start(next_job);
            self.schedule_departure(next_job);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        sim.start_service(mu);
        sim.run_until(500_000.0);
        let util = sim.server_utilization();
        let err = (util - rho).abs() / rho;
        assert!(
            err < 0.01,
            "utilization error {err:.4} (got {util:.4}, expected {rho:.4})"
        );
    }
}
