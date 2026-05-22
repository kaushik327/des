use rand::{SeedableRng, rngs::SmallRng};

use crate::calendar::EventCalendar;
use crate::clock::SimClock;
use crate::distributions::{Distribution, Exponential};
use crate::event::{Event, EventKind};

#[derive(Debug)]
pub struct Simulation {
    pub clock: SimClock,
    pub arrivals_processed: u64,
    calendar: EventCalendar,
    rng: SmallRng,
    arrival_dist: Option<Exponential>,
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
            next_job_id: 0,
        }
    }

    /// Begin a Poisson arrival process at rate `lambda`. Schedules the first
    /// arrival immediately; subsequent arrivals are scheduled on each dispatch.
    pub fn start_arrivals(&mut self, lambda: f64) {
        let dist = Exponential::new(lambda);
        self.arrival_dist = Some(dist);
        self.schedule_next_arrival();
    }

    #[allow(dead_code)] // used by tests and for manually seeding non-Poisson events
    pub fn schedule(&mut self, event: Event) {
        self.calendar.push(event);
    }

    /// Run until no events remain or the next event exceeds `end_time`.
    pub fn run_until(&mut self, end_time: f64) {
        while let Some(event) = self.calendar.pop_next() {
            if event.timestamp > end_time {
                // Put it back — callers may resume or inspect remaining state.
                self.calendar.push(event);
                break;
            }
            self.clock.advance_to(event.timestamp);
            self.dispatch(event);
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
            EventKind::Departure { job_id, server_id } => self.on_departure(job_id, server_id),
        }
    }

    fn on_arrival(&mut self, _job_id: u64) {
        self.arrivals_processed += 1;
        self.schedule_next_arrival();
    }

    fn on_departure(&mut self, _job_id: u64, _server_id: usize) {}
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
}
