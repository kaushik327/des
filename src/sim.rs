use crate::calendar::EventCalendar;
use crate::clock::SimClock;
use crate::event::{Event, EventKind};

#[derive(Debug)]
pub struct Simulation {
    pub clock: SimClock,
    calendar: EventCalendar,
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            clock: SimClock::new(),
            calendar: EventCalendar::new(),
        }
    }

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

    fn dispatch(&mut self, event: Event) {
        match event.kind {
            EventKind::Arrival { job_id } => self.on_arrival(job_id),
            EventKind::Departure { job_id, server_id } => self.on_departure(job_id, server_id),
        }
    }

    fn on_arrival(&mut self, job_id: u64) {
        println!("t={:.6}  Arrival    job={}", self.clock.time, job_id);
    }

    fn on_departure(&mut self, job_id: u64, server_id: usize) {
        println!(
            "t={:.6}  Departure  job={} server={}",
            self.clock.time, job_id, server_id
        );
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
        // SimClock::advance_to asserts monotonicity; this test passes iff no panic.
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
        // Only the t=1.0 event should have been processed.
        assert_eq!(sim.clock.time, 1.0);
    }

    #[test]
    fn empty_simulation_runs_cleanly() {
        let mut sim = Simulation::new();
        sim.run_until(100.0);
        assert_eq!(sim.clock.time, 0.0);
    }
}
