mod calendar;
mod clock;
mod event;
mod sim;

use event::{Event, EventKind};
use sim::Simulation;

fn main() {
    let mut sim = Simulation::new();
    sim.schedule(Event::new(3.0, EventKind::Arrival { job_id: 3 }));
    sim.schedule(Event::new(1.0, EventKind::Arrival { job_id: 1 }));
    sim.schedule(Event::new(
        2.5,
        EventKind::Departure {
            job_id: 1,
            server_id: 0,
        },
    ));
    sim.schedule(Event::new(1.5, EventKind::Arrival { job_id: 2 }));
    sim.run_until(f64::INFINITY);
}
