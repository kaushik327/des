mod calendar;
mod clock;
mod distributions;
mod event;
mod sim;

use sim::Simulation;

fn main() {
    let lambda = 2.0;
    let end_time = 10_000.0;

    let mut sim = Simulation::with_seed(0);
    sim.start_arrivals(lambda);
    sim.run_until(end_time);

    println!(
        "λ={lambda}  T={end_time}  arrivals={}  empirical rate={:.4}",
        sim.arrivals_processed,
        sim.arrivals_processed as f64 / end_time,
    );
}
