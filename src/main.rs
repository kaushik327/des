mod calendar;
mod clock;
mod distributions;
mod event;
mod queue;
mod server;
mod sim;

use sim::Simulation;

fn main() {
    println!(
        "{:<6}  {:>10}  {:>10}  {:>10}  {:>10}",
        "ρ", "E[T] sim", "E[T] theory", "E[N] sim", "E[N] theory"
    );
    println!("{}", "-".repeat(56));

    let mu = 1.0_f64;
    for lambda in [0.5_f64, 0.7, 0.9, 0.99] {
        let rho = lambda / mu;
        // ρ=0.99 mixes in O(1/(1-ρ)²) ≈ 40k time units; run longer to average it out.
        let end_time = if rho >= 0.99 { 2_000_000.0 } else { 200_000.0 };
        let mut sim = Simulation::with_seed(42);
        sim.start_arrivals(lambda);
        sim.start_service(mu);
        sim.run_until(end_time);

        let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
        let en_sim = sim.mean_system_size();
        let et_theory = 1.0 / (mu - lambda);
        let en_theory = rho / (1.0 - rho);

        println!(
            "{:<6.2}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
            rho, et_sim, et_theory, en_sim, en_theory
        );
    }
}
