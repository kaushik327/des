mod calendar;
mod clock;
mod distributions;
mod event;
mod queue;
mod server;
mod sim;
mod stats;

use sim::Simulation;

fn main() {
    println!(
        "{:<6}  {:>9}  {:>9}  {:>9}  {:>9}  {:>6}  {:>9}  {:>9}  {:>9}  {:>9}",
        "ρ",
        "E[T] sim",
        "E[T] thy",
        "σ[T] sim",
        "σ[T] thy",
        "util",
        "E[N] sim",
        "E[N] thy",
        "P>2ET sim",
        "P>2ET thy"
    );
    println!("{}", "-".repeat(95));

    let mu = 1.0_f64;
    for lambda in [0.5_f64, 0.7, 0.9, 0.99] {
        let rho = lambda / mu;
        // ρ=0.99 mixes in O(1/(1-ρ)²) ≈ 40k time units; run longer to average it out.
        let end_time = if rho >= 0.99 { 2_000_000.0 } else { 200_000.0 };
        let mut sim = Simulation::with_seed(42);
        sim.start_arrivals(lambda);
        sim.start_service(mu);
        sim.run_until(end_time);

        let et_thy = 1.0 / (mu - lambda);
        let en_thy = rho / (1.0 - rho);
        // For M/M/1, T ~ Exp(μ-λ), so σ[T] = E[T] and P{T > 2·E[T]} = e^{-2}
        let sig_thy = et_thy;
        let tail_thy = (-2.0_f64).exp();

        let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
        let sig_sim = sim.response_time_std_dev().unwrap_or(f64::NAN);
        let util = sim.server_utilization();
        let en_sim = sim.mean_system_size();
        let tail_sim = sim.tail_prob(2.0 * et_thy);

        println!(
            "{:<6.2}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>6.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}",
            rho, et_sim, et_thy, sig_sim, sig_thy, util, en_sim, en_thy, tail_sim, tail_thy
        );
    }

    println!(
        "\ntheory: σ[T]=E[T] for M/M/1 (response time is exponential); P{{T>2·E[T]}} = e⁻² ≈ {:.4}",
        (-2.0_f64).exp()
    );
}
