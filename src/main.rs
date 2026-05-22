mod calendar;
mod clock;
mod distributions;
mod event;
mod queue;
mod server;
mod sim;
mod stats;

use distributions::{Distribution, Erlang, Exponential, Hyperexponential};
use sim::Simulation;

/// P-K mean response time: E[T] = E[S] + λ·E[S²] / (2·(1−ρ))
fn pk_mean(lambda: f64, dist: &dyn Distribution) -> f64 {
    dist.mean() + lambda * dist.second_moment() / (2.0 * (1.0 - lambda * dist.mean()))
}

fn mm1_table() {
    println!(
        "── M/M/1 validation (μ=1) ──────────────────────────────────────────────────────────"
    );
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
        let end_time = if rho >= 0.99 { 2_000_000.0 } else { 200_000.0 };

        let mut sim = Simulation::with_seed(42);
        sim.start_arrivals(lambda);
        sim.start_service(Exponential::new(mu));
        sim.run_until(end_time);

        // For M/M/1: T ~ Exp(μ-λ), so σ[T] = E[T] and P{T > 2·E[T]} = e^{-2}
        let et_thy = 1.0 / (mu - lambda);
        let en_thy = rho / (1.0 - rho);
        let tail_thy = (-2.0_f64).exp();

        let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
        let sig_sim = sim.response_time_std_dev().unwrap_or(f64::NAN);
        let util = sim.server_utilization();
        let en_sim = sim.mean_system_size();
        let tail_sim = sim.tail_prob(2.0 * et_thy);

        println!(
            "{:<6.2}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>6.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}",
            rho, et_sim, et_thy, sig_sim, et_thy, util, en_sim, en_thy, tail_sim, tail_thy
        );
    }
    println!(
        "theory: σ[T]=E[T] for M/M/1; P{{T>2·E[T]}} = e⁻² ≈ {:.4}",
        (-2.0_f64).exp()
    );
}

fn pk_row(label: &str, lambda: f64, dist: impl Distribution + 'static, end_time: f64) {
    let cv_sq = dist.variance() / (dist.mean() * dist.mean());
    let et_pk = pk_mean(lambda, &dist);
    let en_pk = lambda * et_pk; // Little's law

    let mut sim = Simulation::with_seed(42);
    sim.start_arrivals(lambda);
    sim.start_service(dist);
    sim.run_until(end_time);

    let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
    let en_sim = sim.mean_system_size();
    println!(
        "{:<22}  {:>5.2}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}",
        label, cv_sq, et_sim, et_pk, en_sim, en_pk
    );
}

fn pk_table() {
    println!(
        "\n── P-K formula: effect of service variance (λ=0.9, E[S]=1) ─────────────────────────"
    );
    println!(
        "{:<22}  {:>5}  {:>9}  {:>9}  {:>9}  {:>9}",
        "distribution", "CV²", "E[T] sim", "E[T] P-K", "E[N] sim", "E[N] P-K"
    );
    println!("{}", "-".repeat(67));

    let lambda = 0.9_f64;
    let mu = 1.0_f64;
    let end_time = 500_000.0_f64;

    // Same ρ=0.9 and E[S]=1/μ=1, but increasing service variance.
    // P-K predicts higher variance → higher mean latency.
    pk_row(
        "Erlang-4 (CV²=0.25)",
        lambda,
        Erlang::new(4, 4.0 * mu),
        end_time,
    );
    pk_row(
        "Erlang-2 (CV²=0.5)",
        lambda,
        Erlang::new(2, 2.0 * mu),
        end_time,
    );
    pk_row(
        "Exponential (CV²=1)",
        lambda,
        Exponential::new(mu),
        end_time,
    );
    pk_row(
        "Hyperexp (CV²≈3)",
        lambda,
        Hyperexponential::balanced(mu, 3.0),
        end_time,
    );
    pk_row(
        "Hyperexp (CV²≈5)",
        lambda,
        Hyperexponential::balanced(mu, 5.0),
        end_time,
    );

    println!("theory: P-K formula E[T] = E[S] + λ·E[S²]/(2·(1−ρ)); higher CV² → higher latency");
}

fn main() {
    mm1_table();
    pk_table();
}
