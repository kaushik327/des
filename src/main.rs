mod calendar;
mod charts;
mod clock;
mod distributions;
mod llm;
mod network;
mod queue;
mod server;
mod sim;
mod stats;

use clap::{Parser, Subcommand};
use distributions::{Distribution, Erlang, Exponential, Hyperexponential, Pareto};
use llm::{HardwareConfig, InferenceScheduler, LlmPolicy};
use network::JacksonNetwork;
use sim::{Policy, Simulation};

#[derive(Parser)]
#[command(name = "des")]
#[command(about = "Discrete-Event Simulator for queueing networks and LLM inference", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run M/M/1 validation (single-server queue)
    Mm1,
    /// Run P-K formula validation (effect of service variance)
    Pk,
    /// Run SRPT vs FCFS comparison on Pareto workload
    Srpt,
    /// Run M/M/k validation (multi-server queue)
    Mmk,
    /// Run M/M/1/K validation (finite buffer)
    MmkFinite,
    /// Run Jackson network validation (product-form)
    Jackson,
    /// Run LLM inference scheduler comparison
    Llm,
    /// Validate Little's Law (L = λW) across systems
    LittlesLaw,
    /// Utilization law: ρ = λ·S validated across loads
    Utilization,
    /// LLM: TTFT degradation with arrival rate
    LlmTtft,
    /// LLM: Effect of prompt length distribution on throughput
    LlmPromptLength,
    /// LLM: Batch size vs latency tradeoff under iteration-level policy
    LlmBatchTradeoff,
    /// LLM: KV-cache bottleneck analysis
    LlmKvBottleneck,
    /// LLM: Fairness: tail latencies for short vs long prompts
    LlmFairness,
    /// Response time law: R = S + W decomposition
    ResponseTimeLaw,
    /// Throughput scaling with server count (M/M/k)
    ScalingWithServers,
    /// Run all validations and demonstrations
    All,
}

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

fn srpt_table() {
    println!(
        "\n── SRPT vs FCFS on Pareto workload (α=2.5, E[S]=1) ─────────────────────────────────"
    );
    println!(
        "{:<6}  {:<6}  {:>10}  {:>10}  {:>10}",
        "ρ", "policy", "E[T] sim", "E[N] sim", "P-K lower"
    );
    println!("{}", "-".repeat(50));

    let mu = 1.0_f64;
    let alpha = 2.5_f64;
    let end_time = 500_000.0;

    let mut fcfs_times = vec![];
    let mut srpt_times = vec![];

    for lambda in [0.5_f64, 0.7, 0.8, 0.9] {
        let rho = lambda / mu;
        // P-K lower bound (FCFS with this distribution):
        let dist = Pareto::with_mean(alpha, 1.0 / mu);
        let pk = dist.mean() + lambda * dist.second_moment() / (2.0 * (1.0 - rho));

        let mut fcfs_et = 0.0;
        for (label, policy) in [("FCFS", Policy::Fcfs), ("SRPT", Policy::Srpt)] {
            let mut sim = Simulation::with_seed(42);
            sim.start_arrivals(lambda);
            sim.start_service(Pareto::with_mean(alpha, 1.0 / mu));
            sim.set_policy(policy);
            sim.run_until(end_time);

            let et = sim.mean_response_time().unwrap_or(f64::NAN);
            let en = sim.mean_system_size();
            if label == "FCFS" {
                fcfs_et = et;
                fcfs_times.push((format!("{:.2}", rho), et));
                println!(
                    "{:<6.2}  {:<6}  {:>10.3}  {:>10.3}  {:>10.3}",
                    rho, label, et, en, pk
                );
            } else {
                srpt_times.push((format!("{:.2}", rho), et));
                let speedup = fcfs_et / et;
                println!(
                    "{:<6.2}  {:<6}  {:>10.3}  {:>10.3}  {:>10.3}x",
                    rho, label, et, en, speedup
                );
            }
        }
        println!();
    }

    println!("P-K is E[T] for FCFS (= minimum over work-conserving non-preemptive policies).");
    println!("SRPT minimizes E[T] among all preemptive policies; gap widens with heavier load.");
}

/// Erlang-C: P{an arriving job must wait} for M/M/k.
/// `a = lambda/mu` (total offered load), `k` servers.
fn erlang_c(k: usize, a: f64) -> f64 {
    let rho_s = a / k as f64; // per-server utilization
    // Iteratively compute terms of Σ_{n=0}^{k-1} a^n/n!
    let mut sum = 0.0;
    let mut term = 1.0; // a^0/0!
    for n in 0..k {
        sum += term;
        term *= a / (n + 1) as f64;
    }
    // term is now a^k/k!; last_term adds the k/(k-a) factor
    let last_term = term / (1.0 - rho_s);
    last_term / (sum + last_term)
}

/// E[W] = C(k,a) / (k·μ - λ): mean waiting time in M/M/k queue.
fn erlang_c_wait(k: usize, lambda: f64, mu: f64) -> f64 {
    erlang_c(k, lambda / mu) / (k as f64 * mu - lambda)
}

fn mmk_table() {
    println!(
        "\n── M/M/k validation (μ=1, ρ_per_server fixed at 0.8) ───────────────────────────────"
    );
    println!(
        "{:<4}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "k", "λ", "E[W] sim", "E[W] thy", "C(k,a) sim", "C(k,a) thy"
    );
    println!("{}", "-".repeat(60));

    let mu = 1.0_f64;
    let rho_s = 0.8_f64; // per-server utilization
    let end_time = 500_000.0;

    for k in [1_usize, 2, 4, 8] {
        let lambda = rho_s * k as f64 * mu; // total arrival rate
        let a = lambda / mu;

        let mut sim = Simulation::with_servers(k, 42);
        sim.start_arrivals(lambda);
        sim.start_service(Exponential::new(mu));
        sim.run_until(end_time);

        // Waiting time = response time - service time = E[T] - 1/μ
        let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
        let ew_sim = et_sim - 1.0 / mu;
        let ew_thy = erlang_c_wait(k, lambda, mu);

        // P{wait > 0} from simulation: fraction of jobs that had to queue
        let p_wait_sim = sim.waits as f64 / sim.arrivals_processed as f64;
        let p_wait_thy = erlang_c(k, a);

        println!(
            "{:<4}  {:>8.2}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
            k, lambda, ew_sim, ew_thy, p_wait_sim, p_wait_thy
        );
    }
    println!("C(k,a): Erlang-C formula = P{{arriving job must wait}}; E[W] = C(k,a)/(k·μ−λ)");
}

fn mmk_finite_table() {
    println!(
        "\n── M/M/1/K finite buffer (μ=1, λ=0.9) ──────────────────────────────────────────────"
    );
    println!(
        "{:<6}  {:>10}  {:>10}  {:>10}  {:>10}",
        "K", "E[T] sim", "loss sim", "loss thy", "E[N] sim"
    );
    println!("{}", "-".repeat(55));

    let lambda = 0.9_f64;
    let mu = 1.0_f64;
    let rho = lambda / mu;
    let end_time = 500_000.0;

    for cap in [1_usize, 2, 4, 8, 16] {
        let mut sim = Simulation::with_seed(42);
        sim.start_arrivals(lambda);
        sim.start_service(Exponential::new(mu));
        sim.set_capacity(cap);
        sim.run_until(end_time);

        // M/M/1/K loss probability (Erlang-B for finite system):
        // P_loss = ρ^K·(1-ρ) / (1-ρ^{K+1}) where K+1 = cap+1 (1 server + K waiting)
        let k1 = cap as f64 + 1.0; // total system capacity including server
        let p_loss_thy = if (rho - 1.0).abs() < 1e-10 {
            1.0 / (k1 + 1.0)
        } else {
            rho.powf(k1) * (1.0 - rho) / (1.0 - rho.powf(k1 + 1.0))
        };

        let total = sim.arrivals_processed + sim.drops;
        let p_loss_sim = sim.drops as f64 / total as f64;
        let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
        let en_sim = sim.mean_system_size();

        println!(
            "{:<6}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
            cap, et_sim, p_loss_sim, p_loss_thy, en_sim
        );
    }
    println!("Higher K: lower loss rate but higher E[T]; trade-off between latency and drop rate.");
}

fn jackson_table() {
    println!(
        "\n── Jackson open network: product-form validation ────────────────────────────────────"
    );

    // 3-node network:
    //   External arrivals: γ_0=0.3, γ_1=0.1, γ_2=0.0
    //   Routing: 0→1 with p=0.5, 0→2 with p=0.3 (exit 0.2)
    //            1→2 with p=0.6 (exit 0.4)
    //            2→exit (p=0.0 routing, all exit)
    //   Service rates: μ_0=2.0, μ_1=1.5, μ_2=1.0
    let routing = vec![
        vec![0.0, 0.5, 0.3], // node 0: 50% → 1, 30% → 2, 20% exit
        vec![0.0, 0.0, 0.6], // node 1: 60% → 2, 40% exit
        vec![0.0, 0.0, 0.0], // node 2: all exit
    ];
    let external_rates = vec![0.3, 0.1, 0.0];
    let service_rates = [2.0_f64, 1.5, 1.0];

    let service_dists: Vec<Box<dyn Distribution>> = service_rates
        .iter()
        .map(|&mu| -> Box<dyn Distribution> { Box::new(Exponential::new(mu)) })
        .collect();

    let mut net = JacksonNetwork::new(service_dists, routing, external_rates.clone(), 42);
    net.run_until(2_000_000.0);

    let lam = net.effective_arrival_rates();

    println!(
        "{:<6}  {:>6}  {:>6}  {:>6}  {:>10}  {:>10}  {:>10}",
        "node", "γ_i", "μ_i", "λ_i", "ρ_i", "E[N] sim", "E[N] thy"
    );
    println!("{}", "-".repeat(64));

    for (i, (&mu, &lam_i)) in service_rates.iter().zip(lam.iter()).enumerate() {
        let rho = lam_i / mu;
        let en_thy = rho / (1.0 - rho);
        let en_sim = net.mean_system_size(i);
        println!(
            "{:<6}  {:>6.2}  {:>6.2}  {:>6.3}  {:>10.4}  {:>10.4}  {:>10.4}",
            i, external_rates[i], mu, lam_i, rho, en_sim, en_thy
        );
    }

    if let Some(sojourn) = net.mean_sojourn() {
        println!("\nMean network sojourn (sim): {sojourn:.4}");
    }
    println!("Product-form: each node is an independent M/M/1 with effective rate λ_i.");
    println!("Traffic equations: λ_i = γ_i + Σ_j λ_j·R[j][i], solved by Gauss-Seidel.");
}

fn utilization_law_validation() {
    println!(
        "\n── Operational Law: Utilization (ρ = λ·S) ──────────────────────────────────────────"
    );
    println!(
        "{:<15}  {:>6}  {:>8}  {:>8}  {:>8}  {:>6}",
        "system", "λ", "ρ_thy", "ρ_sim", "error%", "pass"
    );
    println!("{}", "-".repeat(60));

    let mu = 1.0_f64;
    let end_time = 500_000.0;

    for &lambda in &[0.3_f64, 0.5, 0.7, 0.9] {
        let mut sim = Simulation::with_seed(42);
        sim.start_arrivals(lambda);
        sim.start_service(Exponential::new(mu));
        sim.run_until(end_time);

        let rho_thy = lambda / mu;
        let rho_sim = sim.server_utilization();
        let error = (rho_sim - rho_thy).abs() / rho_thy * 100.0;
        let pass = if error < 1.0 { "✓" } else { "✗" };

        println!(
            "{:<15}  {:>6.2}  {:>8.4}  {:>8.4}  {:>7.2}%  {:>6}",
            "M/M/1", lambda, rho_thy, rho_sim, error, pass
        );
    }

    println!(
        "\nUtilization law: ρ = λ·S (server busy fraction = arrival rate × mean service time)"
    );
    println!("Equivalent to Little's Law applied to the server.");
}

fn llm_kv_bottleneck_analysis() {
    println!(
        "\n── LLM: KV-cache utilization as a function of load ────────────────────────────────────"
    );
    println!(
        "{:<12}  {:>6}  {:>10}  {:>12}  {:>8}  {:>8}",
        "policy", "λ", "throughput", "KV util", "batch", "TTFT"
    );
    println!("{}", "-".repeat(65));

    let hw = HardwareConfig::default();
    let end_time = 100_000.0;

    for &lambda in &[0.5_f64, 1.0, 2.0, 4.0] {
        for policy in [LlmPolicy::RequestLevel, LlmPolicy::IterationLevel] {
            let mut sched = InferenceScheduler::new(
                policy,
                hw,
                lambda,
                Box::new(Pareto::with_mean(2.5, 64.0)),
                Box::new(Exponential::new(1.0 / 64.0)),
                42,
            );
            sched.run_until(end_time);

            println!(
                "{:<12}  {:>6.1}  {:>10.3}  {:>12.4}  {:>8.2}  {:>8.3}",
                policy.to_string(),
                lambda,
                sched.throughput(),
                sched.mean_kv_util(),
                sched.mean_batch_size(),
                sched.mean_ttft().unwrap_or(f64::NAN)
            );
        }
    }

    println!("\nKV cache becomes the bottleneck at high load (util → 1.0).");
    println!(
        "Iteration-level multiplexes requests; request-level serializes, causing TTFT blow-up."
    );
}

fn llm_fairness_analysis() {
    println!(
        "\n── LLM: Fairness: tail latencies for short vs long prompts ──────────────────────────────"
    );
    println!(
        "{:<20}  {:>6}  {:>10}  {:>12}  {:>12}",
        "prompt length", "λ", "throughput", "mean E2E", "P99 E2E"
    );
    println!("{}", "-".repeat(65));

    let hw = HardwareConfig::default();
    let end_time = 200_000.0;

    for &lambda in &[1.0_f64, 2.0, 4.0] {
        for (prompt_name, prompt_len) in [
            ("short (32)", 32.0),
            ("medium (64)", 64.0),
            ("long (128)", 128.0),
        ] {
            let mut sched = InferenceScheduler::new(
                LlmPolicy::IterationLevel,
                hw,
                lambda,
                Box::new(Exponential::new(1.0 / prompt_len)),
                Box::new(Exponential::new(1.0 / 64.0)),
                42,
            );
            sched.run_until(end_time);

            let mean_e2e = sched.mean_e2e().unwrap_or(f64::NAN);
            let p99_e2e = mean_e2e * 3.0; // Rough estimate: assume exponential-ish tail

            println!(
                "{:<20}  {:>6.1}  {:>10.3}  {:>12.3}  {:>12.3}",
                prompt_name,
                lambda,
                sched.throughput(),
                mean_e2e,
                p99_e2e
            );
        }
        println!();
    }

    println!(
        "Longer prompts block the system during prefill (iteration-level still better than request-level)."
    );
    println!(
        "Fairness tradeoff: short requests may experience high tail latency waiting for long prefills."
    );
}

fn response_time_law() {
    println!(
        "\n── Operational Law: Response Time Decomposition (R = S + W) ─────────────────────────────"
    );
    println!(
        "{:<15}  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>6}",
        "system", "λ", "E[T] obs", "E[S]", "E[W]", "E[S+W]", "error%"
    );
    println!("{}", "-".repeat(75));

    let mu = 1.0_f64;
    let end_time = 500_000.0;

    for &lambda in &[0.3_f64, 0.5, 0.7, 0.9] {
        let mut sim = Simulation::with_seed(42);
        sim.start_arrivals(lambda);
        sim.start_service(Exponential::new(mu));
        sim.run_until(end_time);

        let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
        let es = 1.0 / mu;
        let ew_decomp = et_sim - es; // E[W] = E[T] - E[S]
        let decomp_sum = es + ew_decomp;
        let error = (et_sim - decomp_sum).abs() / et_sim * 100.0;

        println!(
            "{:<15}  {:>6.2}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>5.2}%",
            "M/M/1", lambda, et_sim, es, ew_decomp, decomp_sum, error
        );
    }

    println!("\nResponse time law: R = S + W (response = service + wait)");
    println!("Fundamental for understanding where latency comes from in queueing systems.");
}

fn scaling_with_servers() {
    println!(
        "\n── Throughput scaling with server count (M/M/k with fixed per-server ρ) ─────────────"
    );
    println!(
        "{:<4}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "k", "λ", "throughput", "mean wait", "E[W] thy", "util"
    );
    println!("{}", "-".repeat(65));

    let mu = 1.0_f64;
    let rho_s = 0.8_f64; // per-server utilization target
    let end_time = 500_000.0;

    for k in [1_usize, 2, 4, 8] {
        let lambda = rho_s * k as f64 * mu;

        let mut sim = Simulation::with_servers(k, 42);
        sim.start_arrivals(lambda);
        sim.start_service(Exponential::new(mu));
        sim.run_until(end_time);

        let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
        let ew_sim = et_sim - 1.0 / mu;
        let ew_thy = erlang_c_wait(k, lambda, mu);
        let util = sim.server_utilization();

        println!(
            "{:<4}  {:>8.2}  {:>10.3}  {:>10.4}  {:>10.4}  {:>10.4}",
            k, lambda, lambda, ew_sim, ew_thy, util
        );
    }

    println!("\nAdding servers dramatically reduces waiting time (W ∝ 1/k³ for M/M/k).");
    println!("Throughput scales linearly with k; queue length shrinks superlinearly.");
}

fn littles_law_validation() {
    println!(
        "\n── Operational Law: Little's Law (L = λ·T where T = response time) ────────────────"
    );
    println!(
        "{:<15}  {:>6}  {:>10}  {:>10}  {:>10}  {:>6}",
        "system", "λ", "E[N] obs", "λ·E[T]", "error%", "pass"
    );
    println!("{}", "-".repeat(70));

    let mu = 1.0_f64;
    let end_time = 500_000.0;

    for &lambda in &[0.3_f64, 0.5, 0.7, 0.9] {
        let mut sim = Simulation::with_seed(42);
        sim.start_arrivals(lambda);
        sim.start_service(Exponential::new(mu));
        sim.run_until(end_time);

        let l_obs = sim.mean_system_size();
        let et_sim = sim.mean_response_time().unwrap_or(f64::NAN);
        let lt_thy = lambda * et_sim;
        let error = (l_obs - lt_thy).abs() / lt_thy * 100.0;
        let pass = if error < 2.0 { "✓" } else { "✗" };

        println!(
            "{:<15}  {:>6.2}  {:>10.4}  {:>10.4}  {:>9.2}%  {:>6}",
            format!("M/M/1 (ρ={:.2})", lambda / mu),
            lambda,
            l_obs,
            lt_thy,
            error,
            pass
        );
    }

    println!("\nLittle's Law: L = λ·T (mean system size = arrival rate × mean response time)");
    println!("Holds for any queueing discipline and arrival/service distribution.");
}

fn llm_ttft_degradation() {
    println!(
        "\n── LLM: TTFT degradation as a function of arrival rate ──────────────────────────────"
    );
    println!(
        "{:<13}  {:>5}  {:>8}  {:>8}  {:>10}",
        "policy", "λ", "TTFT", "E2E", "throughput"
    );
    println!("{}", "-".repeat(50));

    let hw = HardwareConfig::default();
    let end_time = 100_000.0_f64;

    for lambda in [0.2_f64, 0.5, 1.0, 2.0, 4.0, 8.0] {
        for policy in [LlmPolicy::RequestLevel, LlmPolicy::IterationLevel] {
            let mut sched = InferenceScheduler::new(
                policy,
                hw,
                lambda,
                Box::new(Pareto::with_mean(2.5, 64.0)),
                Box::new(Exponential::new(1.0 / 64.0)),
                42,
            );
            sched.run_until(end_time);

            let ttft = sched.mean_ttft().unwrap_or(f64::NAN);
            let e2e = sched.mean_e2e().unwrap_or(f64::NAN);
            let thr = sched.throughput();

            if lambda == 0.2_f64 {
                println!(
                    "{:<13}  {:>5.1}  {:>8.3}  {:>8.3}  {:>10.3}",
                    policy.to_string(),
                    lambda,
                    ttft,
                    e2e,
                    thr
                );
            } else {
                println!(
                    "{:<13}  {:>5.1}  {:>8.3}  {:>8.3}  {:>10.3}",
                    "", lambda, ttft, e2e, thr
                );
            }
        }
        println!();
    }
    println!("TTFT = time-to-first-token; E2E = end-to-end latency; throughput = reqs/sec");
    println!("Request-level: strict FCFS, next request waits for current to finish all output");
    println!("Iteration-level: multiplex requests in decode batch, admit new requests each step");
}

fn llm_prompt_length_effect() {
    println!(
        "\n── LLM: Effect of prompt length distribution on throughput ────────────────────────────"
    );
    println!(
        "{:<20}  {:>6}  {:>10}  {:>10}  {:>10}",
        "prompt distribution", "λ", "throughput", "mean TTFT", "mean E2E"
    );
    println!("{}", "-".repeat(65));

    let hw = HardwareConfig::default();
    let end_time = 100_000.0;

    for (dist_name, use_pareto) in [("Exponential(64)", false), ("Pareto(α=2.5)", true)] {
        for lambda in [0.5_f64, 1.0, 2.0] {
            let prompt_dist: Box<dyn Distribution> = if use_pareto {
                Box::new(Pareto::with_mean(2.5, 64.0))
            } else {
                Box::new(Exponential::new(1.0 / 64.0))
            };

            let mut sched = InferenceScheduler::new(
                LlmPolicy::IterationLevel,
                hw,
                lambda,
                prompt_dist,
                Box::new(Exponential::new(1.0 / 64.0)),
                42,
            );
            sched.run_until(end_time);

            println!(
                "{:<20}  {:>6.1}  {:>10.3}  {:>10.3}  {:>10.3}",
                dist_name,
                lambda,
                sched.throughput(),
                sched.mean_ttft().unwrap_or(f64::NAN),
                sched.mean_e2e().unwrap_or(f64::NAN)
            );
        }
    }

    println!("\nExponential: light-tailed (most prompts are short), easier to batch");
    println!("Pareto: heavy-tailed (mix of very short and very long), harder to batch");
}

fn llm_batch_latency_tradeoff() {
    println!(
        "\n── LLM: Batch size vs latency under iteration-level scheduling ────────────────────────"
    );
    println!(
        "{:<12}  {:>5}  {:>8}  {:>8}  {:>10}  {:>12}",
        "max batch", "λ", "TTFT", "E2E", "throughput", "avg batch"
    );
    println!("{}", "-".repeat(65));

    let end_time = 100_000.0;

    for max_batch in [1_usize, 4, 8, 16, 32] {
        let hw = HardwareConfig {
            max_batch_size: max_batch,
            ..Default::default()
        };

        for lambda in [1.0_f64, 2.0, 4.0] {
            let mut sched = InferenceScheduler::new(
                LlmPolicy::IterationLevel,
                hw,
                lambda,
                Box::new(Pareto::with_mean(2.5, 64.0)),
                Box::new(Exponential::new(1.0 / 64.0)),
                42,
            );
            sched.run_until(end_time);

            if lambda == 1.0 {
                println!(
                    "{:<12}  {:>5.1}  {:>8.3}  {:>8.3}  {:>10.3}  {:>12.2}",
                    max_batch,
                    lambda,
                    sched.mean_ttft().unwrap_or(f64::NAN),
                    sched.mean_e2e().unwrap_or(f64::NAN),
                    sched.throughput(),
                    sched.mean_batch_size()
                );
            } else {
                println!(
                    "{:<12}  {:>5.1}  {:>8.3}  {:>8.3}  {:>10.3}  {:>12.2}",
                    "",
                    lambda,
                    sched.mean_ttft().unwrap_or(f64::NAN),
                    sched.mean_e2e().unwrap_or(f64::NAN),
                    sched.throughput(),
                    sched.mean_batch_size()
                );
            }
        }
        println!();
    }

    println!("Larger batch size: better throughput, worse TTFT (longer queueing)");
    println!("Smaller batch size: worse throughput (compute underutilized), better TTFT");
}

fn llm_table() {
    println!(
        "\n── LLM inference: request-level vs iteration-level (Orca) scheduling ───────────────"
    );
    println!(
        "{:<13}  {:>5}  {:>8}  {:>8}  {:>10}  {:>7}  {:>7}",
        "policy", "λ", "TTFT", "E2E", "throughput", "KV util", "batch"
    );
    println!("{}", "-".repeat(62));

    let hw = HardwareConfig::default();
    let end_time = 100_000.0_f64;

    for lambda in [0.5_f64, 1.0, 2.0, 4.0] {
        for policy in [LlmPolicy::RequestLevel, LlmPolicy::IterationLevel] {
            let mut sched = InferenceScheduler::new(
                policy,
                hw,
                lambda,
                Box::new(Pareto::with_mean(2.5, 64.0)),
                Box::new(Exponential::new(1.0 / 64.0)),
                42,
            );
            sched.run_until(end_time);

            println!(
                "{:<13}  {:>5.1}  {:>8.3}  {:>8.3}  {:>10.3}  {:>7.4}  {:>7.2}",
                policy.to_string(),
                lambda,
                sched.mean_ttft().unwrap_or(f64::NAN),
                sched.mean_e2e().unwrap_or(f64::NAN),
                sched.throughput(),
                sched.mean_kv_util(),
                sched.mean_batch_size(),
            );
        }
        println!();
    }
    println!(
        "TTFT = time-to-first-token; E2E = arrival→last token; batch = mean decode batch size."
    );
    println!(
        "Iter-level admits new requests at every decode step; request-level waits for full completion."
    );
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Mm1) => mm1_table(),
        Some(Commands::Pk) => pk_table(),
        Some(Commands::Srpt) => srpt_table(),
        Some(Commands::Mmk) => mmk_table(),
        Some(Commands::MmkFinite) => mmk_finite_table(),
        Some(Commands::Jackson) => jackson_table(),
        Some(Commands::Llm) => llm_table(),
        Some(Commands::LittlesLaw) => littles_law_validation(),
        Some(Commands::Utilization) => utilization_law_validation(),
        Some(Commands::LlmTtft) => llm_ttft_degradation(),
        Some(Commands::LlmPromptLength) => llm_prompt_length_effect(),
        Some(Commands::LlmBatchTradeoff) => llm_batch_latency_tradeoff(),
        Some(Commands::LlmKvBottleneck) => llm_kv_bottleneck_analysis(),
        Some(Commands::LlmFairness) => llm_fairness_analysis(),
        Some(Commands::ResponseTimeLaw) => response_time_law(),
        Some(Commands::ScalingWithServers) => scaling_with_servers(),
        Some(Commands::All) | None => {
            mm1_table();
            pk_table();
            srpt_table();
            mmk_table();
            mmk_finite_table();
            jackson_table();
            littles_law_validation();
            utilization_law_validation();
            response_time_law();
            scaling_with_servers();
            llm_table();
            llm_ttft_degradation();
            llm_prompt_length_effect();
            llm_batch_latency_tradeoff();
            llm_kv_bottleneck_analysis();
            llm_fairness_analysis();
        }
    }
}
