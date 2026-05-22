//! Probability distributions for simulation workloads.
//!
//! All distributions implement [`Distribution`], which is object-safe so they can
//! be stored as `Box<dyn Distribution>` for runtime dispatch.

use rand::Rng;

fn uniform(rng: &mut dyn Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// A probability distribution that produces samples and reports its moments.
pub trait Distribution: std::fmt::Debug {
    fn sample(&self, rng: &mut dyn Rng) -> f64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;

    /// E[X²] = Var(X) + E[X]²; used by the P-K formula.
    fn second_moment(&self) -> f64 {
        self.variance() + self.mean() * self.mean()
    }
}

/// Exponential(λ): interarrival times for a Poisson process, M/M/1 service times.
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    lambda: f64,
}

impl Exponential {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "lambda must be positive, got {lambda}");
        Self { lambda }
    }
}

impl Distribution for Exponential {
    fn sample(&self, rng: &mut dyn Rng) -> f64 {
        // 1-U is Uniform(0,1] so ln is never -inf.
        -(1.0 - uniform(rng)).ln() / self.lambda
    }

    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    fn variance(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }
}

/// Erlang(k, rate): sum of k independent Exponential(rate) phases.
/// CV² = 1/k — less variable than exponential.
#[derive(Debug, Clone, Copy)]
pub struct Erlang {
    k: u32,
    rate: f64,
}

impl Erlang {
    /// `k` phases each at `rate`; mean = k/rate.
    pub fn new(k: u32, rate: f64) -> Self {
        assert!(k > 0, "k must be at least 1");
        assert!(rate > 0.0, "rate must be positive, got {rate}");
        Self { k, rate }
    }
}

impl Distribution for Erlang {
    fn sample(&self, rng: &mut dyn Rng) -> f64 {
        // -Σ ln(Ui) / rate = -ln(Π Ui) / rate: log-product trick avoids underflow.
        let mut log_prod = 0.0_f64;
        for _ in 0..self.k {
            log_prod += (1.0 - uniform(rng)).ln();
        }
        -log_prod / self.rate
    }

    fn mean(&self) -> f64 {
        self.k as f64 / self.rate
    }

    fn variance(&self) -> f64 {
        self.k as f64 / (self.rate * self.rate)
    }
}

/// Hyperexponential: with probability p sample Exp(lambda1), else Exp(lambda2).
/// CV² > 1 — more variable than exponential.
#[derive(Debug, Clone, Copy)]
pub struct Hyperexponential {
    p: f64,
    lambda1: f64,
    lambda2: f64,
}

impl Hyperexponential {
    pub fn new(p: f64, lambda1: f64, lambda2: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0,1], got {p}");
        assert!(lambda1 > 0.0, "lambda1 must be positive, got {lambda1}");
        assert!(lambda2 > 0.0, "lambda2 must be positive, got {lambda2}");
        Self {
            p,
            lambda1,
            lambda2,
        }
    }

    /// Balanced hyperexponential with mean 1/mu and the given CV².
    /// Uses p=0.1 and solves the two-moment constraints analytically.
    pub fn balanced(mu: f64, cv_sq: f64) -> Self {
        assert!(
            cv_sq > 1.0,
            "hyperexponential requires CV² > 1, got {cv_sq}"
        );
        let p = 0.1_f64;
        // Quadratic 18c² - 36c + (19 - cv_sq) = 0; discriminant = 72·(cv_sq-1).
        let disc = (72.0 * (cv_sq - 1.0)).sqrt();
        let c = (36.0 - disc) / 36.0;
        let mean2 = c / mu;
        let mean1 = (1.0 / mu - (1.0 - p) * mean2) / p;
        Self::new(p, 1.0 / mean1, 1.0 / mean2)
    }
}

impl Distribution for Hyperexponential {
    fn sample(&self, rng: &mut dyn Rng) -> f64 {
        let rate = if uniform(rng) < self.p {
            self.lambda1
        } else {
            self.lambda2
        };
        -(1.0 - uniform(rng)).ln() / rate
    }

    fn mean(&self) -> f64 {
        self.p / self.lambda1 + (1.0 - self.p) / self.lambda2
    }

    fn variance(&self) -> f64 {
        let es2 = 2.0
            * (self.p / (self.lambda1 * self.lambda1)
                + (1.0 - self.p) / (self.lambda2 * self.lambda2));
        es2 - self.mean() * self.mean()
    }
}

/// Pareto(α, x_min): heavy-tailed distribution for real-world service times.
///
/// E[X] = α·x_min/(α−1) for α > 1; Var[X] is finite only for α > 2.
/// Inverse CDF: x = x_min · (1−U)^{−1/α}.
#[derive(Debug, Clone, Copy)]
pub struct Pareto {
    alpha: f64,
    x_min: f64,
}

impl Pareto {
    pub fn new(alpha: f64, x_min: f64) -> Self {
        assert!(alpha > 0.0, "alpha must be positive, got {alpha}");
        assert!(x_min > 0.0, "x_min must be positive, got {x_min}");
        Self { alpha, x_min }
    }

    /// Construct with given shape α and mean E[S] = mean (requires α > 1).
    pub fn with_mean(alpha: f64, mean: f64) -> Self {
        assert!(alpha > 1.0, "mean is finite only for α > 1, got {alpha}");
        Self::new(alpha, mean * (alpha - 1.0) / alpha)
    }
}

impl Distribution for Pareto {
    fn sample(&self, rng: &mut dyn Rng) -> f64 {
        self.x_min * (1.0 - uniform(rng)).powf(-1.0 / self.alpha)
    }

    fn mean(&self) -> f64 {
        if self.alpha > 1.0 {
            self.alpha * self.x_min / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }

    fn variance(&self) -> f64 {
        if self.alpha > 2.0 {
            self.x_min * self.x_min * self.alpha / ((self.alpha - 1.0).powi(2) * (self.alpha - 2.0))
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    const N: usize = 200_000;
    const TOL: f64 = 0.01;

    fn sample_stats(dist: &dyn Distribution, rng: &mut SmallRng) -> (f64, f64) {
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for i in 1..=N {
            let x = dist.sample(rng);
            let delta = x - mean;
            mean += delta / i as f64;
            m2 += delta * (x - mean);
        }
        (mean, m2 / (N - 1) as f64)
    }

    #[test]
    fn exponential_mean() {
        let dist = Exponential::new(3.0);
        let mut rng = SmallRng::seed_from_u64(1);
        let (mean, _) = sample_stats(&dist, &mut rng);
        assert!((mean - dist.mean()).abs() / dist.mean() < TOL);
    }

    #[test]
    fn exponential_variance() {
        let dist = Exponential::new(3.0);
        let mut rng = SmallRng::seed_from_u64(2);
        let (_, var) = sample_stats(&dist, &mut rng);
        assert!((var - dist.variance()).abs() / dist.variance() < TOL);
    }

    #[test]
    #[should_panic(expected = "lambda must be positive")]
    fn rejects_non_positive_lambda() {
        Exponential::new(0.0);
    }

    #[test]
    fn erlang_mean_and_variance() {
        let dist = Erlang::new(4, 2.0);
        let mut rng = SmallRng::seed_from_u64(3);
        let (mean, var) = sample_stats(&dist, &mut rng);
        assert!((mean - dist.mean()).abs() / dist.mean() < TOL);
        assert!((var - dist.variance()).abs() / dist.variance() < TOL);
    }

    #[test]
    fn hyperexponential_mean_and_variance() {
        let dist = Hyperexponential::balanced(1.0, 5.0);
        let mut rng = SmallRng::seed_from_u64(4);
        let (mean, var) = sample_stats(&dist, &mut rng);
        assert!((mean - dist.mean()).abs() / dist.mean() < TOL);
        assert!((var - dist.variance()).abs() / dist.variance() < 0.05);
    }

    #[test]
    fn pareto_mean() {
        let dist = Pareto::with_mean(3.0, 2.0);
        let mut rng = SmallRng::seed_from_u64(5);
        let (mean, _) = sample_stats(&dist, &mut rng);
        assert!((mean - dist.mean()).abs() / dist.mean() < TOL);
    }
}
