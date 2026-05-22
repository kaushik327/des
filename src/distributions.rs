use rand::RngExt;

pub trait Distribution {
    fn sample<R: rand::Rng>(&self, rng: &mut R) -> f64;
    #[allow(dead_code)] // needed for closed-form validation (P-K formula, Erlang-C, etc.)
    fn mean(&self) -> f64;
    #[allow(dead_code)] // needed for closed-form validation (E[S²] = Var(S) + E[S]²)
    fn variance(&self) -> f64;
}

/// Exponential(λ): interarrival times for a Poisson process, M/M/1 service times.
/// Inverse transform: -ln(U) / λ where U ~ Uniform(0, 1).
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
    fn sample<R: rand::Rng>(&self, rng: &mut R) -> f64 {
        // 1 - U is Uniform(0, 1] so ln is never -inf.
        let u: f64 = rng.random();
        -(1.0 - u).ln() / self.lambda
    }

    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    fn variance(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    const N: usize = 200_000;
    const TOL: f64 = 0.01; // 1 %

    fn sample_stats(dist: &impl Distribution, rng: &mut SmallRng) -> (f64, f64) {
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
        let err = (mean - dist.mean()).abs() / dist.mean();
        assert!(err < TOL, "mean relative error {err:.4} > {TOL}");
    }

    #[test]
    fn exponential_variance() {
        let dist = Exponential::new(3.0);
        let mut rng = SmallRng::seed_from_u64(2);
        let (_, var) = sample_stats(&dist, &mut rng);
        let err = (var - dist.variance()).abs() / dist.variance();
        assert!(err < TOL, "variance relative error {err:.4} > {TOL}");
    }

    #[test]
    #[should_panic(expected = "lambda must be positive")]
    fn rejects_non_positive_lambda() {
        Exponential::new(0.0);
    }
}
