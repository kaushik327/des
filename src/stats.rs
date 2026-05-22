//! Online statistics: running mean/variance and empirical tail probabilities.

/// Welford's online algorithm for running mean and sample variance.
/// Single-pass, numerically stable. Call `update` after each observation.
#[derive(Debug, Default, Clone)]
pub struct Welford {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// Sample variance (Bessel-corrected, N−1 denominator).
    pub fn variance(&self) -> Option<f64> {
        (self.count > 1).then(|| self.m2 / (self.count - 1) as f64)
    }

    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }
}

/// Collects samples for empirical tail statistics.
/// Call `finish` once to sort, then query `tail_prob`.
#[derive(Debug, Default, Clone)]
pub struct EmpiricalCdf {
    samples: Vec<f64>,
    sorted: bool,
}

impl EmpiricalCdf {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, x: f64) {
        self.samples.push(x);
        self.sorted = false;
    }

    pub fn finish(&mut self) {
        self.samples.sort_by(f64::total_cmp);
        self.sorted = true;
    }

    /// P{X > threshold}. Requires `finish` to have been called.
    pub fn tail_prob(&self, threshold: f64) -> f64 {
        debug_assert!(self.sorted, "call finish() before querying tail_prob");
        let n = self.samples.len();
        if n == 0 {
            return 0.0;
        }
        let exceed = self.samples.partition_point(|&x| x <= threshold);
        (n - exceed) as f64 / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_mean_and_variance() {
        let mut w = Welford::new();
        for x in [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.update(x);
        }
        assert!((w.mean().unwrap() - 5.0).abs() < 1e-10);
        assert!((w.variance().unwrap() - 4.571_428).abs() < 1e-5);
    }

    #[test]
    fn welford_single_obs_has_no_variance() {
        let mut w = Welford::new();
        w.update(3.0);
        assert_eq!(w.mean(), Some(3.0));
        assert!(w.variance().is_none());
    }

    #[test]
    fn empirical_cdf_tail_prob() {
        let mut cdf = EmpiricalCdf::new();
        for x in [1.0_f64, 2.0, 3.0, 4.0, 5.0] {
            cdf.push(x);
        }
        cdf.finish();
        // 4 and 5 exceed 3.0, so P{X > 3} = 2/5
        assert!((cdf.tail_prob(3.0) - 0.4).abs() < 1e-10);
    }
}
