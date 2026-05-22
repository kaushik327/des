/// Monotonically-advancing simulation clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SimClock {
    pub time: f64,
}

impl SimClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance to `t`. Panics if `t < self.time` (causality violation).
    pub fn advance_to(&mut self, t: f64) {
        assert!(
            t >= self.time,
            "clock went backwards: {} -> {}",
            self.time,
            t
        );
        self.time = t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advances_forward() {
        let mut clock = SimClock::new();
        clock.advance_to(1.0);
        clock.advance_to(1.0);
        clock.advance_to(5.0);
        assert_eq!(clock.time, 5.0);
    }

    #[test]
    #[should_panic(expected = "clock went backwards")]
    fn panics_on_backwards_time() {
        let mut clock = SimClock::new();
        clock.advance_to(2.0);
        clock.advance_to(1.0);
    }
}
