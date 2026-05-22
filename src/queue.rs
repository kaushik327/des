use std::collections::VecDeque;

/// First-come-first-served queue of job IDs.
/// Will become a pluggable `SchedulingPolicy` trait in a later step.
#[derive(Debug, Default)]
pub struct FcfsQueue {
    inner: VecDeque<u64>,
}

impl FcfsQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, job_id: u64) {
        self.inner.push_back(job_id);
    }

    pub fn pop(&mut self) -> Option<u64> {
        self.inner.pop_front()
    }

    #[allow(dead_code)] // needed for finite-buffer drop logic and stats collection
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[allow(dead_code)] // needed for finite-buffer drop logic and stats collection
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
