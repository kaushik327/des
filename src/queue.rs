use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

#[derive(Debug, Default)]
pub(crate) struct FcfsQueue {
    inner: VecDeque<(u64, f64)>,
}

impl FcfsQueue {
    fn push(&mut self, job_id: u64, remaining: f64) {
        self.inner.push_back((job_id, remaining));
    }
    fn pop(&mut self) -> Option<(u64, f64)> {
        self.inner.pop_front()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct SrptEntry {
    remaining: f64,
    job_id: u64,
}

impl PartialEq for SrptEntry {
    fn eq(&self, other: &Self) -> bool {
        self.job_id == other.job_id
    }
}
impl Eq for SrptEntry {}
impl PartialOrd for SrptEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SrptEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.remaining
            .total_cmp(&other.remaining)
            .then(self.job_id.cmp(&other.job_id))
    }
}

#[derive(Debug, Default)]
pub(crate) struct SrptQueue {
    heap: BinaryHeap<std::cmp::Reverse<SrptEntry>>,
}

impl SrptQueue {
    fn push(&mut self, job_id: u64, remaining: f64) {
        self.heap
            .push(std::cmp::Reverse(SrptEntry { remaining, job_id }));
    }
    fn pop(&mut self) -> Option<(u64, f64)> {
        self.heap
            .pop()
            .map(|std::cmp::Reverse(e)| (e.job_id, e.remaining))
    }
    fn len(&self) -> usize {
        self.heap.len()
    }
}

/// Scheduling discipline for the waiting queue.
#[derive(Debug)]
pub enum SimQueue {
    Fcfs(FcfsQueue),
    Srpt(SrptQueue),
}

impl SimQueue {
    pub fn fcfs() -> Self {
        SimQueue::Fcfs(FcfsQueue::default())
    }
    pub fn srpt() -> Self {
        SimQueue::Srpt(SrptQueue::default())
    }

    pub fn push(&mut self, job_id: u64, remaining: f64) {
        match self {
            SimQueue::Fcfs(q) => q.push(job_id, remaining),
            SimQueue::Srpt(q) => q.push(job_id, remaining),
        }
    }

    pub fn pop(&mut self) -> Option<(u64, f64)> {
        match self {
            SimQueue::Fcfs(q) => q.pop(),
            SimQueue::Srpt(q) => q.pop(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SimQueue::Fcfs(q) => q.len(),
            SimQueue::Srpt(q) => q.len(),
        }
    }
}
