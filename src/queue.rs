use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

// ── FCFS ───────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct FcfsQueue {
    inner: VecDeque<(u64, f64)>, // (job_id, remaining_when_queued)
}

impl FcfsQueue {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, job_id: u64, remaining: f64) {
        self.inner.push_back((job_id, remaining));
    }
    pub fn pop(&mut self) -> Option<(u64, f64)> {
        self.inner.pop_front()
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ── SRPT ───────────────────────────────────────────────────────────────────

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
        // For BinaryHeap (max-heap) + Reverse: we want min by remaining.
        self.remaining
            .total_cmp(&other.remaining)
            .then(self.job_id.cmp(&other.job_id))
    }
}

#[derive(Debug, Default)]
pub struct SrptQueue {
    heap: BinaryHeap<std::cmp::Reverse<SrptEntry>>,
}

impl SrptQueue {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, job_id: u64, remaining: f64) {
        self.heap
            .push(std::cmp::Reverse(SrptEntry { remaining, job_id }));
    }
    pub fn pop(&mut self) -> Option<(u64, f64)> {
        self.heap
            .pop()
            .map(|std::cmp::Reverse(e)| (e.job_id, e.remaining))
    }
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

// ── Unified ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum SimQueue {
    Fcfs(FcfsQueue),
    Srpt(SrptQueue),
}

impl SimQueue {
    pub fn fcfs() -> Self {
        SimQueue::Fcfs(FcfsQueue::new())
    }
    pub fn srpt() -> Self {
        SimQueue::Srpt(SrptQueue::new())
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

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        match self {
            SimQueue::Fcfs(q) => q.is_empty(),
            SimQueue::Srpt(q) => q.is_empty(),
        }
    }
}
