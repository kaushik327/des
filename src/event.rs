use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub enum EventKind {
    Arrival { job_id: u64 },
    Departure { job_id: u64, server_id: usize },
}

#[derive(Debug, Clone)]
pub struct Event {
    pub timestamp: f64,
    pub kind: EventKind,
}

impl Event {
    pub fn new(timestamp: f64, kind: EventKind) -> Self {
        Self { timestamp, kind }
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// NaN-free: f64 timestamps are always finite in a valid simulation
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp
            .partial_cmp(&other.timestamp)
            .unwrap_or(Ordering::Equal)
    }
}
