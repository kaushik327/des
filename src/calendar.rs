//! Event calendar: the sorted priority queue at the heart of the DES engine.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

// ── Sim-specific event types ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum EventKind {
    Arrival {
        job_id: u64,
    },
    Departure {
        job_id: u64,
        server_id: usize,
        epoch: u64,
    },
}

/// A timestamped event in the simulation.
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

// Timestamps are always finite in a valid simulation.
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp
            .partial_cmp(&other.timestamp)
            .unwrap_or(Ordering::Equal)
    }
}

// ── Generic calendar ─────────────────────────────────────────────────────────

/// Min-heap of future events ordered by `E`'s `Ord` implementation.
///
/// Any event type that is totally ordered by timestamp can be used; the
/// built-in [`Event`] is for the single-node simulator, but domain-specific
/// schedulers (e.g. `llm`) define and use their own event types.
#[derive(Debug)]
pub struct EventCalendar<E: Ord> {
    heap: BinaryHeap<Reverse<E>>,
}

impl<E: Ord> Default for EventCalendar<E> {
    fn default() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }
}

impl<E: Ord> EventCalendar<E> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event: E) {
        self.heap.push(Reverse(event));
    }

    /// Removes and returns the earliest event, or `None` if the calendar is empty.
    pub fn pop_next(&mut self) -> Option<E> {
        self.heap.pop().map(|Reverse(e)| e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrival(t: f64, id: u64) -> Event {
        Event::new(t, EventKind::Arrival { job_id: id })
    }

    #[test]
    fn pops_in_timestamp_order() {
        let mut cal = EventCalendar::new();
        cal.push(arrival(3.0, 3));
        cal.push(arrival(1.0, 1));
        cal.push(arrival(2.0, 2));
        assert_eq!(cal.pop_next().unwrap().timestamp, 1.0);
        assert_eq!(cal.pop_next().unwrap().timestamp, 2.0);
        assert_eq!(cal.pop_next().unwrap().timestamp, 3.0);
        assert!(cal.pop_next().is_none());
    }

    #[test]
    fn empty_calendar_returns_none() {
        let mut cal: EventCalendar<Event> = EventCalendar::new();
        assert!(cal.pop_next().is_none());
    }

    #[test]
    fn handles_equal_timestamps() {
        let mut cal = EventCalendar::new();
        cal.push(arrival(1.0, 1));
        cal.push(arrival(1.0, 2));
        assert_eq!(cal.pop_next().unwrap().timestamp, 1.0);
        assert_eq!(cal.pop_next().unwrap().timestamp, 1.0);
    }
}
