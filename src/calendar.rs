use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::event::Event;

#[derive(Debug, Default)]
pub struct EventCalendar {
    heap: BinaryHeap<Reverse<Event>>,
}

impl EventCalendar {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event: Event) {
        self.heap.push(Reverse(event));
    }

    pub fn pop_next(&mut self) -> Option<Event> {
        self.heap.pop().map(|Reverse(e)| e)
    }

    #[allow(dead_code)] // needed for finite-buffer drop logic and stats collection
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    #[allow(dead_code)] // needed for finite-buffer drop logic and stats collection
    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventKind;

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
        let mut cal = EventCalendar::new();
        assert!(cal.pop_next().is_none());
    }

    #[test]
    fn handles_equal_timestamps() {
        let mut cal = EventCalendar::new();
        cal.push(arrival(1.0, 1));
        cal.push(arrival(1.0, 2));
        // Both should pop without panicking; order between ties is unspecified.
        assert_eq!(cal.pop_next().unwrap().timestamp, 1.0);
        assert_eq!(cal.pop_next().unwrap().timestamp, 1.0);
    }
}
