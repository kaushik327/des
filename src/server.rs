/// A single server, tracking its current job and service epoch.
///
/// The epoch counter invalidates stale departure events after SRPT preemption:
/// each call to `begin` increments it; the corresponding `Departure` event
/// stores the epoch, and `is_current_epoch` rejects any event that no longer matches.
#[derive(Debug, Default)]
pub struct Server {
    current_job: Option<u64>,
    /// Wall-clock time of the scheduled departure; used to compute remaining service time.
    pub departure_time: f64,
    epoch: u64,
}

impl Server {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_idle(&self) -> bool {
        self.current_job.is_none()
    }

    pub fn current_job(&self) -> Option<u64> {
        self.current_job
    }

    /// Start serving `job_id`, recording when its departure is expected.
    /// Returns the new epoch (stored in the `Departure` event for validation).
    pub fn begin(&mut self, job_id: u64, departure_at: f64) -> u64 {
        debug_assert!(self.is_idle(), "started job on busy server");
        self.current_job = Some(job_id);
        self.departure_time = departure_at;
        self.epoch += 1;
        self.epoch
    }

    /// Mark server as busy without scheduling a departure (used when no service dist is configured).
    pub fn start_bare(&mut self, job_id: u64) {
        debug_assert!(self.is_idle(), "started job on busy server");
        self.current_job = Some(job_id);
    }

    pub fn finish(&mut self) {
        debug_assert!(!self.is_idle(), "finished job on idle server");
        self.current_job = None;
    }

    pub fn is_current_epoch(&self, epoch: u64) -> bool {
        epoch == self.epoch
    }
}
