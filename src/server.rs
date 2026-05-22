#[derive(Debug)]
pub struct Server {
    id: usize,
    current_job: Option<u64>,
    /// Simulated time at which the current departure is scheduled; used to compute remaining service time.
    pub departure_time: f64,
    epoch: u64,
}

impl Server {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            current_job: None,
            departure_time: 0.0,
            epoch: 0,
        }
    }

    #[allow(dead_code)]
    pub fn id(&self) -> usize {
        self.id
    }
    pub fn is_idle(&self) -> bool {
        self.current_job.is_none()
    }
    pub fn current_job(&self) -> Option<u64> {
        self.current_job
    }

    /// Start serving `job_id`, recording when its departure is expected.
    /// Returns the new epoch (stored in the Departure event for validation).
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

    /// Finish the current job (normal completion).
    pub fn finish(&mut self) {
        debug_assert!(!self.is_idle(), "finished job on idle server");
        self.current_job = None;
    }

    /// Return true only for the departure event that was most recently scheduled.
    /// A false result means the event is stale due to preemption.
    pub fn is_current_epoch(&self, epoch: u64) -> bool {
        epoch == self.epoch
    }
}
