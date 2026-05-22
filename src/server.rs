#[derive(Debug)]
pub struct Server {
    id: usize,
    current_job: Option<u64>,
}

impl Server {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            current_job: None,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn is_idle(&self) -> bool {
        self.current_job.is_none()
    }

    pub fn start(&mut self, job_id: u64) {
        debug_assert!(self.is_idle(), "started job on busy server");
        self.current_job = Some(job_id);
    }

    pub fn finish(&mut self) {
        debug_assert!(!self.is_idle(), "finished job on idle server");
        self.current_job = None;
    }
}
