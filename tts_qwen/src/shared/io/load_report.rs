#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    pub applied: usize,
    pub skipped: usize,
    pub missing: usize,
    pub unused: usize,
}
