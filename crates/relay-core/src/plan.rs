use serde::{Deserialize, Serialize};

use crate::model::DeliveryMode;

/// Execution plan produced by the Understand Agent; also the data source for the
/// user confirmation card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub title: String,
    pub summary: String,
    pub steps: Vec<String>,
    /// Reference materials / context clues that were matched.
    pub references: Vec<String>,
    pub delivery: DeliveryMode,
    /// Raw model output, kept for auditing.
    pub raw: String,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            title: "Untitled task".into(),
            summary: String::new(),
            steps: Vec::new(),
            references: Vec::new(),
            delivery: DeliveryMode::OpenFile,
            raw: String::new(),
        }
    }
}

impl Plan {
    pub fn summary_line(&self) -> String {
        if self.steps.is_empty() {
            self.summary.clone()
        } else {
            format!("{} ({} steps)", self.summary, self.steps.len())
        }
    }
}
