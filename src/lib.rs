pub mod dicom_io;
pub mod io;
pub mod preview;
pub mod report;
pub mod resample;
pub mod runtime;
pub mod server;
pub mod tasks;
pub mod volume;
pub mod windows_ml;

// Compatibility paths for existing library consumers. New integrations should
// use `runtime` and `tasks::totalseg` directly.
pub mod components {
    pub use crate::runtime::components::*;
}

pub mod model {
    pub use crate::runtime::model::*;
}

pub mod registry {
    pub use crate::tasks::totalseg::registry::*;
}

pub mod segmentator {
    pub use crate::runtime::segmentator::*;
    pub use crate::tasks::totalseg::{build_predictor, build_predictor_with_full_cards};
}
