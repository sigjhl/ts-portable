//! Generic model-card inference runtime shared by all segmentation tasks.

pub mod components;
pub mod model;
pub mod segmentator;

pub use components::{Detection, DetectionSummary, filter_components};
pub use model::{Inference, ModelCard, Normalization, Postprocessing, ResamplingDataKwargs};
pub use segmentator::{
    CropEmptyBehavior, LocalizerCropOptions, MultiModelSegmentator, PredictionOutput, Predictor,
    RuntimeOptions, Segmentator, build_predictor_from_card, crop_bbox_from_localizer_mask,
    crop_with_localizer, filter_roi, provider_plan_uses_gpu, read_raw_f32_volume,
    resolve_provider_names, write_raw_u8_volume,
};
