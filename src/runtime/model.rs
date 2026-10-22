use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    pub mean: f32,
    pub std: f32,
    pub lower_bound: f32,
    pub upper_bound: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResamplingDataKwargs {
    pub order: usize,
    pub order_z: usize,
    pub force_separate_z: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Inference {
    pub tile_step_size: f32,
    pub use_gaussian: bool,
    pub gaussian_sigma_scale: f32,
    pub use_mirroring: Option<bool>,
    pub mirror_axes: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Postprocessing {
    pub min_component_volume_mm3: Option<f32>,
    pub connectivity: Option<u8>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCard {
    pub name: String,
    pub onnx_file: String,
    pub onnx_files: Option<Vec<String>>,
    pub patch_size: [usize; 3],
    pub target_spacing: [f32; 3],
    pub transpose_forward: [usize; 3],
    pub transpose_backward: [usize; 3],
    pub num_seg_heads: usize,
    pub normalization: Normalization,
    pub resampling_data_kwargs: ResamplingDataKwargs,
    pub inference: Inference,
    pub labels: HashMap<String, u8>,
    pub outer_resample: Option<bool>,
    pub quantize_outer_resample: Option<bool>,
    pub normalize_before_resample: Option<bool>,
    pub preserve_input_orientation: Option<bool>,
    pub postprocessing: Option<Postprocessing>,
}

impl ModelCard {
    pub fn read(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read model card {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn id_to_name(&self) -> HashMap<u8, String> {
        self.labels
            .iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect()
    }
}
