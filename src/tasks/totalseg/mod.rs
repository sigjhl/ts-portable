//! TotalSegmentator-specific model registry and predictor construction.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::runtime::{MultiModelSegmentator, Predictor, RuntimeOptions, Segmentator};

pub mod registry;

pub use crate::runtime::segmentator::crop_cascade;

pub fn build_predictor(
    model: &str,
    models_dir: &Path,
    opts: RuntimeOptions,
    resample_order: Option<usize>,
) -> Result<Predictor> {
    build_predictor_with_full_cards(model, models_dir, opts, resample_order, None)
}

pub fn build_predictor_with_full_cards(
    model: &str,
    models_dir: &Path,
    opts: RuntimeOptions,
    resample_order: Option<usize>,
    full_cards: Option<Vec<PathBuf>>,
) -> Result<Predictor> {
    if model == "full" {
        Ok(Predictor::Multi(Box::new(MultiModelSegmentator::new(
            full_cards.unwrap_or_else(|| registry::full_cards(models_dir)),
            &registry::label_map_path(models_dir),
            opts,
        )?)))
    } else {
        Ok(Predictor::Single(Box::new(Segmentator::new(
            &registry::resolve_card(models_dir, model)?,
            &opts,
            resample_order,
        )?)))
    }
}
