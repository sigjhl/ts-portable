use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};

use crate::runtime::model::ModelCard;

const FAST: &str = "total_3mm_297";
const FASTEST: &str = "total_6mm_298";
const LIVER_VESSELS: &str = "liver_vessels_008";
const LIVER_LESIONS: &str = "liver_lesions_591";
const LIVER_SEGMENTS: &str = "liver_segments_570";
const ABDOMINAL_MUSCLES: &str = "abdominal_muscles_952";
const KIDNEY_CYSTS: &str = "kidney_cysts_789";
const BODY: &str = "body_6mm_300";
const FULL_PARTS: [&str; 5] = [
    "total_1p5mm_291_organs",
    "total_1p5mm_292_vertebrae",
    "total_1p5mm_293_cardiac",
    "total_1p5mm_294_muscles",
    "total_1p5mm_295_ribs",
];

#[derive(Clone, Copy, Debug)]
pub struct CropSpec {
    pub localizer_model: &'static str,
    pub roi_names: &'static [&'static str],
    pub margin_mm: f32,
    pub empty_is_empty: bool,
    pub localizer_low_memory_cpu: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct TaskModel {
    pub model: &'static str,
    pub resample_order: Option<usize>,
    pub lowres: bool,
    pub crop: Option<CropSpec>,
    pub output_roi_names: Option<&'static [&'static str]>,
}

pub fn models_dir(cli: Option<&Path>) -> PathBuf {
    if let Some(path) = cli {
        return path.to_path_buf();
    }
    if let Ok(path) = env::var("ONNX_TOTALSEG_MODELS") {
        return PathBuf::from(path);
    }
    PathBuf::from("models")
}

pub fn resolve_card(models_dir: &Path, model: &str) -> Result<PathBuf> {
    if let Ok(path) = env::var("ONNX_TOTALSEG_CARD") {
        return Ok(PathBuf::from(path));
    }
    let tag = match model {
        "fast" => FAST,
        "fastest" => FASTEST,
        "liver_vessels" => LIVER_VESSELS,
        "liver_lesions" => LIVER_LESIONS,
        "liver_segments" => LIVER_SEGMENTS,
        "abdominal_muscles" => ABDOMINAL_MUSCLES,
        "kidney_cysts" => KIDNEY_CYSTS,
        "body" => BODY,
        other => bail!(
            "unknown single model '{other}', expected fast, fastest, liver_vessels, liver_lesions, liver_segments, abdominal_muscles, kidney_cysts, or body"
        ),
    };
    Ok(models_dir.join(format!("{tag}.model_card.json")))
}

pub fn resolve_task_model(task: &str, tier: &str) -> Result<TaskModel> {
    match task {
        "total" => match tier {
            "full" => Ok(TaskModel {
                model: "full",
                resample_order: None,
                lowres: false,
                crop: None,
                output_roi_names: None,
            }),
            "fast" => Ok(TaskModel {
                model: "fast",
                resample_order: None,
                lowres: false,
                crop: None,
                output_roi_names: None,
            }),
            "fastest" => Ok(TaskModel {
                model: "fastest",
                resample_order: None,
                lowres: false,
                crop: None,
                output_roi_names: None,
            }),
            "ultrafast" => Ok(TaskModel {
                model: "fastest",
                resample_order: Some(1),
                lowres: true,
                crop: None,
                output_roi_names: None,
            }),
            _ => bail!("unknown tier '{tier}' (full|fast|fastest|ultrafast)"),
        },
        "liver_vessels" => single_task_model(
            task,
            tier,
            "liver_vessels",
            CropSpec {
                localizer_model: "fastest",
                roi_names: &["liver"],
                margin_mm: 20.0,
                empty_is_empty: true,
                localizer_low_memory_cpu: false,
            },
            None,
        ),
        "liver_lesions" => single_task_model(
            task,
            tier,
            "liver_lesions",
            CropSpec {
                localizer_model: "fast",
                roi_names: &["liver"],
                margin_mm: 20.0,
                empty_is_empty: true,
                localizer_low_memory_cpu: true,
            },
            None,
        ),
        "liver_segments" => single_task_model(
            task,
            tier,
            "liver_segments",
            CropSpec {
                localizer_model: "fastest",
                roi_names: &["liver"],
                margin_mm: 20.0,
                empty_is_empty: true,
                localizer_low_memory_cpu: false,
            },
            None,
        ),
        "abdominal_muscles" => single_task_model(
            task,
            tier,
            "abdominal_muscles",
            CropSpec {
                localizer_model: "body",
                roi_names: &["body_trunc"],
                margin_mm: 20.0,
                empty_is_empty: true,
                localizer_low_memory_cpu: false,
            },
            None,
        ),
        "kidney_cysts" => single_task_model(
            task,
            tier,
            "kidney_cysts",
            CropSpec {
                localizer_model: "fastest",
                roi_names: &["kidney_left", "kidney_right", "liver", "spleen", "colon"],
                margin_mm: 20.0,
                empty_is_empty: true,
                localizer_low_memory_cpu: false,
            },
            Some(&["kidney_cyst_left", "kidney_cyst_right"]),
        ),
        other => bail!(
            "task '{other}' is not supported; supported tasks are total, liver_vessels, liver_lesions, liver_segments, abdominal_muscles, kidney_cysts"
        ),
    }
}

pub fn low_memory_cpu_model(model: &str) -> bool {
    matches!(
        model,
        "fastest" | "liver_vessels" | "liver_segments" | "body"
    )
}

fn single_task_model(
    task: &str,
    tier: &str,
    model: &'static str,
    crop: CropSpec,
    output_roi_names: Option<&'static [&'static str]>,
) -> Result<TaskModel> {
    if tier != "full" {
        bail!("task '{task}' does not support --fast, --fastest, or --ultrafast");
    }
    Ok(TaskModel {
        model,
        resample_order: None,
        lowres: false,
        crop: Some(crop),
        output_roi_names,
    })
}

pub fn full_cards(models_dir: &Path) -> Vec<PathBuf> {
    FULL_PARTS
        .iter()
        .map(|tag| models_dir.join(format!("{tag}.model_card.json")))
        .collect()
}

pub fn full_cards_for_roi(models_dir: &Path, roi_ids: &[u8]) -> Result<Vec<PathBuf>> {
    let (_, id_to_name) = total_label_maps(models_dir)?;
    let roi_names = roi_ids
        .iter()
        .filter_map(|id| id_to_name.get(id))
        .cloned()
        .collect::<HashSet<_>>();
    let mut selected = Vec::new();
    for path in full_cards(models_dir) {
        let card = ModelCard::read(&path)?;
        if card.labels.keys().any(|name| roi_names.contains(name)) {
            selected.push(path);
        }
    }
    if selected.is_empty() {
        bail!("no full model part contains ROI label ids {roi_ids:?}");
    }
    Ok(selected)
}

pub fn label_map_path(models_dir: &Path) -> PathBuf {
    models_dir.join("total_label_map.json")
}

pub fn total_label_maps(models_dir: &Path) -> Result<(HashMap<String, u8>, HashMap<u8, String>)> {
    let path = label_map_path(models_dir);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read label map {}", path.display()))?;
    let raw: HashMap<String, u8> = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let id_to_name = raw
        .iter()
        .filter(|(name, _)| name.as_str() != "background")
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    Ok((raw, id_to_name))
}

pub fn model_label_maps(
    models_dir: &Path,
    model: &str,
) -> Result<(HashMap<String, u8>, HashMap<u8, String>)> {
    if model == "full" {
        return total_label_maps(models_dir);
    }
    let card = ModelCard::read(&resolve_card(models_dir, model)?)?;
    let name_to_id = card.labels;
    let id_to_name = name_to_id
        .iter()
        .filter(|(name, _)| name.as_str() != "background")
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    Ok((name_to_id, id_to_name))
}

pub fn resolve_roi_for_model(models_dir: &Path, model: &str, roi: &str) -> Result<Vec<u8>> {
    let (name_to_id, _) = model_label_maps(models_dir, model)?;
    let mut out = Vec::new();
    let mut unknown = Vec::new();
    for name in roi.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(id) = name_to_id.get(name) {
            out.push(*id);
        } else {
            unknown.push(name.to_string());
        }
    }
    if !unknown.is_empty() {
        bail!("unknown ROI structure(s): {unknown:?}");
    }
    Ok(out)
}

pub fn resolve_labels_for_model(
    models_dir: &Path,
    model: &str,
    labels: &[&str],
) -> Result<Vec<u8>> {
    let (name_to_id, _) = model_label_maps(models_dir, model)?;
    labels
        .iter()
        .map(|name| {
            name_to_id
                .get(*name)
                .copied()
                .with_context(|| format!("unknown label '{name}' for model '{model}'"))
        })
        .collect()
}
