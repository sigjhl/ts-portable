use std::collections::HashMap;

use serde_json::{Value, json};

use crate::{resample::change_spacing_to_shape, volume::Volume};

pub fn compute_statistics(
    image_hu: &Volume<f32>,
    seg: &Volume<u8>,
    voxel_ml: f64,
    id_to_name: &HashMap<u8, String>,
    with_intensity: bool,
) -> Value {
    let seg_int = if with_intensity && seg.shape != image_hu.shape {
        Some(change_spacing_to_shape(seg, image_hu.shape, 0))
    } else {
        None
    };
    let intensity_seg = seg_int.as_ref().unwrap_or(seg);
    let mut counts = vec![0usize; 256];
    for &v in &seg.data {
        counts[v as usize] += 1;
    }
    let mut structures = serde_json::Map::new();
    let mut ids: Vec<_> = id_to_name.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        let n = counts[id as usize];
        if n == 0 {
            continue;
        }
        let mut entry = serde_json::Map::new();
        entry.insert("label".to_string(), json!(id));
        entry.insert("voxel_count".to_string(), json!(n));
        entry.insert(
            "volume_ml".to_string(),
            json!(round_to(n as f64 * voxel_ml, 3)),
        );
        if with_intensity {
            let mut vals = Vec::new();
            for (i, &label) in intensity_seg.data.iter().enumerate() {
                if label == id && i < image_hu.data.len() {
                    vals.push(image_hu.data[i] as f64);
                }
            }
            if !vals.is_empty() {
                vals.sort_by(|a, b| a.total_cmp(b));
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let median = if vals.len() % 2 == 1 {
                    vals[vals.len() / 2]
                } else {
                    (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) * 0.5
                };
                entry.insert("intensity_mean".to_string(), json!(round_to(mean, 2)));
                entry.insert("intensity_median".to_string(), json!(round_to(median, 2)));
            }
        }
        if let Some(name) = id_to_name.get(&id) {
            structures.insert(name.clone(), Value::Object(entry));
        }
    }
    json!({
        "voxel_ml": round_to(voxel_ml, 6),
        "n_structures": structures.len(),
        "structures": structures,
    })
}

fn round_to(v: f64, places: i32) -> f64 {
    let m = 10_f64.powi(places);
    (v * m).round() / m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistics_report_counts_volume_and_intensity_per_structure() {
        let image = Volume {
            data: vec![0.0, 10.0, 20.0, 30.0],
            shape: [1, 1, 4],
        };
        let seg = Volume {
            data: vec![0u8, 1, 1, 2],
            shape: [1, 1, 4],
        };
        let names = HashMap::from([
            (1u8, "spleen".to_string()),
            (2u8, "liver".to_string()),
            (3u8, "absent".to_string()),
        ]);
        let v = compute_statistics(&image, &seg, 0.5, &names, true);
        assert_eq!(v["voxel_ml"], 0.5);
        assert_eq!(v["n_structures"], 2);
        let spleen = &v["structures"]["spleen"];
        assert_eq!(spleen["label"], 1);
        assert_eq!(spleen["voxel_count"], 2);
        assert_eq!(spleen["volume_ml"], 1.0);
        assert_eq!(spleen["intensity_mean"], 15.0);
        assert_eq!(spleen["intensity_median"], 15.0);
        assert_eq!(v["structures"]["liver"]["voxel_count"], 1);
        assert!(v["structures"].get("absent").is_none());
    }
}
