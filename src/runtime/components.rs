use anyhow::{Result, bail};
use serde::Serialize;

use crate::volume::Volume;

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub id: usize,
    pub label: u8,
    pub voxel_count: usize,
    pub volume_mm3: f64,
    pub bbox_zyx: [[usize; 2]; 3],
}

#[derive(Clone, Debug, Serialize)]
pub struct DetectionSummary {
    pub coordinate_space: &'static str,
    pub voxel_spacing_zyx_mm: [f32; 3],
    pub min_component_volume_mm3: f64,
    pub connectivity: u8,
    pub detections: Vec<Detection>,
}

pub fn filter_components(
    seg: &mut Volume<u8>,
    spacing_zyx: [f32; 3],
    min_volume_mm3: f64,
    connectivity: u8,
) -> Result<DetectionSummary> {
    if !matches!(connectivity, 6 | 18 | 26) {
        bail!("component connectivity must be 6, 18, or 26");
    }
    if !min_volume_mm3.is_finite() || min_volume_mm3 < 0.0 {
        bail!("minimum component volume must be finite and non-negative");
    }
    let voxel_volume_mm3 = spacing_zyx.iter().map(|v| *v as f64).product::<f64>();
    let shape = seg.shape;
    let plane = shape[1] * shape[2];
    let mut visited = vec![false; seg.data.len()];
    let mut detections = Vec::new();

    for seed in 0..seg.data.len() {
        let label = seg.data[seed];
        if label == 0 || visited[seed] {
            continue;
        }
        visited[seed] = true;
        let mut queue = vec![seed];
        let mut cursor = 0usize;
        let mut bbox = [[usize::MAX, 0usize]; 3];
        while cursor < queue.len() {
            let index = queue[cursor];
            cursor += 1;
            let z = index / plane;
            let rem = index % plane;
            let y = rem / shape[2];
            let x = rem % shape[2];
            for (axis, value) in [z, y, x].into_iter().enumerate() {
                bbox[axis][0] = bbox[axis][0].min(value);
                bbox[axis][1] = bbox[axis][1].max(value + 1);
            }

            for dz in -1isize..=1 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let changed =
                            usize::from(dz != 0) + usize::from(dy != 0) + usize::from(dx != 0);
                        if changed == 0
                            || (connectivity == 6 && changed != 1)
                            || (connectivity == 18 && changed == 3)
                        {
                            continue;
                        }
                        let nz = z as isize + dz;
                        let ny = y as isize + dy;
                        let nx = x as isize + dx;
                        if nz < 0
                            || ny < 0
                            || nx < 0
                            || nz >= shape[0] as isize
                            || ny >= shape[1] as isize
                            || nx >= shape[2] as isize
                        {
                            continue;
                        }
                        let neighbor = nz as usize * plane + ny as usize * shape[2] + nx as usize;
                        if !visited[neighbor] && seg.data[neighbor] == label {
                            visited[neighbor] = true;
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }

        let volume_mm3 = queue.len() as f64 * voxel_volume_mm3;
        if volume_mm3 < min_volume_mm3 {
            for index in queue {
                seg.data[index] = 0;
            }
            continue;
        }
        detections.push(Detection {
            id: detections.len() + 1,
            label,
            voxel_count: queue.len(),
            volume_mm3,
            bbox_zyx: bbox,
        });
    }

    Ok(DetectionSummary {
        coordinate_space: "model-input voxel indices; bbox upper bounds are exclusive",
        voxel_spacing_zyx_mm: spacing_zyx,
        min_component_volume_mm3: min_volume_mm3,
        connectivity,
        detections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_small_components_and_reports_bbox() {
        let mut seg = Volume::from_vec(
            [2, 3, 4],
            vec![
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // z0
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // z1
            ],
        )
        .unwrap();
        let summary = filter_components(&mut seg, [1.0, 1.0, 1.0], 2.0, 6).unwrap();
        assert_eq!(summary.detections.len(), 1);
        assert_eq!(summary.detections[0].voxel_count, 2);
        assert_eq!(summary.detections[0].bbox_zyx, [[0, 1], [0, 1], [0, 2]]);
        assert_eq!(seg.data.iter().filter(|value| **value != 0).count(), 2);
    }

    #[test]
    fn connectivity_controls_diagonal_grouping() {
        let data = vec![1, 0, 0, 1];
        let mut six = Volume::from_vec([1, 2, 2], data.clone()).unwrap();
        let mut twenty_six = Volume::from_vec([1, 2, 2], data).unwrap();
        assert_eq!(
            filter_components(&mut six, [1.0; 3], 0.0, 6)
                .unwrap()
                .detections
                .len(),
            2
        );
        assert_eq!(
            filter_components(&mut twenty_six, [1.0; 3], 0.0, 26)
                .unwrap()
                .detections
                .len(),
            1
        );
    }
}
