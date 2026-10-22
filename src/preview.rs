use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{Rgb, RgbImage};

use crate::volume::Volume;

const TILE: u32 = 256;
const GAP: u32 = 8;
const WINDOW_LO: f32 = -200.0;
const WINDOW_HI: f32 = 300.0;
pub const DEFAULT_CONTACT_SHEET_SIDE: usize = 8;
pub const MAX_CONTACT_SHEET_SIDE: usize = 32;

pub fn write_contact_sheet(
    image: &Volume<f32>,
    seg: &Volume<u8>,
    path: &Path,
    grid_side: usize,
) -> Result<()> {
    if !(1..=MAX_CONTACT_SHEET_SIDE).contains(&grid_side) {
        bail!("preview grid side must be between 1 and {MAX_CONTACT_SHEET_SIDE}");
    }

    let slices = contact_sheet_slices(seg, image.shape[0], grid_side * grid_side);
    let cols = grid_side.min(slices.len().max(1)) as u32;
    let rows = (slices.len() as u32).div_ceil(cols).max(1);
    let width = cols * TILE + (cols + 1) * GAP;
    let height = rows * TILE + (rows + 1) * GAP;
    let mut out = RgbImage::from_pixel(width, height, Rgb([18, 18, 18]));

    for (i, &z) in slices.iter().enumerate() {
        let col = i as u32 % cols;
        let row = i as u32 / cols;
        let x0 = GAP + col * (TILE + GAP);
        let y0 = GAP + row * (TILE + GAP);
        draw_axial_tile(image, seg, z, &mut out, x0, y0);
    }

    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create preview directory {}", parent.display()))?;
    }
    out.save(path)
        .with_context(|| format!("failed to write preview {}", path.display()))
}

fn preview_slices(seg: &Volume<u8>, image_z: usize, n: usize) -> Vec<usize> {
    let (mut lo, mut hi) = (usize::MAX, 0usize);
    let plane = seg.shape[1] * seg.shape[2];
    for z in 0..seg.shape[0] {
        let start = z * plane;
        let end = start + plane;
        if seg.data[start..end].iter().any(|v| *v != 0) {
            lo = lo.min(z);
            hi = hi.max(z + 1);
        }
    }
    if lo == usize::MAX {
        lo = 0;
        hi = seg.shape[0].max(1);
    }
    let lo_img = scale_index(lo, seg.shape[0], image_z);
    let hi_img = scale_index(hi.saturating_sub(1), seg.shape[0], image_z).saturating_add(1);
    evenly_spaced(lo_img, hi_img.min(image_z), n)
}

/// Slices in reading order: superior (head) at the top left, inferior last.
/// Canonical RAS z increases toward the head, so the sampled order is reversed.
fn contact_sheet_slices(seg: &Volume<u8>, image_z: usize, n: usize) -> Vec<usize> {
    let mut slices = preview_slices(seg, image_z, n);
    slices.reverse();
    slices
}

fn evenly_spaced(lo: usize, hi: usize, n: usize) -> Vec<usize> {
    if hi <= lo {
        return vec![lo];
    }
    let len = hi - lo;
    let count = n.min(len).max(1);
    if count == 1 {
        return vec![lo + len / 2];
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(lo + (i * (len - 1) / (count - 1)));
    }
    out.dedup();
    out
}

fn draw_axial_tile(
    image: &Volume<f32>,
    seg: &Volume<u8>,
    z: usize,
    out: &mut RgbImage,
    x0: u32,
    y0: u32,
) {
    // Volumes arrive in canonical RAS order, so both in-plane axes are mirrored
    // for radiological display: anterior at the top, patient right at image left.
    for ty in 0..TILE {
        let y = scale_index((TILE - 1 - ty) as usize, TILE as usize, image.shape[1]);
        let sy = scale_index(y, image.shape[1], seg.shape[1]);
        for tx in 0..TILE {
            let x = scale_index((TILE - 1 - tx) as usize, TILE as usize, image.shape[2]);
            let sx = scale_index(x, image.shape[2], seg.shape[2]);
            let sz = scale_index(z, image.shape[0], seg.shape[0]);
            let hu = *image.get(z.min(image.shape[0] - 1), y, x);
            let base = window_u8(hu);
            let label = *seg.get(sz, sy, sx);
            let rgb = if label == 0 {
                [base, base, base]
            } else {
                blend([base, base, base], label_color(label), 0.45)
            };
            out.put_pixel(x0 + tx, y0 + ty, Rgb(rgb));
        }
    }
}

fn window_u8(v: f32) -> u8 {
    let t = ((v - WINDOW_LO) / (WINDOW_HI - WINDOW_LO)).clamp(0.0, 1.0);
    (t * 255.0).round() as u8
}

fn blend(bg: [u8; 3], fg: [u8; 3], alpha: f32) -> [u8; 3] {
    [
        ((bg[0] as f32 * (1.0 - alpha)) + (fg[0] as f32 * alpha)).round() as u8,
        ((bg[1] as f32 * (1.0 - alpha)) + (fg[1] as f32 * alpha)).round() as u8,
        ((bg[2] as f32 * (1.0 - alpha)) + (fg[2] as f32 * alpha)).round() as u8,
    ]
}

fn label_color(label: u8) -> [u8; 3] {
    const COLORS: [[u8; 3]; 16] = [
        [230, 57, 70],
        [69, 123, 157],
        [42, 157, 143],
        [244, 162, 97],
        [131, 56, 236],
        [255, 183, 3],
        [33, 158, 188],
        [251, 86, 7],
        [46, 196, 182],
        [255, 0, 110],
        [142, 202, 230],
        [154, 205, 50],
        [255, 127, 80],
        [123, 44, 191],
        [0, 180, 216],
        [255, 214, 10],
    ];
    COLORS[label as usize % COLORS.len()]
}

fn scale_index(i: usize, from: usize, to: usize) -> usize {
    if to <= 1 || from <= 1 {
        return 0;
    }
    ((i as f64) * ((to - 1) as f64) / ((from - 1) as f64)).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn foreground_seg(z: usize, start: usize, end: usize) -> Volume<u8> {
        let mut seg = Volume::zeros([z, 2, 2]);
        for zi in start..end {
            *seg.get_mut(zi, 0, 0) = 1;
        }
        seg
    }

    #[test]
    fn segmentation_tile_is_radiological() {
        // Canonical RAS volume: the last y index is most anterior and the last
        // x index is most toward the patient's right. Both belong at the top left.
        let mut image = Volume::<f32>::zeros([1, 4, 4]);
        for v in image.data.iter_mut() {
            *v = -1000.0;
        }
        *image.get_mut(0, 3, 3) = 300.0;
        let mut seg = Volume::<u8>::zeros([1, 4, 4]);
        *seg.get_mut(0, 0, 0) = 1; // posterior, patient left: bottom right
        let mut out = RgbImage::new(TILE, TILE);
        draw_axial_tile(&image, &seg, 0, &mut out, 0, 0);
        assert_eq!(out.get_pixel(0, 0).0, [255, 255, 255]);
        assert_ne!(out.get_pixel(TILE - 1, TILE - 1).0, [0, 0, 0]);
        assert_eq!(out.get_pixel(TILE - 1, 0).0, [0, 0, 0]);
    }

    #[test]
    fn preview_slices_include_every_label_slice_when_under_limit() {
        let seg = foreground_seg(10, 2, 8);
        assert_eq!(preview_slices(&seg, 10, 64), vec![2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn preview_slices_sample_evenly_across_label_range() {
        let seg = foreground_seg(10, 2, 8);
        assert_eq!(preview_slices(&seg, 10, 4), vec![2, 3, 5, 7]);
    }

    #[test]
    fn contact_sheet_slices_display_superior_to_inferior() {
        let seg = foreground_seg(10, 2, 8);
        assert_eq!(contact_sheet_slices(&seg, 10, 4), vec![7, 5, 3, 2]);
    }

    #[test]
    fn preview_slices_use_full_image_when_segmentation_is_empty() {
        let seg = Volume::<u8>::zeros([10, 2, 2]);
        assert_eq!(preview_slices(&seg, 10, 4), vec![0, 3, 6, 9]);
    }
}
