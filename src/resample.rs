use crate::volume::Volume;
use rayon::prelude::*;

const ANISO_THRESHOLD: f32 = 3.0;

pub fn compute_new_shape(
    old_shape: [usize; 3],
    old_spacing: [f32; 3],
    new_spacing: [f32; 3],
) -> [usize; 3] {
    [
        py_round_usize(old_spacing[0] / new_spacing[0] * old_shape[0] as f32),
        py_round_usize(old_spacing[1] / new_spacing[1] * old_shape[1] as f32),
        py_round_usize(old_spacing[2] / new_spacing[2] * old_shape[2] as f32),
    ]
}

pub fn change_spacing_to_spacing(
    data: &Volume<f32>,
    current_spacing: [f32; 3],
    target_spacing: [f32; 3],
    order: usize,
) -> Volume<f32> {
    let zoom = [
        current_spacing[0] / target_spacing[0],
        current_spacing[1] / target_spacing[1],
        current_spacing[2] / target_spacing[2],
    ];
    let target = [
        py_round_usize(data.shape[0] as f32 * zoom[0]).max(1),
        py_round_usize(data.shape[1] as f32 * zoom[1]).max(1),
        py_round_usize(data.shape[2] as f32 * zoom[2]).max(1),
    ];
    if target == data.shape {
        return data.clone();
    }
    zoom_to_shape(data, target, order, CoordMode::ScipyZoom)
}

fn py_round_usize(v: f32) -> usize {
    let floor = v.floor();
    let frac = v - floor;
    let rounded = if (frac - 0.5).abs() < 1e-6 {
        let floor_i = floor as usize;
        if floor_i.is_multiple_of(2) {
            floor
        } else {
            floor + 1.0
        }
    } else {
        v.round()
    };
    rounded.max(0.0) as usize
}

fn nearest_index(v: f32, len: usize) -> usize {
    clamp_index(v.round() as isize, len)
}

pub fn change_spacing_to_shape<T: Copy + Default + Into<f32> + FromSample + Send + Sync>(
    data: &Volume<T>,
    target_shape: [usize; 3],
    order: usize,
) -> Volume<T> {
    if target_shape == data.shape {
        return data.clone();
    }
    zoom_to_shape_generic(data, target_shape, order, CoordMode::ScipyZoom)
}

pub fn resize_to_shape(data: &Volume<f32>, target_shape: [usize; 3], order: usize) -> Volume<f32> {
    if target_shape == data.shape {
        return data.clone();
    }
    zoom_to_shape(data, target_shape, order, CoordMode::SkimageResize)
}

pub fn resize_labels_to_shape(data: &Volume<u8>, target_shape: [usize; 3]) -> Volume<u8> {
    if target_shape == data.shape {
        return data.clone();
    }
    zoom_to_shape_generic(data, target_shape, 0, CoordMode::SkimageResize)
}

pub fn resample_data_or_seg_to_shape(
    data: &Volume<f32>,
    new_shape: [usize; 3],
    current_spacing: [f32; 3],
    new_spacing: [f32; 3],
    order: usize,
    order_z: usize,
    force_separate_z: Option<bool>,
) -> Volume<f32> {
    let (do_sep, axis) =
        determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing);
    if !do_sep {
        return resize_to_shape(data, new_shape, order);
    }
    let axis = axis.unwrap_or(0);
    resize_separate_z(data, new_shape, axis, order, order_z)
}

fn determine_do_sep_z_and_axis(
    force_separate_z: Option<bool>,
    current_spacing: [f32; 3],
    new_spacing: [f32; 3],
) -> (bool, Option<usize>) {
    if let Some(force) = force_separate_z {
        if force {
            return lowres_axis(current_spacing).map_or((false, None), |a| (true, Some(a)));
        }
        return (false, None);
    }
    if is_aniso(current_spacing) {
        return lowres_axis(current_spacing).map_or((false, None), |a| (true, Some(a)));
    }
    if is_aniso(new_spacing) {
        return lowres_axis(new_spacing).map_or((false, None), |a| (true, Some(a)));
    }
    (false, None)
}

fn is_aniso(spacing: [f32; 3]) -> bool {
    let max = spacing.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = spacing.iter().copied().fold(f32::INFINITY, f32::min);
    max / min > ANISO_THRESHOLD
}

fn lowres_axis(spacing: [f32; 3]) -> Option<usize> {
    let max = spacing.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let axes: Vec<_> = spacing
        .iter()
        .enumerate()
        .filter_map(|(i, s)| ((*s - max).abs() < 1e-5).then_some(i))
        .collect();
    (axes.len() == 1).then_some(axes[0])
}

fn resize_separate_z(
    data: &Volume<f32>,
    new_shape: [usize; 3],
    axis: usize,
    order: usize,
    order_z: usize,
) -> Volume<f32> {
    let mut intermediate_shape = new_shape;
    intermediate_shape[axis] = data.shape[axis];
    let tmp = if order == 3 {
        spline_cubic_resize(data, intermediate_shape, CoordMode::SkimageResize)
    } else {
        let mut tmp = Volume::<f32>::zeros(intermediate_shape);
        let plane = intermediate_shape[1] * intermediate_shape[2];

        match axis {
            0 => {
                tmp.data
                    .par_chunks_mut(plane)
                    .enumerate()
                    .for_each(|(z, out_plane)| {
                        debug_assert!(z < data.shape[0]);
                        let nx = new_shape[2];
                        let ny = new_shape[1];
                        for y in 0..ny {
                            let sy = skimage_coord(y, data.shape[1], ny);
                            for x in 0..nx {
                                let sx = skimage_coord(x, data.shape[2], nx);
                                out_plane[y * nx + x] = sample2(data, z, sy, sx, order);
                            }
                        }
                    });
            }
            1 => {
                tmp.data
                    .par_chunks_mut(plane)
                    .enumerate()
                    .for_each(|(z, out_plane)| {
                        let sz = skimage_coord(z, data.shape[0], new_shape[0]);
                        let nx = new_shape[2];
                        for y in 0..data.shape[1] {
                            for x in 0..nx {
                                let sx = skimage_coord(x, data.shape[2], nx);
                                out_plane[y * nx + x] = sample2_zx(data, y, sz, sx, order);
                            }
                        }
                    });
            }
            _ => {
                tmp.data
                    .par_chunks_mut(plane)
                    .enumerate()
                    .for_each(|(z, out_plane)| {
                        let sz = skimage_coord(z, data.shape[0], new_shape[0]);
                        let nx = data.shape[2];
                        for y in 0..new_shape[1] {
                            let sy = skimage_coord(y, data.shape[1], new_shape[1]);
                            for x in 0..nx {
                                out_plane[y * nx + x] = sample2_zy(data, x, sz, sy, order);
                            }
                        }
                    });
            }
        }
        tmp
    };
    if intermediate_shape == new_shape {
        tmp
    } else {
        zoom_to_shape(&tmp, new_shape, order_z, CoordMode::SkimageResize)
    }
}

#[derive(Clone, Copy)]
enum CoordMode {
    ScipyZoom,
    SkimageResize,
}

pub trait FromSample {
    fn from_sample(v: f32) -> Self;
}

impl FromSample for f32 {
    fn from_sample(v: f32) -> Self {
        v
    }
}

impl FromSample for u8 {
    fn from_sample(v: f32) -> Self {
        v.round().clamp(0.0, 255.0) as u8
    }
}

fn zoom_to_shape(
    data: &Volume<f32>,
    target_shape: [usize; 3],
    order: usize,
    mode: CoordMode,
) -> Volume<f32> {
    if order == 3 && matches!(mode, CoordMode::ScipyZoom) {
        return spline_cubic_resize(data, target_shape, CoordMode::ScipyZoom);
    }
    if order == 3 && matches!(mode, CoordMode::SkimageResize) {
        return spline_cubic_resize(data, target_shape, CoordMode::SkimageResize);
    }
    zoom_to_shape_generic(data, target_shape, order, mode)
}

fn spline_cubic_resize(
    data: &Volume<f32>,
    target_shape: [usize; 3],
    mode: CoordMode,
) -> Volume<f32> {
    if data.shape == target_shape {
        return data.clone();
    }
    let pad = 12usize;
    let padded_shape = [
        data.shape[0] + 2 * pad,
        data.shape[1] + 2 * pad,
        data.shape[2] + 2 * pad,
    ];
    let mut coeff = Volume::<f64>::zeros(padded_shape);
    let coeff_plane = padded_shape[1] * padded_shape[2];
    coeff
        .data
        .par_chunks_mut(coeff_plane)
        .enumerate()
        .for_each(|(z, plane)| {
            let src_z = z.saturating_sub(pad).min(data.shape[0] - 1);
            for y in 0..padded_shape[1] {
                let src_y = y.saturating_sub(pad).min(data.shape[1] - 1);
                for x in 0..padded_shape[2] {
                    let src_x = x.saturating_sub(pad).min(data.shape[2] - 1);
                    plane[y * padded_shape[2] + x] = *data.get(src_z, src_y, src_x) as f64;
                }
            }
        });
    spline_filter_axis(&mut coeff, 0);
    spline_filter_axis(&mut coeff, 1);
    spline_filter_axis(&mut coeff, 2);

    let z_samples = cubic_axis_samples(data.shape[0], target_shape[0], pad, padded_shape[0], mode);
    let y_samples = cubic_axis_samples(data.shape[1], target_shape[1], pad, padded_shape[1], mode);
    let x_samples = cubic_axis_samples(data.shape[2], target_shape[2], pad, padded_shape[2], mode);
    let mut out = Volume::<f32>::zeros(target_shape);
    let out_plane = target_shape[1] * target_shape[2];
    out.data
        .par_chunks_mut(out_plane)
        .enumerate()
        .for_each(|(z, plane)| {
            let z_sample = z_samples[z];
            for y in 0..target_shape[1] {
                let y_sample = y_samples[y];
                for x in 0..target_shape[2] {
                    plane[y * target_shape[2] + x] =
                        cubic_bspline_sample(&coeff, &z_sample, &y_sample, &x_samples[x]) as f32;
                }
            }
        });
    if matches!(mode, CoordMode::SkimageResize) {
        let (min_v, max_v) = data
            .data
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min_v, max_v), &v| {
                (min_v.min(v), max_v.max(v))
            });
        out.data
            .par_iter_mut()
            .for_each(|v| *v = v.clamp(min_v, max_v));
    }
    out
}

fn scipy_zoom_coord(i: usize, old: usize, new: usize) -> f64 {
    if new <= 1 {
        0.0
    } else {
        (i as f64) * (old.saturating_sub(1) as f64) / (new.saturating_sub(1) as f64)
    }
}

fn spline_filter_axis(vol: &mut Volume<f64>, axis: usize) {
    let nz = vol.shape[0];
    let ny = vol.shape[1];
    let nx = vol.shape[2];
    match axis {
        0 => {
            let src = &vol.data;
            let mut lines = vec![0.0; vol.data.len()];
            lines.par_chunks_mut(nz).enumerate().for_each_init(
                Vec::new,
                |diag, (line_idx, line)| {
                    let y = line_idx / nx;
                    let x = line_idx % nx;
                    for z in 0..nz {
                        line[z] = src[(z * ny + y) * nx + x];
                    }
                    solve_cubic_spline_coefficients_with_diag(line, diag);
                },
            );
            for line_idx in 0..ny * nx {
                let y = line_idx / nx;
                let x = line_idx % nx;
                let line = &lines[line_idx * nz..(line_idx + 1) * nz];
                for (z, &v) in line.iter().enumerate() {
                    vol.data[(z * ny + y) * nx + x] = v;
                }
            }
        }
        1 => {
            let plane = ny * nx;
            vol.data.par_chunks_mut(plane).for_each(|plane_data| {
                let mut line = vec![0.0; ny];
                let mut diag = Vec::new();
                for x in 0..nx {
                    for y in 0..ny {
                        line[y] = plane_data[y * nx + x];
                    }
                    solve_cubic_spline_coefficients_with_diag(&mut line, &mut diag);
                    for (y, &v) in line.iter().enumerate() {
                        plane_data[y * nx + x] = v;
                    }
                }
            });
        }
        _ => {
            vol.data
                .par_chunks_mut(nx)
                .for_each_init(Vec::new, |diag, line| {
                    solve_cubic_spline_coefficients_with_diag(line, diag);
                });
        }
    }
}

fn solve_cubic_spline_coefficients_with_diag(line: &mut [f64], diag: &mut Vec<f64>) {
    let n = line.len();
    if n <= 1 {
        return;
    }
    diag.clear();
    diag.resize(n, 4.0);
    diag[0] = 5.0;
    diag[n - 1] = 5.0;
    for v in line.iter_mut() {
        *v *= 6.0;
    }
    for i in 1..n {
        let w = 1.0 / diag[i - 1];
        diag[i] -= w;
        line[i] -= w * line[i - 1];
    }
    line[n - 1] /= diag[n - 1];
    for i in (0..n - 1).rev() {
        line[i] = (line[i] - line[i + 1]) / diag[i];
    }
}

#[derive(Clone, Copy)]
struct CubicAxisSample {
    idx: [usize; 4],
    weights: [f64; 4],
}

fn cubic_axis_samples(
    old: usize,
    new: usize,
    pad: usize,
    len: usize,
    mode: CoordMode,
) -> Vec<CubicAxisSample> {
    (0..new)
        .map(|i| {
            let coord = match mode {
                CoordMode::ScipyZoom => scipy_zoom_coord(i, old, new),
                CoordMode::SkimageResize => skimage_coord(i, old, new) as f64,
            } + pad as f64;
            let base = coord.floor() as isize;
            let weights = cubic_bspline_weights(coord - base as f64);
            let mut idx = [0; 4];
            for (offset, dst) in idx.iter_mut().enumerate() {
                *dst = clamp_index(base + offset as isize - 1, len);
            }
            CubicAxisSample { idx, weights }
        })
        .collect()
}

#[inline]
fn cubic_bspline_sample(
    coeff: &Volume<f64>,
    z: &CubicAxisSample,
    y: &CubicAxisSample,
    x: &CubicAxisSample,
) -> f64 {
    let nx = coeff.shape[2];
    let plane = coeff.shape[1] * nx;
    let data = &coeff.data;
    let mut acc = 0.0;
    for oz in 0..4 {
        let z_base = z.idx[oz] * plane;
        let mut y_acc = 0.0;
        for oy in 0..4 {
            let row = z_base + y.idx[oy] * nx;
            let x_acc = x.weights[0] * data[row + x.idx[0]]
                + x.weights[1] * data[row + x.idx[1]]
                + x.weights[2] * data[row + x.idx[2]]
                + x.weights[3] * data[row + x.idx[3]];
            y_acc += y.weights[oy] * x_acc;
        }
        acc += z.weights[oz] * y_acc;
    }
    acc
}

fn cubic_bspline_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
        (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
        (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
        t3 / 6.0,
    ]
}

fn zoom_to_shape_generic<T: Copy + Default + Into<f32> + FromSample + Send + Sync>(
    data: &Volume<T>,
    target_shape: [usize; 3],
    order: usize,
    mode: CoordMode,
) -> Volume<T> {
    if data.shape == target_shape {
        return data.clone();
    }
    let mut out = Volume::<T>::zeros(target_shape);
    let zoom = [
        target_shape[0] as f32 / data.shape[0] as f32,
        target_shape[1] as f32 / data.shape[1] as f32,
        target_shape[2] as f32 / data.shape[2] as f32,
    ];
    let plane = target_shape[1] * target_shape[2];
    out.data
        .par_chunks_mut(plane)
        .enumerate()
        .for_each(|(z, out_plane)| {
            let sz = coord(z, data.shape[0], target_shape[0], zoom[0], mode);
            for y in 0..target_shape[1] {
                let sy = coord(y, data.shape[1], target_shape[1], zoom[1], mode);
                for x in 0..target_shape[2] {
                    let sx = coord(x, data.shape[2], target_shape[2], zoom[2], mode);
                    let v = sample3(data, sz, sy, sx, order);
                    out_plane[y * target_shape[2] + x] = T::from_sample(v);
                }
            }
        });
    out
}

fn coord(i: usize, old: usize, new: usize, zoom: f32, mode: CoordMode) -> f32 {
    match mode {
        CoordMode::ScipyZoom => {
            if new <= 1 {
                0.0
            } else {
                let _ = zoom;
                (i as f32) * (old.saturating_sub(1) as f32) / (new.saturating_sub(1) as f32)
            }
        }
        CoordMode::SkimageResize => skimage_coord(i, old, new),
    }
}

fn skimage_coord(i: usize, old: usize, new: usize) -> f32 {
    (old as f32 / new as f32) * (i as f32 + 0.5) - 0.5
}

fn clamp_index(v: isize, len: usize) -> usize {
    v.clamp(0, len as isize - 1) as usize
}

fn sample3<T: Copy + Into<f32>>(data: &Volume<T>, z: f32, y: f32, x: f32, order: usize) -> f32 {
    match order {
        0 => {
            let zi = nearest_index(z, data.shape[0]);
            let yi = nearest_index(y, data.shape[1]);
            let xi = nearest_index(x, data.shape[2]);
            (*data.get(zi, yi, xi)).into()
        }
        1 => trilinear(data, z, y, x),
        _ => tricubic(data, z, y, x),
    }
}

fn sample2(data: &Volume<f32>, z: usize, y: f32, x: f32, order: usize) -> f32 {
    match order {
        0 => *data.get(
            z,
            nearest_index(y, data.shape[1]),
            nearest_index(x, data.shape[2]),
        ),
        1 => bilinear_yx(data, z, y, x),
        _ => bicubic_yx(data, z, y, x),
    }
}

fn sample2_zx(data: &Volume<f32>, y: usize, z: f32, x: f32, order: usize) -> f32 {
    match order {
        0 => *data.get(
            nearest_index(z, data.shape[0]),
            y,
            nearest_index(x, data.shape[2]),
        ),
        1 => bilinear_zx(data, y, z, x),
        _ => bicubic_zx(data, y, z, x),
    }
}

fn sample2_zy(data: &Volume<f32>, x: usize, z: f32, y: f32, order: usize) -> f32 {
    match order {
        0 => *data.get(
            nearest_index(z, data.shape[0]),
            nearest_index(y, data.shape[1]),
            x,
        ),
        1 => bilinear_zy(data, x, z, y),
        _ => bicubic_zy(data, x, z, y),
    }
}

fn trilinear<T: Copy + Into<f32>>(data: &Volume<T>, z: f32, y: f32, x: f32) -> f32 {
    let z0 = z.floor();
    let y0 = y.floor();
    let x0 = x.floor();
    let dz = z - z0;
    let dy = y - y0;
    let dx = x - x0;
    let z0 = z0 as isize;
    let y0 = y0 as isize;
    let x0 = x0 as isize;
    let mut acc = 0.0;
    for oz in 0..2 {
        let wz = if oz == 0 { 1.0 - dz } else { dz };
        let zi = clamp_index(z0 + oz, data.shape[0]);
        for oy in 0..2 {
            let wy = if oy == 0 { 1.0 - dy } else { dy };
            let yi = clamp_index(y0 + oy, data.shape[1]);
            for ox in 0..2 {
                let wx = if ox == 0 { 1.0 - dx } else { dx };
                let xi = clamp_index(x0 + ox, data.shape[2]);
                acc += wz * wy * wx * (*data.get(zi, yi, xi)).into();
            }
        }
    }
    acc
}

fn cubic_weight(t: f32) -> f32 {
    let a = -0.5;
    let x = t.abs();
    if x <= 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
    } else {
        0.0
    }
}

fn tricubic<T: Copy + Into<f32>>(data: &Volume<T>, z: f32, y: f32, x: f32) -> f32 {
    let zf = z.floor() as isize;
    let yf = y.floor() as isize;
    let xf = x.floor() as isize;
    let mut acc = 0.0;
    for oz in -1..=2 {
        let wz = cubic_weight(z - (zf + oz) as f32);
        let zi = clamp_index(zf + oz, data.shape[0]);
        for oy in -1..=2 {
            let wy = cubic_weight(y - (yf + oy) as f32);
            let yi = clamp_index(yf + oy, data.shape[1]);
            for ox in -1..=2 {
                let wx = cubic_weight(x - (xf + ox) as f32);
                let xi = clamp_index(xf + ox, data.shape[2]);
                acc += wz * wy * wx * (*data.get(zi, yi, xi)).into();
            }
        }
    }
    acc
}

fn bilinear_yx(data: &Volume<f32>, z: usize, y: f32, x: f32) -> f32 {
    let y0 = y.floor() as isize;
    let x0 = x.floor() as isize;
    let dy = y - y0 as f32;
    let dx = x - x0 as f32;
    let mut acc = 0.0;
    for oy in 0..2 {
        let wy = if oy == 0 { 1.0 - dy } else { dy };
        for ox in 0..2 {
            let wx = if ox == 0 { 1.0 - dx } else { dx };
            acc += wy
                * wx
                * *data.get(
                    z,
                    clamp_index(y0 + oy, data.shape[1]),
                    clamp_index(x0 + ox, data.shape[2]),
                );
        }
    }
    acc
}

fn bilinear_zx(data: &Volume<f32>, y: usize, z: f32, x: f32) -> f32 {
    let z0 = z.floor() as isize;
    let x0 = x.floor() as isize;
    let dz = z - z0 as f32;
    let dx = x - x0 as f32;
    let mut acc = 0.0;
    for oz in 0..2 {
        let wz = if oz == 0 { 1.0 - dz } else { dz };
        for ox in 0..2 {
            let wx = if ox == 0 { 1.0 - dx } else { dx };
            acc += wz
                * wx
                * *data.get(
                    clamp_index(z0 + oz, data.shape[0]),
                    y,
                    clamp_index(x0 + ox, data.shape[2]),
                );
        }
    }
    acc
}

fn bilinear_zy(data: &Volume<f32>, x: usize, z: f32, y: f32) -> f32 {
    let z0 = z.floor() as isize;
    let y0 = y.floor() as isize;
    let dz = z - z0 as f32;
    let dy = y - y0 as f32;
    let mut acc = 0.0;
    for oz in 0..2 {
        let wz = if oz == 0 { 1.0 - dz } else { dz };
        for oy in 0..2 {
            let wy = if oy == 0 { 1.0 - dy } else { dy };
            acc += wz
                * wy
                * *data.get(
                    clamp_index(z0 + oz, data.shape[0]),
                    clamp_index(y0 + oy, data.shape[1]),
                    x,
                );
        }
    }
    acc
}

fn bicubic_yx(data: &Volume<f32>, z: usize, y: f32, x: f32) -> f32 {
    let yf = y.floor() as isize;
    let xf = x.floor() as isize;
    let mut acc = 0.0;
    for oy in -1..=2 {
        let wy = cubic_weight(y - (yf + oy) as f32);
        for ox in -1..=2 {
            let wx = cubic_weight(x - (xf + ox) as f32);
            acc += wy
                * wx
                * *data.get(
                    z,
                    clamp_index(yf + oy, data.shape[1]),
                    clamp_index(xf + ox, data.shape[2]),
                );
        }
    }
    acc
}

fn bicubic_zx(data: &Volume<f32>, y: usize, z: f32, x: f32) -> f32 {
    let zf = z.floor() as isize;
    let xf = x.floor() as isize;
    let mut acc = 0.0;
    for oz in -1..=2 {
        let wz = cubic_weight(z - (zf + oz) as f32);
        for ox in -1..=2 {
            let wx = cubic_weight(x - (xf + ox) as f32);
            acc += wz
                * wx
                * *data.get(
                    clamp_index(zf + oz, data.shape[0]),
                    y,
                    clamp_index(xf + ox, data.shape[2]),
                );
        }
    }
    acc
}

fn bicubic_zy(data: &Volume<f32>, x: usize, z: f32, y: f32) -> f32 {
    let zf = z.floor() as isize;
    let yf = y.floor() as isize;
    let mut acc = 0.0;
    for oz in -1..=2 {
        let wz = cubic_weight(z - (zf + oz) as f32);
        for oy in -1..=2 {
            let wy = cubic_weight(y - (yf + oy) as f32);
            acc += wz
                * wy
                * *data.get(
                    clamp_index(zf + oz, data.shape[0]),
                    clamp_index(yf + oy, data.shape[1]),
                    x,
                );
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_label_resize_keeps_labels() {
        let v = Volume::from_vec([1, 2, 2], vec![0_u8, 1, 2, 3]).unwrap();
        let out = resize_labels_to_shape(&v, [1, 4, 4]);
        assert_eq!(out.shape, [1, 4, 4]);
        assert!(out.data.iter().all(|v| *v <= 3));
    }

    #[test]
    fn shape_rounding_matches_python_bankers_round() {
        assert_eq!(py_round_usize(62.5), 62);
        assert_eq!(py_round_usize(63.5), 64);
    }

    #[test]
    fn scipy_cubic_zoom_matches_reference_probe() {
        let v = Volume::from_vec([1, 1, 5], vec![0.0_f32, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = change_spacing_to_spacing(&v, [1.0, 1.0, 1.0], [1.0, 1.0, 0.5], 3);
        let expected = [
            0.0, 0.35974592, 0.87076735, 1.35992473, 1.79294472, 2.20705528, 2.64007527,
            3.12923266, 3.64025408, 4.0,
        ];
        assert_eq!(out.shape, [1, 1, 10]);
        for (a, b) in out.data.iter().zip(expected) {
            assert!((*a as f64 - b).abs() < 2e-5, "{a} != {b}");
        }
    }

    #[test]
    fn separate_z_cubic_matches_nnunet_v1_reference_probe() {
        let mut data = Vec::new();
        for z in 0..3 {
            for y in 0..4 {
                for x in 0..5 {
                    data.push(
                        (z as f32 * 0.7).sin()
                            + (y as f32 * 0.4).cos()
                            + (x * x) as f32 * 0.13
                            + (z * y) as f32 * 0.07,
                    );
                }
            }
        }
        let v = Volume::from_vec([3, 4, 5], data).unwrap();
        let out = resample_data_or_seg_to_shape(
            &v,
            [7, 6, 8],
            [4.0, 1.0, 1.0],
            [1.0, 0.6, 0.625],
            3,
            0,
            Some(true),
        );
        let probes = [
            ([0, 0, 0], 0.99875635),
            ([0, 2, 3], 1.27240479),
            ([1, 4, 6], 2.25215411),
            ([3, 1, 1], 1.67774832),
            ([3, 3, 4], 2.20124793),
            ([5, 2, 7], 4.12651062),
            ([6, 5, 7], 3.89830709),
        ];
        for (point, expected) in probes {
            let actual = *out.get(point[0], point[1], point[2]);
            assert!(
                (actual - expected).abs() < 2e-4,
                "{point:?}: {actual} != {expected}"
            );
        }
    }
}
