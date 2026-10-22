use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use byteordered::Endianness;
use ndarray16::{Array3, ArrayView3};
use nifti::{NiftiHeader, NiftiObject, NiftiType, ReaderOptions, writer::WriterOptions};
use rayon::prelude::*;

use crate::volume::Volume;

pub type Affine = [[f64; 4]; 4];

#[derive(Clone, Debug)]
pub struct ImageProps {
    pub original_affine: Affine,
    pub canonical_affine: Affine,
    pub original_shape_xyz: [usize; 3],
    pub canonical_shape_xyz: [usize; 3],
    pub canonicalization: Canonicalization,
    pub spacing_zyx: [f32; 3],
}

/// How an input volume was mapped onto the canonical grid used for inference, and therefore how
/// the label output must be mapped back to the caller's original grid. Input voxel data is never
/// interpolated: the affine is classified by each column's dominant world axis (residual rotation
/// or shear is ignored, as TotalSegmentator's reference pipeline does) and the volume is
/// canonicalized by pure index reindexing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canonicalization {
    /// Axis permutation/flip by index reindexing (no interpolation), inverted the same way.
    /// Carries the [`Orientation`].
    Orthogonal(Orientation),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orientation {
    pub axis_for_world: [usize; 3],
    pub flip_for_world: [bool; 3],
    pub world_for_axis: [usize; 3],
    pub flip_for_axis: [bool; 3],
}

pub fn read_image(path: &Path) -> Result<(Volume<f32>, ImageProps)> {
    read_image_with_orientation(path, false)
}

pub fn read_image_with_orientation(
    path: &Path,
    preserve_input_orientation: bool,
) -> Result<(Volume<f32>, ImageProps)> {
    if path.is_dir() {
        return crate::dicom_io::read_dicom_dir(path, preserve_input_orientation);
    }
    let obj = ReaderOptions::new()
        .read_file(path)
        .with_context(|| format!("failed to read NIfTI {}", path.display()))?;
    let header = obj.header().clone();
    if header.dim[0] != 3 {
        bail!(
            "only 3D NIfTI images are supported, got dim[0]={}",
            header.dim[0]
        );
    }
    let affine = header_affine(&header)?;
    let original_shape_xyz = [
        header.dim[1] as usize,
        header.dim[2] as usize,
        header.dim[3] as usize,
    ];
    let volume = obj.into_volume();
    let raw_data = volume.into_raw_data();
    let datatype = header.data_type().context("unsupported NIfTI datatype")?;
    canonicalize_raw_xyz_volume(
        &raw_data,
        datatype,
        header.endianness,
        header.scl_slope,
        header.scl_inter,
        original_shape_xyz,
        affine,
        preserve_input_orientation,
    )
}

/// Canonicalize a native `[z, y, x]` volume (x fastest) under the same orientation policy as
/// file input. `affine` maps `[x, y, z]` voxel indices to RAS millimetres. The volume is read
/// through a strided view, so no full-volume transpose is allocated.
pub fn canonicalize_zyx_volume(
    raw: &Volume<f32>,
    affine: Affine,
    preserve_input_orientation: bool,
) -> Result<(Volume<f32>, ImageProps)> {
    let [nz, ny, nx] = raw.shape;
    let xyz = ArrayView3::from_shape((nz, ny, nx), raw.data.as_slice())?.permuted_axes([2, 1, 0]);
    canonicalize_xyz_volume(xyz, [nx, ny, nz], affine, preserve_input_orientation)
}

pub(crate) fn canonicalize_xyz_volume(
    raw: ArrayView3<'_, f32>,
    original_shape_xyz: [usize; 3],
    affine: Affine,
    preserve_input_orientation: bool,
) -> Result<(Volume<f32>, ImageProps)> {
    let orientation = if preserve_input_orientation {
        Orientation {
            axis_for_world: [0, 1, 2],
            flip_for_world: [false; 3],
            world_for_axis: [0, 1, 2],
            flip_for_axis: [false; 3],
        }
    } else {
        orientation_from_affine(affine)?
    };
    let canonical_shape_xyz = [
        original_shape_xyz[orientation.axis_for_world[0]],
        original_shape_xyz[orientation.axis_for_world[1]],
        original_shape_xyz[orientation.axis_for_world[2]],
    ];
    let canonical_affine = if preserve_input_orientation {
        affine
    } else {
        canonical_affine_from_orientation(original_shape_xyz, affine, orientation)
    };
    let spacing_xyz = column_spacings(canonical_affine);
    let spacing_zyx = [
        spacing_xyz[2] as f32,
        spacing_xyz[1] as f32,
        spacing_xyz[0] as f32,
    ];
    let nx = canonical_shape_xyz[0];
    let ny = canonical_shape_xyz[1];
    let nz = canonical_shape_xyz[2];
    let mut data = vec![0.0_f32; nx * ny * nz];
    data.par_chunks_mut(nx * ny)
        .enumerate()
        .for_each(|(z, plane)| {
            for y in 0..ny {
                for x in 0..nx {
                    let canon = [x, y, z];
                    let mut old = [0usize; 3];
                    for (world, &axis) in orientation.axis_for_world.iter().enumerate() {
                        let len = original_shape_xyz[axis];
                        old[axis] = if orientation.flip_for_world[world] {
                            len - 1 - canon[world]
                        } else {
                            canon[world]
                        };
                    }
                    plane[y * nx + x] = raw[[old[0], old[1], old[2]]];
                }
            }
        });
    Ok((
        Volume::from_vec([nz, ny, nx], data)?,
        ImageProps {
            original_affine: affine,
            canonical_affine,
            original_shape_xyz,
            canonical_shape_xyz,
            canonicalization: Canonicalization::Orthogonal(orientation),
            spacing_zyx,
        },
    ))
}

fn canonicalize_raw_xyz_volume(
    raw_data: &[u8],
    datatype: NiftiType,
    endianness: Endianness,
    slope: f32,
    intercept: f32,
    original_shape_xyz: [usize; 3],
    affine: Affine,
    preserve_input_orientation: bool,
) -> Result<(Volume<f32>, ImageProps)> {
    let bytes_per_voxel = datatype.size_of();
    match datatype {
        NiftiType::Uint8
        | NiftiType::Int8
        | NiftiType::Uint16
        | NiftiType::Int16
        | NiftiType::Uint32
        | NiftiType::Int32
        | NiftiType::Float32
        | NiftiType::Float64 => {}
        other => bail!("unsupported NIfTI datatype for CT volume: {other:?}"),
    }
    let voxels = original_shape_xyz[0]
        .checked_mul(original_shape_xyz[1])
        .and_then(|n| n.checked_mul(original_shape_xyz[2]))
        .context("NIfTI shape overflow")?;
    let expected_bytes = voxels
        .checked_mul(bytes_per_voxel)
        .context("NIfTI byte length overflow")?;
    if raw_data.len() != expected_bytes {
        bail!(
            "NIfTI data length mismatch: got {} bytes, expected {} bytes for shape {:?}",
            raw_data.len(),
            expected_bytes,
            original_shape_xyz
        );
    }

    let little_endian = matches!(endianness, Endianness::Little);
    let orientation = if preserve_input_orientation {
        Orientation {
            axis_for_world: [0, 1, 2],
            flip_for_world: [false; 3],
            world_for_axis: [0, 1, 2],
            flip_for_axis: [false; 3],
        }
    } else {
        orientation_from_affine(affine)?
    };
    let canonical_shape_xyz = [
        original_shape_xyz[orientation.axis_for_world[0]],
        original_shape_xyz[orientation.axis_for_world[1]],
        original_shape_xyz[orientation.axis_for_world[2]],
    ];
    let canonical_affine = if preserve_input_orientation {
        affine
    } else {
        canonical_affine_from_orientation(original_shape_xyz, affine, orientation)
    };
    let spacing_xyz = column_spacings(canonical_affine);
    let spacing_zyx = [
        spacing_xyz[2] as f32,
        spacing_xyz[1] as f32,
        spacing_xyz[0] as f32,
    ];
    let nx = canonical_shape_xyz[0];
    let ny = canonical_shape_xyz[1];
    let nz = canonical_shape_xyz[2];
    let src_nx = original_shape_xyz[0];
    let src_ny = original_shape_xyz[1];
    let mut data = vec![0.0_f32; nx * ny * nz];
    data.par_chunks_mut(nx * ny)
        .enumerate()
        .for_each(|(z, plane)| {
            for y in 0..ny {
                for x in 0..nx {
                    let canon = [x, y, z];
                    let mut old = [0usize; 3];
                    for (world, &axis) in orientation.axis_for_world.iter().enumerate() {
                        let len = original_shape_xyz[axis];
                        old[axis] = if orientation.flip_for_world[world] {
                            len - 1 - canon[world]
                        } else {
                            canon[world]
                        };
                    }
                    let raw_index = old[0] + src_nx * (old[1] + src_ny * old[2]);
                    let offset = raw_index * bytes_per_voxel;
                    plane[y * nx + x] = read_raw_voxel_f32(
                        raw_data,
                        offset,
                        datatype,
                        little_endian,
                        slope,
                        intercept,
                    );
                }
            }
        });
    Ok((
        Volume::from_vec([nz, ny, nx], data)?,
        ImageProps {
            original_affine: affine,
            canonical_affine,
            original_shape_xyz,
            canonical_shape_xyz,
            canonicalization: Canonicalization::Orthogonal(orientation),
            spacing_zyx,
        },
    ))
}

fn read_raw_voxel_f32(
    raw_data: &[u8],
    offset: usize,
    datatype: NiftiType,
    little_endian: bool,
    slope: f32,
    intercept: f32,
) -> f32 {
    let value = match datatype {
        NiftiType::Uint8 => raw_data[offset] as f32,
        NiftiType::Int8 => (raw_data[offset] as i8) as f32,
        NiftiType::Uint16 => {
            let bytes = [raw_data[offset], raw_data[offset + 1]];
            (if little_endian {
                u16::from_le_bytes(bytes)
            } else {
                u16::from_be_bytes(bytes)
            }) as f32
        }
        NiftiType::Int16 => {
            let bytes = [raw_data[offset], raw_data[offset + 1]];
            (if little_endian {
                i16::from_le_bytes(bytes)
            } else {
                i16::from_be_bytes(bytes)
            }) as f32
        }
        NiftiType::Uint32 => {
            let bytes = [
                raw_data[offset],
                raw_data[offset + 1],
                raw_data[offset + 2],
                raw_data[offset + 3],
            ];
            (if little_endian {
                u32::from_le_bytes(bytes)
            } else {
                u32::from_be_bytes(bytes)
            }) as f32
        }
        NiftiType::Int32 => {
            let bytes = [
                raw_data[offset],
                raw_data[offset + 1],
                raw_data[offset + 2],
                raw_data[offset + 3],
            ];
            (if little_endian {
                i32::from_le_bytes(bytes)
            } else {
                i32::from_be_bytes(bytes)
            }) as f32
        }
        NiftiType::Float32 => {
            let bytes = [
                raw_data[offset],
                raw_data[offset + 1],
                raw_data[offset + 2],
                raw_data[offset + 3],
            ];
            if little_endian {
                f32::from_le_bytes(bytes)
            } else {
                f32::from_be_bytes(bytes)
            }
        }
        NiftiType::Float64 => {
            let bytes = [
                raw_data[offset],
                raw_data[offset + 1],
                raw_data[offset + 2],
                raw_data[offset + 3],
                raw_data[offset + 4],
                raw_data[offset + 5],
                raw_data[offset + 6],
                raw_data[offset + 7],
            ];
            (if little_endian {
                f64::from_le_bytes(bytes)
            } else {
                f64::from_be_bytes(bytes)
            }) as f32
        }
        _ => unreachable!("unsupported NIfTI datatype was validated before read"),
    };
    if slope == 0.0 {
        value
    } else {
        value * slope + intercept
    }
}

pub fn write_seg(
    seg_zyx: &Volume<u8>,
    output: &Path,
    props: &ImageProps,
    canonical_affine: Option<Affine>,
) -> Result<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    let affine = canonical_affine.unwrap_or(props.canonical_affine);
    let Canonicalization::Orthogonal(orientation) = props.canonicalization;
    let (out_xyz, out_affine) = volume_zyx_to_original_xyz(seg_zyx, affine, orientation);
    let mut header = NiftiHeader {
        datatype: NiftiType::Uint8 as i16,
        bitpix: 8,
        scl_slope: 1.0,
        scl_inter: 0.0,
        ..NiftiHeader::default()
    };
    set_sform_affine(&mut header, out_affine);
    WriterOptions::new(output)
        .reference_header(&header)
        .write_nifti(&out_xyz)
        .with_context(|| format!("failed to write {}", output.display()))?;
    Ok(())
}

pub fn write_label_masks(
    seg: &Volume<u8>,
    out_dir: &Path,
    props: &ImageProps,
    id_to_name: &std::collections::HashMap<u8, String>,
    restrict: Option<&[u8]>,
    canonical_affine: Option<Affine>,
    threads: usize,
) -> Result<Vec<String>> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory {}", out_dir.display()))?;
    let mut present: Vec<u8> = seg.data.iter().copied().filter(|v| *v != 0).collect();
    present.sort_unstable();
    present.dedup();
    if let Some(ids) = restrict {
        present.retain(|id| ids.contains(id));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .context("failed to build saving thread pool")?;
    pool.install(|| {
        present.par_iter().try_for_each(|id| -> Result<()> {
            let Some(name) = id_to_name.get(id) else {
                return Ok(());
            };
            let mask = seg.map(|v| u8::from(*v == *id));
            write_seg(
                &mask,
                &out_dir.join(format!("{name}.nii.gz")),
                props,
                canonical_affine,
            )
        })
    })?;
    Ok(present
        .iter()
        .filter_map(|id| id_to_name.get(id).cloned())
        .collect())
}

pub fn lowres_affine(props: &ImageProps, target_spacing_zyx: [f32; 3]) -> Affine {
    let mut aff = props.canonical_affine;
    let orig_xyz = [
        props.spacing_zyx[2] as f64,
        props.spacing_zyx[1] as f64,
        props.spacing_zyx[0] as f64,
    ];
    let target_xyz = [
        target_spacing_zyx[2] as f64,
        target_spacing_zyx[1] as f64,
        target_spacing_zyx[0] as f64,
    ];
    for j in 0..3 {
        let scale = target_xyz[j] / orig_xyz[j];
        for row in aff.iter_mut().take(3) {
            row[j] *= scale;
        }
    }
    aff
}

pub fn is_nifti_path(path: &Path) -> bool {
    let s = path.to_string_lossy().to_ascii_lowercase();
    s.ends_with(".nii") || s.ends_with(".nii.gz")
}

pub fn unzip_to_temp(path: &Path) -> Result<tempfile::TempDir> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file).context("failed to open zip archive")?;
    let tmp = tempfile::tempdir().context("failed to create temporary directory")?;
    archive
        .extract(tmp.path())
        .context("failed to extract DICOM zip")?;
    Ok(tmp)
}

fn canonical_affine_from_orientation(
    original_shape: [usize; 3],
    affine: Affine,
    orientation: Orientation,
) -> Affine {
    let mut new_aff = [[0.0; 4]; 4];
    new_aff[3][3] = 1.0;
    for row in 0..3 {
        new_aff[row][3] = affine[row][3];
    }
    for world in 0..3 {
        let axis = orientation.axis_for_world[world];
        let sign = if orientation.flip_for_world[world] {
            -1.0
        } else {
            1.0
        };
        if orientation.flip_for_world[world] {
            for row in 0..3 {
                new_aff[row][3] += affine[row][axis] * (original_shape[axis] - 1) as f64;
            }
        }
        for row in 0..3 {
            new_aff[row][world] = sign * affine[row][axis];
        }
    }
    new_aff
}

/// Map canonical `[z, y, x]` labels back onto the caller's native `[z, y, x]` grid.
pub fn restore_labels_zyx(seg: &Volume<u8>, props: &ImageProps) -> Result<Volume<u8>> {
    let Canonicalization::Orthogonal(orientation) = props.canonicalization;
    let (xyz, _) = volume_zyx_to_original_xyz(seg, props.canonical_affine, orientation);
    let (nx, ny, nz) = xyz.dim();
    let mut result = Volume::zeros([nz, ny, nx]);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                *result.get_mut(z, y, x) = xyz[[x, y, z]];
            }
        }
    }
    Ok(result)
}

fn volume_zyx_to_original_xyz<T: Copy + Default>(
    canonical_zyx: &Volume<T>,
    affine: Affine,
    orientation: Orientation,
) -> (Array3<T>, Affine) {
    let canon_shape = [
        canonical_zyx.shape[2],
        canonical_zyx.shape[1],
        canonical_zyx.shape[0],
    ];
    let out_shape = [
        canon_shape[orientation.world_for_axis[0]],
        canon_shape[orientation.world_for_axis[1]],
        canon_shape[orientation.world_for_axis[2]],
    ];
    let mut out = Array3::<T>::default((out_shape[0], out_shape[1], out_shape[2]));
    for ox in 0..out_shape[0] {
        for oy in 0..out_shape[1] {
            for oz in 0..out_shape[2] {
                let old = [ox, oy, oz];
                let mut canon = [0usize; 3];
                for (axis, &world) in orientation.world_for_axis.iter().enumerate() {
                    let len = canon_shape[world];
                    canon[world] = if orientation.flip_for_axis[axis] {
                        len - 1 - old[axis]
                    } else {
                        old[axis]
                    };
                }
                out[[ox, oy, oz]] = *canonical_zyx.get(canon[2], canon[1], canon[0]);
            }
        }
    }
    let mut out_aff = [[0.0; 4]; 4];
    out_aff[3][3] = 1.0;
    for row in 0..3 {
        out_aff[row][3] = affine[row][3];
    }
    for axis in 0..3 {
        let world = orientation.world_for_axis[axis];
        let sign = if orientation.flip_for_axis[axis] {
            -1.0
        } else {
            1.0
        };
        if orientation.flip_for_axis[axis] {
            for row in 0..3 {
                out_aff[row][3] += affine[row][world] * (canon_shape[world] - 1) as f64;
            }
        }
        for row in 0..3 {
            out_aff[row][axis] = sign * affine[row][world];
        }
    }
    (out, out_aff)
}

/// Classify an affine as an axis permutation with per-axis flips from each column's dominant
/// world axis. Residual rotation or shear (an oblique acquisition) is ignored, matching nibabel's
/// closest-canonical rule used by TotalSegmentator's reference pipeline; voxel data is never
/// interpolated on input and the labels are written back onto the caller's grid and affine.
/// Errors only for a degenerate column, two columns sharing a dominant axis, or a column tilted
/// 45° or more from every world axis (no dominant axis).
fn orientation_from_affine(aff: Affine) -> Result<Orientation> {
    let mut used_world = [false; 3];
    let mut world_for_axis = [0usize; 3];
    let mut flip_for_axis = [false; 3];
    for axis in 0..3 {
        let mut best = 0usize;
        let mut best_abs = -1.0;
        let mut norm_sq = 0.0;
        for (world, row) in aff.iter().enumerate().take(3) {
            let v = row[axis].abs();
            norm_sq += v * v;
            if v > best_abs {
                best_abs = v;
                best = world;
            }
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm <= 0.0 || used_world[best] {
            bail!(
                "unsupported NIfTI affine orientation; each voxel axis must map to a distinct world axis"
            );
        }
        if best_abs * std::f64::consts::SQRT_2 < norm {
            bail!(
                "unsupported NIfTI affine orientation; a voxel axis is tilted 45° or more from every world axis"
            );
        }
        used_world[best] = true;
        world_for_axis[axis] = best;
        flip_for_axis[axis] = aff[best][axis] < 0.0;
    }
    let mut axis_for_world = [0usize; 3];
    let mut flip_for_world = [false; 3];
    for axis in 0..3 {
        let world = world_for_axis[axis];
        axis_for_world[world] = axis;
        flip_for_world[world] = flip_for_axis[axis];
    }
    Ok(Orientation {
        axis_for_world,
        flip_for_world,
        world_for_axis,
        flip_for_axis,
    })
}

fn header_affine(header: &NiftiHeader) -> Result<Affine> {
    if header.sform_code != 0 {
        Ok([
            [
                header.srow_x[0] as f64,
                header.srow_x[1] as f64,
                header.srow_x[2] as f64,
                header.srow_x[3] as f64,
            ],
            [
                header.srow_y[0] as f64,
                header.srow_y[1] as f64,
                header.srow_y[2] as f64,
                header.srow_y[3] as f64,
            ],
            [
                header.srow_z[0] as f64,
                header.srow_z[1] as f64,
                header.srow_z[2] as f64,
                header.srow_z[3] as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
    } else if header.qform_code != 0 {
        qform_affine(header)
    } else {
        base_affine(header)
    }
}

fn column_spacings(aff: Affine) -> [f64; 3] {
    let mut out = [1.0; 3];
    for (j, slot) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for row in aff.iter().take(3) {
            s += row[j] * row[j];
        }
        *slot = s.sqrt();
    }
    out
}

fn qform_affine(header: &NiftiHeader) -> Result<Affine> {
    let pixdim = header.pixdim;
    if pixdim[1] < 0.0 || pixdim[2] < 0.0 || pixdim[3] < 0.0 {
        bail!("invalid NIfTI qform: pixdim spacings must be non-negative");
    }
    if (pixdim[0].abs() - 1.0).abs() >= 1e-6 {
        bail!("invalid NIfTI qform: qfac pixdim[0] must be 1 or -1");
    }

    let b = header.quatern_b as f64;
    let c = header.quatern_c as f64;
    let d = header.quatern_d as f64;
    let a2 = 1.0 - (b * b + c * c + d * d);
    if a2 < -f32::EPSILON as f64 * 3.0 {
        bail!("invalid NIfTI qform quaternion");
    }
    let a = a2.max(0.0).sqrt();
    let nq = a * a + b * b + c * c + d * d;
    let r = if nq < f64::EPSILON {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    } else {
        let s = 2.0 / nq;
        let x = b * s;
        let y = c * s;
        let z = d * s;
        let wx = a * x;
        let wy = a * y;
        let wz = a * z;
        let xx = b * x;
        let xy = b * y;
        let xz = b * z;
        let yy = c * y;
        let yz = c * z;
        let zz = d * z;
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    };
    let spacing = [
        pixdim[1] as f64,
        pixdim[2] as f64,
        pixdim[3] as f64 * pixdim[0] as f64,
    ];
    Ok([
        [
            r[0][0] * spacing[0],
            r[0][1] * spacing[1],
            r[0][2] * spacing[2],
            header.quatern_x as f64,
        ],
        [
            r[1][0] * spacing[0],
            r[1][1] * spacing[1],
            r[1][2] * spacing[2],
            header.quatern_y as f64,
        ],
        [
            r[2][0] * spacing[0],
            r[2][1] * spacing[1],
            r[2][2] * spacing[2],
            header.quatern_z as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn base_affine(header: &NiftiHeader) -> Result<Affine> {
    if header.dim[0] < 3 {
        bail!("cannot build NIfTI base affine for fewer than 3 dimensions");
    }
    let shape = [
        header.dim[1] as f64,
        header.dim[2] as f64,
        header.dim[3] as f64,
    ];
    let spacing = [
        -(header.pixdim[1] as f64),
        header.pixdim[2] as f64,
        header.pixdim[3] as f64,
    ];
    let origin = [
        (shape[0] - 1.0) * 0.5,
        (shape[1] - 1.0) * 0.5,
        (shape[2] - 1.0) * 0.5,
    ];
    Ok([
        [spacing[0], 0.0, 0.0, -origin[0] * spacing[0]],
        [0.0, spacing[1], 0.0, -origin[1] * spacing[1]],
        [0.0, 0.0, spacing[2], -origin[2] * spacing[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn set_sform_affine(header: &mut NiftiHeader, affine: Affine) {
    header.sform_code = 2;
    header.qform_code = 0;
    header.srow_x = [
        affine[0][0] as f32,
        affine[0][1] as f32,
        affine[0][2] as f32,
        affine[0][3] as f32,
    ];
    header.srow_y = [
        affine[1][0] as f32,
        affine[1][1] as f32,
        affine[1][2] as f32,
        affine[1][3] as f32,
    ];
    header.srow_z = [
        affine[2][0] as f32,
        affine[2][1] as f32,
        affine[2][2] as f32,
        affine[2][3] as f32,
    ];
    let spacing = column_spacings(affine);
    header.pixdim[0] = 1.0;
    header.pixdim[1] = spacing[0] as f32;
    header.pixdim[2] = spacing[1] as f32;
    header.pixdim[3] = spacing[2] as f32;
}

pub fn collect_nifti_or_dicom_input(path: &Path) -> Result<(PathBuf, Option<tempfile::TempDir>)> {
    if path
        .to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".zip")
    {
        let tmp = unzip_to_temp(path)?;
        Ok((tmp.path().to_path_buf(), Some(tmp)))
    } else {
        Ok((path.to_path_buf(), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_accepts_axis_aligned_affine_with_tiny_noise() {
        let orientation = orientation_from_affine([
            [-1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 2e-7, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .expect("near-axis-aligned affine should be accepted");

        assert_eq!(orientation.world_for_axis, [0, 1, 2]);
        assert_eq!(orientation.flip_for_axis, [true, false, false]);
    }

    #[test]
    fn orientation_ignores_small_obliquity() {
        // TotalSegmentator development-set scan s1389 (4.4° oblique): dominant-axis classification,
        // no interpolation, as the reference pipeline does.
        let orientation = orientation_from_affine([
            [1.4956, -0.0914, -0.0700, -168.4233],
            [0.0913, 1.4972, -0.0037, 8.928],
            [0.0698, 0.0, 1.4984, 1775.149],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .expect("mildly oblique affine should classify by dominant axis");

        assert_eq!(orientation.world_for_axis, [0, 1, 2]);
        assert_eq!(orientation.flip_for_axis, [false, false, false]);
    }

    #[test]
    fn orientation_rejects_affine_without_dominant_axis() {
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let err = orientation_from_affine([
            [c, -c, 0.0, 0.0],
            [c, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .expect_err("45° rotation has no dominant axis");

        assert!(
            err.to_string()
                .contains("unsupported NIfTI affine orientation"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn axis_aligned_affine_stays_on_orthogonal_path() {
        let raw = Array3::from_shape_vec((2, 2, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let affine = [
            [1.5, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let (canonical, props) = canonicalize_xyz_volume(raw.view(), [2, 2, 1], affine, false).unwrap();

        assert_eq!(canonical.shape, [1, 2, 2]);
        assert_eq!(canonical.data, vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(props.canonical_affine, affine);
        assert_eq!(props.canonical_shape_xyz, [2, 2, 1]);
        assert_eq!(
            props.canonicalization,
            Canonicalization::Orthogonal(Orientation {
                axis_for_world: [0, 1, 2],
                flip_for_world: [false; 3],
                world_for_axis: [0, 1, 2],
                flip_for_axis: [false; 3],
            })
        );
    }

    #[test]
    fn xyz_volume_can_preserve_stored_index_order() {
        let raw = Array3::from_shape_vec((2, 2, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let affine = [
            [-1.0, 0.0, 0.0, 10.0],
            [0.0, -1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let (preserved, props) = canonicalize_xyz_volume(raw.view(), [2, 2, 1], affine, true).unwrap();

        assert_eq!(preserved.data, vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(props.canonical_affine, affine);
        let Canonicalization::Orthogonal(orientation) = props.canonicalization;
        assert_eq!(orientation.axis_for_world, [0, 1, 2]);
        assert_eq!(orientation.flip_for_world, [false; 3]);
    }

    #[test]
    fn qform_affine_matches_identity_quaternion_scaling() {
        let header = NiftiHeader {
            sform_code: 0,
            qform_code: 1,
            pixdim: [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            quatern_x: 10.0,
            quatern_y: 20.0,
            quatern_z: 30.0,
            ..NiftiHeader::default()
        };

        let affine = header_affine(&header).expect("valid qform");

        assert_eq!(
            affine,
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
    }

    #[test]
    fn base_affine_matches_nifti_shape_zoom_convention() {
        let header = NiftiHeader {
            dim: [3, 3, 5, 7, 1, 1, 1, 1],
            sform_code: 0,
            qform_code: 0,
            pixdim: [1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ..NiftiHeader::default()
        };

        let affine = header_affine(&header).expect("valid base affine");

        assert_eq!(
            affine,
            [
                [-3.0, 0.0, 0.0, 3.0],
                [0.0, 2.0, 0.0, -4.0],
                [0.0, 0.0, 1.0, -3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
    }
}
