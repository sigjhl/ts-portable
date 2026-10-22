use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use dicom_object::{DefaultDicomObject, OpenFileOptions, Tag, file::ReadPreamble};
use dicom_pixeldata::PixelDecoder;
use rayon::prelude::*;
use std::io::Cursor;
use walkdir::WalkDir;

use crate::{io, io::ImageProps, volume::Volume};

struct Slice {
    path: PathBuf,
    series_uid: String,
    instance: i32,
    ipp: Option<[f64; 3]>,
    iop: Option<[f64; 6]>,
    rows: usize,
    cols: usize,
    pixel_spacing: [f32; 2],
    slice_thickness: Option<f32>,
}

/// Open a DICOM file, tolerating a file meta group that lacks `(0002,0000)`
/// FileMetaInformationGroupLength. dicom-rs requires that element to be the
/// first in group 2 and fails with `UnexpectedTag` otherwise; some exports
/// (e.g. the RSNA-2023 abdominal trauma DICOMs) omit it entirely. When the
/// strict open fails, the group-2 elements are scanned (explicit VR little
/// endian, as the standard requires for the meta group) and a group length
/// element is synthesised in front of them; the pixel data is untouched.
pub fn open_file_lenient(path: &Path) -> Result<dicom_object::DefaultDicomObject> {
    open_file_with_options(path, OpenFileOptions::new())
}

fn open_header_lenient(path: &Path) -> Result<DefaultDicomObject> {
    open_file_with_options(path, OpenFileOptions::new().read_until(Tag(0x7FE0, 0x0010)))
}

fn open_file_with_options(path: &Path, options: OpenFileOptions) -> Result<DefaultDicomObject> {
    match options.clone().open_file(path) {
        Ok(obj) => Ok(obj),
        Err(strict_err) => {
            let bytes =
                fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
            let Some(repaired) = insert_meta_group_length(&bytes) else {
                return Err(strict_err)
                    .context("strict open failed and the meta group could not be repaired");
            };
            options
                .read_preamble(ReadPreamble::Always)
                .from_reader(Cursor::new(repaired))
                .with_context(|| {
                    format!("failed to open {} after meta-group repair", path.display())
                })
        }
    }
}

/// If `bytes` is a Part-10 file (128-byte preamble + "DICM") whose meta group
/// does not start with (0002,0000), return a copy with a synthesised group
/// length element inserted. Returns `None` when the file has no "DICM" magic,
/// already carries a group length, or the meta group cannot be walked.
pub(crate) fn insert_meta_group_length(bytes: &[u8]) -> Option<Vec<u8>> {
    const HEAD: usize = 132;
    if bytes.len() < HEAD + 8 || &bytes[128..132] != b"DICM" {
        return None;
    }
    let tag_at = |off: usize| -> Option<(u16, u16)> {
        let g = u16::from_le_bytes([*bytes.get(off)?, *bytes.get(off + 1)?]);
        let e = u16::from_le_bytes([*bytes.get(off + 2)?, *bytes.get(off + 3)?]);
        Some((g, e))
    };
    if tag_at(HEAD)? == (0x0002, 0x0000) {
        return None;
    }
    // Walk explicit-VR little-endian elements while they belong to group 2.
    let mut off = HEAD;
    loop {
        let (g, _) = tag_at(off)?;
        if g != 0x0002 {
            break;
        }
        let vr = [*bytes.get(off + 4)?, *bytes.get(off + 5)?];
        let (header_len, value_len) = match &vr {
            b"OB" | b"OW" | b"OF" | b"SQ" | b"UT" | b"UN" | b"OD" | b"OL" | b"OV" | b"SV"
            | b"UV" | b"UC" | b"UR" => {
                let len = u32::from_le_bytes([
                    *bytes.get(off + 8)?,
                    *bytes.get(off + 9)?,
                    *bytes.get(off + 10)?,
                    *bytes.get(off + 11)?,
                ]);
                (12usize, len as usize)
            }
            _ => {
                let len = u16::from_le_bytes([*bytes.get(off + 6)?, *bytes.get(off + 7)?]);
                (8usize, len as usize)
            }
        };
        if value_len == 0xFFFF_FFFF {
            return None; // undefined length is not valid in the meta group
        }
        off = off.checked_add(header_len)?.checked_add(value_len)?;
        if off > bytes.len() {
            return None;
        }
    }
    let group_len = (off - HEAD) as u32;
    let mut out = Vec::with_capacity(bytes.len() + 12);
    out.extend_from_slice(&bytes[..HEAD]);
    out.extend_from_slice(&[0x02, 0x00, 0x00, 0x00, b'U', b'L', 4, 0]);
    out.extend_from_slice(&group_len.to_le_bytes());
    out.extend_from_slice(&bytes[HEAD..]);
    Some(out)
}

/// A decoded DICOM image before inference reorientation, resampling or normalization.
/// `image` is x-fastest, with shape `[z, y, x]`; `affine_ras` maps `[x, y, z]`
/// voxel indices to RAS millimetres. Keep this image for radiomics and prepare
/// inference input with `io::canonicalize_zyx_volume` without reopening DICOM.
pub struct NativeDicom {
    pub image: Volume<f32>,
    pub affine_ras: io::Affine,
    pub series_uid: String,
}

pub fn read_dicom_dir(
    path: &Path,
    preserve_input_orientation: bool,
) -> Result<(Volume<f32>, ImageProps)> {
    let native = read_dicom_native(path)?;
    io::canonicalize_zyx_volume(&native.image, native.affine_ras, preserve_input_orientation)
}

/// Decode the largest image series, preserving its native voxel order.
pub fn read_dicom_native(path: &Path) -> Result<NativeDicom> {
    let files = WalkDir::new(path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path());
    let slices = read_slice_headers(files);
    if slices.is_empty() {
        bail!("no readable DICOM image slices found in {}", path.display());
    }
    let mut by_series: HashMap<String, Vec<Slice>> = HashMap::new();
    for slice in slices {
        by_series
            .entry(slice.series_uid.clone())
            .or_default()
            .push(slice);
    }
    let series = by_series
        .into_values()
        .max_by_key(|s| s[0].rows * s[0].cols * s.len())
        .context("no DICOM series found")?;
    decode_series(series)
}

/// Decode an explicitly selected series (e.g. `SeriesInfo::files`) without
/// staging hard links/copies or walking the source directory again.
/// Every supplied file must be a readable image from the same series.
pub fn read_dicom_files_native(files: &[PathBuf]) -> Result<NativeDicom> {
    if files.is_empty() {
        bail!("no DICOM files selected");
    }
    let slices = read_slice_headers(files.iter().cloned());
    if slices.len() != files.len() {
        bail!("selected DICOM files include unreadable or non-image instances");
    }
    if slices.iter().any(|s| s.series_uid != slices[0].series_uid) {
        bail!("selected DICOM files contain more than one series");
    }
    decode_series(slices)
}

fn read_slice_headers(files: impl Iterator<Item = PathBuf>) -> Vec<Slice> {
    let mut slices = Vec::new();
    for path in files {
        let p = path.as_path();
        let Ok(obj) = open_header_lenient(p) else {
            continue;
        };
        let Ok(series_uid_elem) = obj.element_by_name("SeriesInstanceUID") else {
            continue;
        };
        let Ok(series_uid) = series_uid_elem.to_str() else {
            continue;
        };
        let Ok(rows_elem) = obj.element_by_name("Rows") else {
            continue;
        };
        let Ok(rows) = rows_elem.to_int::<u16>() else {
            continue;
        };
        let Ok(cols_elem) = obj.element_by_name("Columns") else {
            continue;
        };
        let Ok(cols) = cols_elem.to_int::<u16>() else {
            continue;
        };
        let pixel_spacing = parse_multi_f32(&obj, "PixelSpacing").unwrap_or([1.0, 1.0].to_vec());
        let ipp = parse_multi_f64(&obj, "ImagePositionPatient")
            .ok()
            .and_then(|v| (v.len() >= 3).then_some([v[0], v[1], v[2]]));
        let iop = parse_multi_f64(&obj, "ImageOrientationPatient")
            .ok()
            .and_then(|v| (v.len() >= 6).then_some([v[0], v[1], v[2], v[3], v[4], v[5]]));
        let instance = obj
            .element_by_name("InstanceNumber")
            .ok()
            .and_then(|e| e.to_int::<i32>().ok())
            .unwrap_or(0);
        let slice_thickness = obj
            .element_by_name("SliceThickness")
            .ok()
            .and_then(|e| e.to_float32().ok());
        slices.push(Slice {
            path: p.to_path_buf(),
            series_uid: series_uid.to_string(),
            instance,
            ipp,
            iop,
            rows: rows as usize,
            cols: cols as usize,
            pixel_spacing: [
                *pixel_spacing.first().unwrap_or(&1.0),
                *pixel_spacing.get(1).unwrap_or(&1.0),
            ],
            slice_thickness,
        });
    }
    slices
}

fn decode_series(mut series: Vec<Slice>) -> Result<NativeDicom> {
    sort_slices(&mut series);
    let rows = series[0].rows;
    let cols = series[0].cols;
    if series.iter().any(|s| s.rows != rows || s.cols != cols) {
        bail!("DICOM series has mixed row/column sizes");
    }
    let z = series.len();
    let plane_len = rows
        .checked_mul(cols)
        .context("DICOM plane dimensions overflow")?;
    let len = plane_len
        .checked_mul(z)
        .context("DICOM volume dimensions overflow")?;
    if plane_len == 0 {
        bail!("DICOM image dimensions must be nonzero");
    }
    let mut data = vec![0.0_f32; len];
    // Each worker owns one destination plane. In-flight decoder allocations are
    // bounded by the existing Rayon pool; slice order is independent of scheduling.
    data.par_chunks_mut(plane_len)
        .zip(series.par_iter())
        .try_for_each(|(plane, s)| -> Result<()> {
            let obj = open_file_lenient(&s.path)
                .with_context(|| format!("failed to open {}", s.path.display()))?;
            let decoded = obj
                .decode_pixel_data()
                .with_context(|| format!("failed to decode pixel data in {}", s.path.display()))?;
            // `to_ndarray` applies the DICOM Modality LUT (rescale slope/intercept)
            // by default. Decode directly to f32 so valid modality values outside
            // the i16 range (seen on some PACS CT exports) remain representable.
            // Do not apply slope/intercept again below: that would double-rescale HU.
            let arr = decoded
                .to_ndarray::<f32>()
                .context("failed to convert DICOM pixels to ndarray")?;
            let shape = arr.shape().to_vec();
            let (raw, offset) = arr.into_raw_vec_and_offset();
            if offset != Some(0) {
                bail!("unsupported DICOM pixel array memory offset {:?}", offset);
            }
            validate_pixel_array_shape(&shape, rows, cols)?;
            if raw.len() != plane.len() {
                bail!("DICOM pixel count does not match Rows/Columns");
            }
            plane.copy_from_slice(&raw);
            Ok(())
        })?;
    Ok(NativeDicom {
        image: Volume::from_vec([z, rows, cols], data)?,
        affine_ras: dicom_affine_ras(&series),
        series_uid: series[0].series_uid.clone(),
    })
}

/// One CT series discovered under a directory, with enough metadata to identify
/// the patient and study and the list of files that make it up.
pub struct SeriesInfo {
    pub series_uid: String,
    pub study_uid: String,
    pub patient_name: String,
    pub patient_id: String,
    pub protocol: String,
    pub description: String,
    pub modality: String,
    pub rows: u16,
    pub cols: u16,
    pub files: Vec<PathBuf>,
}

impl SeriesInfo {
    /// A short, human-friendly label for a chooser and for output-folder names.
    pub fn label(&self) -> String {
        let tail: String = self
            .series_uid
            .chars()
            .rev()
            .take(6)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        let desc = self.description.trim();
        if !desc.is_empty() {
            desc.to_string()
        } else if !self.modality.is_empty() {
            format!("{} series {tail}", self.modality)
        } else {
            format!("series {tail}")
        }
    }

    /// Patient name with DICOM component delimiters flattened to spaces.
    pub fn patient_display(&self) -> String {
        let name = self.patient_name.replace('^', " ");
        let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
        match (name.is_empty(), self.patient_id.is_empty()) {
            (false, false) => format!("{name} · MRN {}", self.patient_id),
            (false, true) => name,
            (true, false) => format!("MRN {}", self.patient_id),
            (true, true) => "unknown patient".to_string(),
        }
    }

    pub fn slice_count(&self) -> usize {
        self.files.len()
    }
}

/// Enumerate the DICOM series under `path` (recursively), largest first. Used to
/// let a caller choose a series when a folder holds more than one, without
/// reading any pixel data.
pub fn list_dicom_series(path: &Path) -> Result<Vec<SeriesInfo>> {
    struct Acc {
        study: String,
        patient_name: String,
        patient_id: String,
        protocol: String,
        desc: String,
        modality: String,
        rows: u16,
        cols: u16,
        files: Vec<PathBuf>,
    }
    let mut by_series: HashMap<String, Acc> = HashMap::new();
    for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let Ok(obj) = open_header_lenient(p) else {
            continue;
        };
        let Ok(series_elem) = obj.element_by_name("SeriesInstanceUID") else {
            continue;
        };
        let Ok(series_uid) = series_elem.to_str() else {
            continue;
        };
        // Require Rows/Columns so only image slices are counted.
        let Ok(rows_elem) = obj.element_by_name("Rows") else {
            continue;
        };
        let Ok(rows) = rows_elem.to_int::<u16>() else {
            continue;
        };
        let Ok(cols_elem) = obj.element_by_name("Columns") else {
            continue;
        };
        let Ok(cols) = cols_elem.to_int::<u16>() else {
            continue;
        };
        let read_str = |name: &str| -> String {
            obj.element_by_name(name)
                .ok()
                .and_then(|e| e.to_str().ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };
        let protocol = {
            let p = read_str("ProtocolName");
            if p.is_empty() {
                read_str("StudyDescription")
            } else {
                p
            }
        };
        let acc = by_series
            .entry(series_uid.to_string())
            .or_insert_with(|| Acc {
                study: read_str("StudyInstanceUID"),
                patient_name: read_str("PatientName"),
                patient_id: read_str("PatientID"),
                protocol,
                desc: read_str("SeriesDescription"),
                modality: read_str("Modality"),
                rows,
                cols,
                files: Vec::new(),
            });
        acc.files.push(p.to_path_buf());
    }
    let mut out: Vec<SeriesInfo> = by_series
        .into_iter()
        .map(|(series_uid, a)| SeriesInfo {
            series_uid,
            study_uid: a.study,
            patient_name: a.patient_name,
            patient_id: a.patient_id,
            protocol: a.protocol,
            description: a.desc,
            modality: a.modality,
            rows: a.rows,
            cols: a.cols,
            files: a.files,
        })
        .collect();
    // Largest by voxel count first, matching read_dicom_dir's default choice.
    out.sort_by_key(|s| std::cmp::Reverse(s.rows as usize * s.cols as usize * s.files.len()));
    Ok(out)
}

/// Copy the given slice files into a fresh temp directory that contains only
/// them, so a subsequent `read_dicom_dir` loads exactly this series. Hard-links
/// when possible (no data copy) and falls back to a byte copy across volumes.
/// The returned `TempDir` deletes the directory when dropped.
pub fn stage_files(files: &[PathBuf]) -> std::io::Result<tempfile::TempDir> {
    let dir = tempfile::Builder::new()
        .prefix("onnx_totalseg_series_")
        .tempdir()?;
    for (i, f) in files.iter().enumerate() {
        let dest = dir.path().join(format!("{i:06}.dcm"));
        if std::fs::hard_link(f, &dest).is_err() {
            std::fs::copy(f, &dest)?;
        }
    }
    Ok(dir)
}

/// Stage the series with the given SeriesInstanceUID from a directory into a
/// temp directory containing only that series.
pub fn stage_series_from_dir(dir: &Path, series_uid: &str) -> Result<tempfile::TempDir> {
    let series = list_dicom_series(dir)?;
    let target = series
        .into_iter()
        .find(|s| s.series_uid == series_uid)
        .with_context(|| format!("series {series_uid} not found under {}", dir.display()))?;
    stage_files(&target.files).map_err(Into::into)
}

fn validate_pixel_array_shape(shape: &[usize], rows: usize, cols: usize) -> Result<()> {
    if shape.len() != 4 || shape[1] != rows || shape[2] != cols {
        bail!("unsupported DICOM pixel array shape {:?}", shape);
    }
    if shape[0] != 1 {
        bail!(
            "unsupported multi-frame DICOM pixel array shape {:?}; provide a single-frame CT series",
            shape
        );
    }
    if shape[3] != 1 {
        bail!(
            "unsupported multi-sample DICOM pixel array shape {:?}; expected monochrome CT pixels",
            shape
        );
    }
    Ok(())
}

fn parse_multi_f32(obj: &DefaultDicomObject, name: &str) -> Result<Vec<f32>> {
    let text = obj.element_by_name(name)?.to_str()?;
    Ok(text
        .split('\\')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect())
}

fn parse_multi_f64(obj: &DefaultDicomObject, name: &str) -> Result<Vec<f64>> {
    let text = obj.element_by_name(name)?.to_str()?;
    Ok(text
        .split('\\')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect())
}

fn sort_slices(series: &mut [Slice]) {
    let normal = series.first().and_then(|s| s.iop).map(slice_normal);
    series.sort_by(|a, b| match (normal, a.ipp, b.ipp) {
        (Some(n), Some(aa), Some(bb)) => dot(aa, n).total_cmp(&dot(bb, n)),
        _ => a.instance.cmp(&b.instance),
    });
}

fn dicom_affine_ras(series: &[Slice]) -> [[f64; 4]; 4] {
    let first = &series[0];
    let iop = first.iop.unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let row = [iop[0], iop[1], iop[2]];
    let col = [iop[3], iop[4], iop[5]];
    let normal = slice_normal(iop);
    let origin = first.ipp.unwrap_or([0.0, 0.0, 0.0]);
    let spacing_z = slice_spacing(series, normal)
        .or(first.slice_thickness.map(|v| v as f64))
        .unwrap_or(1.0)
        .max(1e-6);

    let col_spacing = first.pixel_spacing[1] as f64;
    let row_spacing = first.pixel_spacing[0] as f64;
    let mut aff = [[0.0_f64; 4]; 4];
    aff[0][0] = row[0] * col_spacing;
    aff[1][0] = row[1] * col_spacing;
    aff[2][0] = row[2] * col_spacing;
    aff[0][1] = col[0] * row_spacing;
    aff[1][1] = col[1] * row_spacing;
    aff[2][1] = col[2] * row_spacing;
    aff[0][2] = normal[0] * spacing_z;
    aff[1][2] = normal[1] * spacing_z;
    aff[2][2] = normal[2] * spacing_z;
    aff[0][3] = origin[0];
    aff[1][3] = origin[1];
    aff[2][3] = origin[2];
    aff[3][3] = 1.0;
    lps_to_ras(aff)
}

fn slice_spacing(series: &[Slice], normal: [f64; 3]) -> Option<f64> {
    if series.len() < 2 {
        return None;
    }
    let mut positions = series
        .iter()
        .filter_map(|s| s.ipp.map(|ipp| dot(ipp, normal)))
        .collect::<Vec<_>>();
    if positions.len() < 2 {
        return None;
    }
    positions.sort_by(|a, b| a.total_cmp(b));
    let mut deltas = positions
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .filter(|d| *d > 1e-6)
        .collect::<Vec<_>>();
    if deltas.is_empty() {
        return None;
    }
    deltas.sort_by(|a, b| a.total_cmp(b));
    Some(deltas[deltas.len() / 2])
}

fn slice_normal(iop: [f64; 6]) -> [f64; 3] {
    normalize(cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]]))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = dot(v, v).sqrt();
    if n <= 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn lps_to_ras(mut aff: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    for value in &mut aff[0] {
        *value = -*value;
    }
    for value in &mut aff[1] {
        *value = -*value;
    }
    aff
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_instance(path: &Path, z: usize, uid: &str) {
        fn element(out: &mut Vec<u8>, tag: (u16, u16), vr: &[u8; 2], mut value: Vec<u8>) {
            if !value.len().is_multiple_of(2) {
                value.push(if vr == b"UI" { 0 } else { b' ' });
            }
            out.extend_from_slice(&tag.0.to_le_bytes());
            out.extend_from_slice(&tag.1.to_le_bytes());
            out.extend_from_slice(vr);
            if vr == b"OW" {
                out.extend_from_slice(&[0, 0]);
                out.extend_from_slice(&(value.len() as u32).to_le_bytes());
            } else {
                out.extend_from_slice(&(value.len() as u16).to_le_bytes());
            }
            out.extend_from_slice(&value);
        }
        let sop = format!("1.2.3.4.{}", z + 1);
        let mut meta = Vec::new();
        element(
            &mut meta,
            (2, 2),
            b"UI",
            b"1.2.840.10008.5.1.4.1.1.2".to_vec(),
        );
        element(&mut meta, (2, 3), b"UI", sop.as_bytes().to_vec());
        element(&mut meta, (2, 0x10), b"UI", b"1.2.840.10008.1.2.1".to_vec());
        element(&mut meta, (2, 0x12), b"UI", b"1.2.3.4.5".to_vec());
        let mut bytes = vec![0; 128];
        bytes.extend_from_slice(b"DICM");
        element(
            &mut bytes,
            (2, 0),
            b"UL",
            (meta.len() as u32).to_le_bytes().to_vec(),
        );
        bytes.extend_from_slice(&meta);
        element(
            &mut bytes,
            (8, 0x16),
            b"UI",
            b"1.2.840.10008.5.1.4.1.1.2".to_vec(),
        );
        element(&mut bytes, (8, 0x18), b"UI", sop.into_bytes());
        element(&mut bytes, (0x20, 0x0e), b"UI", uid.as_bytes().to_vec());
        element(
            &mut bytes,
            (0x20, 0x13),
            b"IS",
            (z + 1).to_string().into_bytes(),
        );
        element(
            &mut bytes,
            (0x20, 0x32),
            b"DS",
            format!("10\\20\\{}", 30 - z * 4).into_bytes(),
        );
        element(
            &mut bytes,
            (0x20, 0x37),
            b"DS",
            b"0\\1\\0\\1\\0\\0".to_vec(),
        );
        element(&mut bytes, (0x28, 2), b"US", 1u16.to_le_bytes().to_vec());
        element(&mut bytes, (0x28, 4), b"CS", b"MONOCHROME2".to_vec());
        element(&mut bytes, (0x28, 0x10), b"US", 3u16.to_le_bytes().to_vec());
        element(&mut bytes, (0x28, 0x11), b"US", 4u16.to_le_bytes().to_vec());
        element(&mut bytes, (0x28, 0x30), b"DS", b"3\\2".to_vec());
        for (tag, value) in [(0x100, 16u16), (0x101, 16), (0x102, 15), (0x103, 1)] {
            element(&mut bytes, (0x28, tag), b"US", value.to_le_bytes().to_vec());
        }
        element(&mut bytes, (0x28, 0x1052), b"DS", b"-1024".to_vec());
        element(&mut bytes, (0x28, 0x1053), b"DS", b"2".to_vec());
        let pixels: Vec<u8> = (0..12)
            .flat_map(|i| ((z * 12 + i) as i16 - 12).to_le_bytes())
            .collect();
        element(&mut bytes, (0x7fe0, 0x10), b"OW", pixels);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn native_dicom_preserves_voxels_geometry_and_selected_series() {
        let dir = tempfile::tempdir().unwrap();
        let files = [dir.path().join("b.dcm"), dir.path().join("a.dcm")];
        for (z, path) in files.iter().enumerate() {
            write_test_instance(path, z, "1.2.3");
        }
        let header = open_header_lenient(&files[0]).unwrap();
        assert!(header.element(Tag(0x7fe0, 0x10)).is_err());
        let listed = list_dicom_series(dir.path()).unwrap();
        assert_eq!(listed.len(), 1);
        let native = read_dicom_files_native(&listed[0].files).unwrap();
        assert_eq!(native.image.shape, [2, 3, 4]);
        assert_eq!(native.series_uid, "1.2.3");
        let expected: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 2.0 - 1024.0).collect();
        assert_eq!(native.image.data, expected);
        assert_eq!(
            native.affine_ras,
            [
                [0.0, -3.0, 0.0, -10.0],
                [-2.0, 0.0, 0.0, -20.0],
                [0.0, 0.0, -4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
        let (canonical, props) = read_dicom_dir(dir.path(), false).unwrap();
        // Known asymmetric landmarks follow the same transform as model labels.
        let labels = canonical.map(|v| ((*v + 1024.0) / 2.0 + 13.0) as u8);
        let restored = io::restore_labels_zyx(&labels, &props).unwrap();
        assert_eq!(restored.data, (1..=24).collect::<Vec<u8>>());
        assert_eq!(restored.shape, native.image.shape);

        write_test_instance(&dir.path().join("other.dcm"), 0, "9.8.7");
        let mut mixed = files.to_vec();
        mixed.push(dir.path().join("other.dcm"));
        assert!(read_dicom_files_native(&mixed).is_err());
        assert!(read_dicom_files_native(&[]).is_err());
        // The explicitly selected file list still loads only the requested series.
        assert_eq!(
            read_dicom_files_native(&files).unwrap().image.data,
            expected
        );

        let empty = tempfile::tempdir().unwrap();
        let Err(err) = read_dicom_native(empty.path()) else {
            panic!("an empty directory must not load");
        };
        assert!(
            err.to_string()
                .contains(&empty.path().display().to_string())
        );
    }

    fn test_slice(instance: i32, ipp: [f64; 3]) -> Slice {
        Slice {
            path: PathBuf::new(),
            series_uid: "1".to_string(),
            instance,
            ipp: Some(ipp),
            iop: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            rows: 2,
            cols: 3,
            pixel_spacing: [2.0, 3.0],
            slice_thickness: Some(4.0),
        }
    }

    #[test]
    fn dicom_affine_converts_axial_lps_to_ras() {
        let series = vec![
            test_slice(1, [10.0, 20.0, 30.0]),
            test_slice(2, [10.0, 20.0, 34.0]),
        ];
        let aff = dicom_affine_ras(&series);
        assert_eq!(aff[0], [-3.0, -0.0, -0.0, -10.0]);
        assert_eq!(aff[1], [-0.0, -2.0, -0.0, -20.0]);
        assert_eq!(aff[2], [0.0, 0.0, 4.0, 30.0]);
        assert_eq!(aff[3], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn sort_slices_uses_position_along_orientation_normal() {
        let mut series = vec![
            test_slice(10, [0.0, 0.0, 8.0]),
            test_slice(2, [0.0, 0.0, 0.0]),
            test_slice(1, [0.0, 0.0, 4.0]),
        ];
        sort_slices(&mut series);
        let z_positions = series
            .iter()
            .map(|s| s.ipp.expect("test slice has ipp")[2])
            .collect::<Vec<_>>();
        assert_eq!(z_positions, vec![0.0, 4.0, 8.0]);
    }

    #[test]
    fn pixel_shape_accepts_single_frame_monochrome() {
        validate_pixel_array_shape(&[1, 16, 32, 1], 16, 32)
            .expect("single-frame monochrome pixels are supported");
    }

    #[test]
    fn pixel_shape_rejects_multiframe() {
        let err = validate_pixel_array_shape(&[2, 16, 32, 1], 16, 32)
            .expect_err("multi-frame pixels must fail clearly");
        assert!(
            err.to_string().contains("multi-frame"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn pixel_shape_rejects_color_samples() {
        let err = validate_pixel_array_shape(&[1, 16, 32, 3], 16, 32)
            .expect_err("multi-sample pixels must fail clearly");
        assert!(
            err.to_string().contains("multi-sample"),
            "unexpected error: {err:#}"
        );
    }
}
