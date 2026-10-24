// Friendly desktop front-end for onnx-totalseg.
//
// It is a thin shell over the tested CLI: add NIfTI scans and/or DICOM series,
// choose one or more organs and a resolution, press Run. It spawns the `onnx-totalseg`
// binary on the crop-cascade path (`--roi_subset <organ>`, INT8, OpenVINO CPU),
// writes masks to `output/<scan>/` next to the executable, and shows the
// per-organ volume and HU that the CLI reports via `--statistics -`.
//
// DICOM folders are enumerated by series; when a folder holds more than one
// series the user chooses which to segment. Each chosen series is staged into a
// temporary directory that contains only that series, so the existing DICOM
// reader loads exactly the intended series with no ambiguity.
//
// The GUI runs no inference itself. All segmentation goes through the same
// binary and code path the command line and the paper use.

// Suppress the console window in release builds on Windows.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::Instant;

use eframe::egui;
use onnx_totalseg_rs::dicom_io::{SeriesInfo, list_dicom_series};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([520.0, 420.0])
            .with_min_inner_size([460.0, 340.0]),
        ..Default::default()
    };
    eframe::run_native(
        "onnx-totalseg — CT organ segmentation",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}

#[derive(Clone, Copy, PartialEq)]
enum Resolution {
    Fast, // 3 mm  (--fast)
    Full, // 1.5 mm (no tier flag)
}

impl Resolution {
    fn label(self) -> &'static str {
        match self {
            Resolution::Fast => "Fast (3 mm)",
            Resolution::Full => "Full (1.5 mm)",
        }
    }
}

// A queued scan: either a NIfTI file or a DICOM series (identified by the folder
// it lives in and its SeriesInstanceUID, which the CLI selects with
// --dicom-series).
#[derive(Clone)]
enum InputItem {
    Nifti(PathBuf),
    Dicom { label: String, dir: PathBuf, series_uid: String },
}

impl InputItem {
    // Base name used for the output folder and result rows (before deduplication).
    fn base_name(&self) -> String {
        match self {
            InputItem::Nifti(p) => scan_stem(p),
            InputItem::Dicom { label, .. } => sanitize(label),
        }
    }
}

// One organ row parsed from the CLI's statistics JSON.
struct Row {
    organ: String,
    volume_ml: f64,
    mean_hu: Option<f64>,
    median_hu: Option<f64>,
    source: String,
}

// Messages from the worker thread back to the UI.
enum Msg {
    Status(String), // transient: the scan currently being processed
    Note(String),   // durable: a per-scan problem or timing, kept in the log
    Rows(Vec<Row>),
    Done(String), // final summary line
}

struct App {
    exe_dir: PathBuf,
    organs: Vec<String>,
    selected_organs: Vec<String>,
    resolution: Resolution,
    save_preview: bool,
    inputs: Vec<InputItem>,
    // When a DICOM folder has multiple series, the user picks which to add.
    // Holds the chosen folder and the candidate series with their checked state.
    series_picker: Option<(PathBuf, Vec<(SeriesInfo, bool)>)>,
    running: bool,
    rx: Option<Receiver<Msg>>,
    status: String,
    log: Vec<String>,
    rows: Vec<Row>,
    setup_error: Option<String>,
}

impl App {
    fn new() -> Self {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."));

        // Populate the organ list from the shipped label map so the names always
        // match the models next to the executable.
        let (organs, setup_error) = match load_organs(&exe_dir) {
            Ok(list) if !list.is_empty() => (list, None),
            Ok(_) => (fallback_organs(), None),
            Err(e) => (fallback_organs(), Some(e)),
        };
        let selected_organs = if organs.iter().any(|o| o == "spleen") {
            vec!["spleen".to_string()]
        } else {
            organs.first().cloned().into_iter().collect()
        };

        Self {
            exe_dir,
            organs,
            selected_organs,
            resolution: Resolution::Full,
            save_preview: true,
            inputs: Vec::new(),
            series_picker: None,
            running: false,
            rx: None,
            status: String::new(),
            log: Vec::new(),
            rows: Vec::new(),
            setup_error,
        }
    }

    // Enumerate a chosen DICOM folder and either queue its single series or open
    // the chooser for multiple.
    fn open_dicom_folder(&mut self, dir: &Path) {
        match list_dicom_series(dir) {
            Ok(series) if series.is_empty() => {
                self.log.push(format!("No DICOM series found in {}.", dir.display()));
            }
            Ok(mut series) if series.len() == 1 => {
                let s = series.remove(0);
                self.inputs = vec![InputItem::Dicom {
                    label: s.label(),
                    dir: dir.to_path_buf(),
                    series_uid: s.series_uid,
                }];
            }
            Ok(series) => {
                // Multiple series: pre-check the largest (first, already sorted).
                let mut picks: Vec<(SeriesInfo, bool)> =
                    series.into_iter().map(|s| (s, false)).collect();
                if let Some(first) = picks.first_mut() {
                    first.1 = true;
                }
                self.series_picker = Some((dir.to_path_buf(), picks));
            }
            Err(e) => {
                self.log.push(format!("Could not read {}: {e}", dir.display()));
            }
        }
    }

    fn start_run(&mut self) {
        self.running = true;
        self.rows.clear();
        self.log.clear();
        self.status = "Starting…".to_string();

        let (tx, rx) = channel();
        self.rx = Some(rx);

        let job = Job {
            exe_dir: self.exe_dir.clone(),
            inputs: self.inputs.clone(),
            organs: self.selected_organs.clone(),
            resolution: self.resolution,
            save_preview: self.save_preview,
        };
        thread::spawn(move || run_job(job, tx));
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain worker messages.
        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Status(s) => self.status = s,
                    Msg::Note(s) => self.log.push(s),
                    Msg::Rows(mut r) => self.rows.append(&mut r),
                    Msg::Done(summary) => {
                        self.status = summary;
                        self.running = false;
                        self.rx = None;
                        break;
                    }
                }
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("CT organ segmentation");
            ui.label("8-bit integer TotalSegmentator · OpenVINO CPU · crop cascade");
            ui.hyperlink_to(
                egui::RichText::new("Companion app of Lee JH, medRxiv 2026, doi:10.64898/2026.10.23.26366189").small(),
                "https://doi.org/10.64898/2026.10.23.26366189",
            );
            ui.add_space(8.0);

            if let Some(err) = &self.setup_error {
                ui.colored_label(egui::Color32::from_rgb(200, 120, 0), err);
                ui.add_space(6.0);
            }

            // --- Inputs ---
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.running, egui::Button::new("Open NIfTI(s)…"))
                    .clicked()
                    && let Some(files) = rfd::FileDialog::new()
                        .add_filter("NIfTI", &["nii", "gz"])
                        .pick_files()
                {
                    // The selected file(s) become the current study/batch. This
                    // keeps them available for an immediate rerun while making
                    // a newly opened study evict the previous one.
                    self.inputs = files.into_iter().map(InputItem::Nifti).collect();
                }
                if ui
                    .add_enabled(!self.running, egui::Button::new("Open DICOM folder…"))
                    .clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                {
                    self.open_dicom_folder(&dir);
                }
                if ui
                    .add_enabled(!self.running && !self.inputs.is_empty(), egui::Button::new("Clear"))
                    .clicked()
                {
                    self.inputs.clear();
                }
            });

            // Queued inputs.
            if self.inputs.is_empty() {
                ui.label("No scans queued.");
            } else {
                ui.label(format!("Current study ({} scan(s)):", self.inputs.len()));
                for item in &self.inputs {
                    ui.label(format!("  • {}", item.base_name()));
                }
            }

            // --- Series chooser (shown when a DICOM folder had multiple series) ---
            let mut add = false;
            let mut cancel = false;
            if let Some((_, picker)) = &mut self.series_picker {
                ui.add_space(6.0);
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.label("This folder has multiple series — choose which to segment:");
                    ui.add_space(2.0);
                    for (info, checked) in picker.iter_mut() {
                        ui.checkbox(checked, info.patient_display());
                        let mut line = String::new();
                        for part in [info.protocol.as_str(), info.description.as_str(), info.modality.as_str()] {
                            if !part.trim().is_empty() {
                                if !line.is_empty() {
                                    line.push_str("  ·  ");
                                }
                                line.push_str(part.trim());
                            }
                        }
                        if !line.is_empty() {
                            line.push_str("  ·  ");
                        }
                        line.push_str(&format!("{} slices", info.slice_count()));
                        ui.horizontal(|ui| {
                            ui.add_space(24.0);
                            ui.weak(line);
                        });
                    }
                    ui.horizontal(|ui| {
                        add = ui.button("Add selected").clicked();
                        cancel = ui.button("Cancel").clicked();
                    });
                });
            }
            if add {
                if let Some((dir, picker)) = self.series_picker.take() {
                    let mut selected = Vec::new();
                    for (info, checked) in picker {
                        if checked {
                            selected.push(InputItem::Dicom {
                                label: info.label(),
                                dir: dir.clone(),
                                series_uid: info.series_uid,
                            });
                        }
                    }
                    if !selected.is_empty() {
                        self.inputs = selected;
                    }
                }
            } else if cancel {
                self.series_picker = None;
            }
            ui.add_space(8.0);

            // --- Organ + resolution ---
            egui::Grid::new("controls").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                ui.label("ROI subset");
                let selected_text = match self.selected_organs.len() {
                    0 => "Select organ(s)…".to_string(),
                    1..=2 => self.selected_organs.join(", "),
                    n => format!("{n} organs selected"),
                };
                egui::ComboBox::from_id_salt("organ")
                    .selected_text(selected_text)
                    // A normal ComboBox closes after any item click. ROI selection
                    // is multi-select, so keep it open until the user clicks away
                    // or presses Escape.
                    .close_behavior(egui::popup::PopupCloseBehavior::CloseOnClickOutside)
                    .show_ui(ui, |ui| {
                        for o in &self.organs {
                            let mut selected = self.selected_organs.contains(o);
                            if ui.checkbox(&mut selected, o).changed() {
                                if selected {
                                    self.selected_organs.push(o.clone());
                                } else {
                                    self.selected_organs.retain(|selected| selected != o);
                                }
                            }
                        }
                    });
                ui.end_row();

                ui.label("Resolution");
                egui::ComboBox::from_id_salt("resolution")
                    .selected_text(self.resolution.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.resolution, Resolution::Fast, Resolution::Fast.label());
                        ui.selectable_value(&mut self.resolution, Resolution::Full, Resolution::Full.label());
                    });
                ui.end_row();

                ui.label("Output");
                ui.add_enabled_ui(!self.running, |ui| {
                    ui.checkbox(&mut self.save_preview, "Save thumbnail preview");
                });
                ui.end_row();
            });
            ui.add_space(10.0);

            // --- Run ---
            let can_run =
                !self.running && !self.inputs.is_empty() && !self.selected_organs.is_empty();
            ui.horizontal(|ui| {
                if ui.add_enabled(can_run, egui::Button::new("  Run  ")).clicked() {
                    self.start_run();
                }
                if self.running {
                    ui.spinner();
                }
                if !self.status.is_empty() {
                    ui.label(&self.status);
                }
            });
            ui.add_space(6.0);
            if !self.running && !self.rows.is_empty() {
                ui.label(format!("Masks written under: {}", self.exe_dir.join("output").display()));
            }

            // Per-scan problems, kept visible for the whole run.
            for note in &self.log {
                ui.colored_label(egui::Color32::from_rgb(200, 120, 0), note);
            }
            ui.separator();

            // --- Results ---
            egui::ScrollArea::vertical()
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                if self.rows.is_empty() {
                    ui.label("Results will appear here.");
                } else {
                    egui::Grid::new("results")
                        .striped(true)
                        .num_columns(5)
                        .spacing([16.0, 4.0])
                        .show(ui, |ui| {
                            ui.strong("Scan");
                            ui.strong("Organ");
                            ui.strong("Volume (mL)");
                            ui.strong("Mean HU");
                            ui.strong("Median HU");
                            ui.end_row();
                            for r in &self.rows {
                                ui.label(&r.source);
                                ui.label(&r.organ);
                                ui.label(format!("{:.1}", r.volume_ml));
                                ui.label(r.mean_hu.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into()));
                                ui.label(r.median_hu.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into()));
                                ui.end_row();
                            }
                        });
                }
            });
        });
    }
}

struct Job {
    exe_dir: PathBuf,
    inputs: Vec<InputItem>,
    organs: Vec<String>,
    resolution: Resolution,
    save_preview: bool,
}

// Runs every queued scan through the CLI and streams results back.
fn run_job(job: Job, tx: Sender<Msg>) {
    let cli = job.exe_dir.join(cli_binary_name());
    let total = job.inputs.len();

    let mut used_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut ok = 0usize;
    let mut failed = 0usize;
    let job_start = Instant::now();

    for (i, item) in job.inputs.iter().enumerate() {
        // Disambiguate scans that share a name so neither their output folders
        // nor their result rows collide.
        let base = item.base_name();
        let mut stem = base.clone();
        let mut k = 2;
        while !used_names.insert(stem.clone()) {
            stem = format!("{base} ({k})");
            k += 1;
        }
        let _ = tx.send(Msg::Status(format!("Segmenting {stem}  ({}/{total})…", i + 1)));

        // A NIfTI input is a file path; a DICOM input is its folder plus a
        // --dicom-series selector that the CLI stages to exactly that series.
        let (input_path, series_arg): (&Path, Option<&str>) = match item {
            InputItem::Nifti(p) => (p.as_path(), None),
            InputItem::Dicom { dir, series_uid, .. } => (dir.as_path(), Some(series_uid.as_str())),
        };

        let out_dir = job.exe_dir.join("output").join(&stem);
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            failed += 1;
            let _ = tx.send(Msg::Note(format!("{stem}: could not create output folder ({e}).")));
            continue;
        }

        let mut cmd = Command::new(&cli);
        cmd.arg("-i").arg(input_path).arg("-o").arg(&out_dir).arg("--device").arg("openvino");
        if let Some(uid) = series_arg {
            cmd.arg("--dicom-series").arg(uid);
        }

        // Point at the bundled ONNX Runtime / OpenVINO libraries if present, and
        // put their directory on PATH so the OpenVINO + TBB plugins load.
        if let Some(ort_lib) = ort_lib_path(&job.exe_dir) {
            cmd.arg("--ort-lib").arg(&ort_lib);
            if let Some(dir) = ort_lib.parent() {
                prepend_path(&mut cmd, dir);
            }
        }
        let models = job.exe_dir.join("models");
        if models.is_dir() {
            cmd.arg("--models-dir").arg(&models);
        }

        // `--statistics -` writes machine-readable JSON to stdout. Suppress the
        // normal progress lines so stdout contains exactly that JSON document.
        cmd.arg("--quiet").arg("--int8");
        if matches!(job.resolution, Resolution::Fast) {
            cmd.arg("--fast");
        }
        cmd.arg("--roi_subset").arg(job.organs.join(","));
        cmd.arg("--statistics").arg("-"); // print JSON to stdout
        if job.save_preview {
            // With no path argument, the CLI writes preview_<task>.png in the
            // same output directory as the generated NIfTI mask(s). Its
            // multi-label segmentation volume produces one combined overlay.
            cmd.arg("--preview");
        }

        let scan_start = Instant::now();
        let output = cmd.output();
        let scan_secs = scan_start.elapsed().as_secs_f64();
        match output {
            Ok(out) if out.status.success() => match parse_rows(&out.stdout, &stem) {
                Ok(rows) if !rows.is_empty() => {
                    ok += 1;
                    let _ = tx.send(Msg::Rows(rows));
                    let _ = tx.send(Msg::Note(format!("{stem}: finished in {scan_secs:.1} s.")));
                }
                Ok(_) => {
                    failed += 1;
                    let _ = tx.send(Msg::Note(format!(
                        "{stem}: none of the selected organs ({}) were found in this scan.",
                        job.organs.join(", ")
                    )));
                }
                Err(e) => {
                    failed += 1;
                    let _ = tx.send(Msg::Note(format!("{stem}: could not read results ({e}).")));
                }
            },
            Ok(out) => {
                failed += 1;
                let err = String::from_utf8_lossy(&out.stderr);
                let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("segmentation failed");
                let _ = tx.send(Msg::Note(format!("{stem}: {last}")));
            }
            Err(e) => {
                failed += 1;
                let _ = tx.send(Msg::Note(format!("{stem}: could not launch {} ({e}).", cli.display())));
            }
        }
    }

    let secs = job_start.elapsed().as_secs_f64();
    let timing = if total > 1 {
        format!("{secs:.1} s total, {:.1} s per scan", secs / total as f64)
    } else {
        format!("{secs:.1} s")
    };
    let summary = if failed == 0 {
        format!("Done — {ok} of {total} scan(s) succeeded in {timing}.")
    } else {
        format!("Done — {ok} of {total} succeeded in {timing}; {failed} had problems (see below).")
    };
    let _ = tx.send(Msg::Done(summary));
}

// Parse the `--statistics -` JSON: { "structures": { name: { volume_ml, intensity_mean, intensity_median } } }
fn parse_rows(stdout: &[u8], source: &str) -> Result<Vec<Row>, String> {
    // `--quiet` normally makes stdout a single JSON document. Be defensive
    // about third-party runtime messages anyway: find the first suffix that is
    // itself valid JSON instead of losing an otherwise successful result.
    let v: serde_json::Value = serde_json::from_slice(stdout).or_else(|original| {
        stdout
            .iter()
            .enumerate()
            .filter(|(_, byte)| **byte == b'{')
            .find_map(|(start, _)| serde_json::from_slice(&stdout[start..]).ok())
            .ok_or(original)
    })
    .map_err(|e| e.to_string())?;
    let structures = v
        .get("structures")
        .and_then(|s| s.as_object())
        .ok_or("no 'structures' in statistics output")?;
    let mut rows = Vec::new();
    for (name, entry) in structures {
        rows.push(Row {
            organ: name.clone(),
            volume_ml: entry.get("volume_ml").and_then(|x| x.as_f64()).unwrap_or(0.0),
            mean_hu: entry.get("intensity_mean").and_then(|x| x.as_f64()),
            median_hu: entry.get("intensity_median").and_then(|x| x.as_f64()),
            source: source.to_string(),
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::parse_rows;

    const STATS: &str = r#"{
        "n_structures": 1,
        "structures": {
            "spleen": {
                "volume_ml": 119.053,
                "intensity_mean": 87.63,
                "intensity_median": 89.0
            }
        }
    }"#;

    #[test]
    fn parses_statistics_json() {
        let rows = parse_rows(STATS.as_bytes(), "scan").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].organ, "spleen");
        assert_eq!(rows[0].source, "scan");
        assert_eq!(rows[0].volume_ml, 119.053);
    }

    #[test]
    fn tolerates_progress_text_before_statistics_json() {
        let stdout = format!("task=total  provider=OpenVINOExecutionProvider\nwrote mask.nii.gz\n{STATS}");
        let rows = parse_rows(stdout.as_bytes(), "scan").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].organ, "spleen");
    }
}

fn load_organs(exe_dir: &Path) -> Result<Vec<String>, String> {
    let path = exe_dir.join("models").join("total_label_map.json");
    if !path.exists() {
        return Err(format!(
            "Label map not found at {}. Using a built-in organ list.",
            path.display()
        ));
    }
    let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let map: std::collections::HashMap<String, u8> =
        serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut names: Vec<String> = map.into_keys().filter(|n| n != "background").collect();
    names.sort();
    Ok(names)
}

// Used only if the shipped label map is missing: the abdominal organs the paper
// validated externally on AMOS.
fn fallback_organs() -> Vec<String> {
    [
        "spleen", "liver", "kidney_left", "kidney_right", "pancreas", "gallbladder",
        "stomach", "duodenum", "aorta", "inferior_vena_cava", "esophagus",
        "urinary_bladder", "adrenal_gland_left", "adrenal_gland_right", "prostate",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn cli_binary_name() -> &'static str {
    if cfg!(windows) {
        "onnx-totalseg.exe"
    } else {
        "onnx-totalseg"
    }
}

// The bundled ONNX Runtime shared library, if present next to the executable.
fn ort_lib_path(exe_dir: &Path) -> Option<PathBuf> {
    let candidates = if cfg!(windows) {
        vec!["onnxruntime/onnxruntime.dll"]
    } else if cfg!(target_os = "macos") {
        vec!["onnxruntime/libonnxruntime.dylib"]
    } else {
        vec!["onnxruntime/libonnxruntime.so"]
    };
    candidates.into_iter().map(|c| exe_dir.join(c)).find(|p| p.exists())
}

fn prepend_path(cmd: &mut Command, dir: &Path) {
    let existing = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![dir.to_path_buf()];
    paths.extend(std::env::split_paths(&existing));
    if let Ok(joined) = std::env::join_paths(paths) {
        cmd.env("PATH", joined);
    }
}

// "scan.nii.gz" -> "scan"
fn scan_stem(p: &Path) -> String {
    let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    name.strip_suffix(".nii.gz")
        .or_else(|| name.strip_suffix(".nii"))
        .unwrap_or(&name)
        .to_string()
}

// Make a series label safe to use as a folder name.
fn sanitize(label: &str) -> String {
    let cleaned: String = label
        .chars()
        .map(|c| if c.is_control() || "/\\:*?\"<>|".contains(c) { '_' } else { c })
        .collect();
    let trimmed = cleaned.trim().trim_matches('.').trim();
    if trimmed.is_empty() { "series".to_string() } else { trimmed.to_string() }
}
