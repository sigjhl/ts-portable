use std::{
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
    process::{self, ExitCode},
    time::Instant,
};

use anyhow::{Context, Result};
use clap::{ArgAction, ArgGroup, Parser, Subcommand};
use onnx_totalseg_rs::{
    io::{self, ImageProps},
    preview,
    report::compute_statistics,
    runtime::{
        CropEmptyBehavior, DetectionSummary, ModelCard, Predictor, RuntimeOptions, Segmentator,
        build_predictor_from_card, crop_bbox_from_localizer_mask, filter_components, filter_roi,
        provider_plan_uses_gpu, read_raw_f32_volume, resolve_provider_names, write_raw_u8_volume,
    },
    server,
    tasks::totalseg::{build_predictor, build_predictor_with_full_cards, registry},
    volume::Volume,
};

#[derive(Parser, Debug)]
#[command(name = "onnx-totalseg")]
#[command(version)]
#[command(about = "TS-Portable: 8-bit integer TotalSegmentator for CT organ segmentation on the CPU (ONNX Runtime with OpenVINO)")]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    segment: SegmentArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(about = "Run one segmentation")]
    Segment(SegmentArgs),

    #[command(about = "Run the warm HTTP file-handoff server")]
    Serve(server::ServerArgs),

    #[command(hide = true)]
    FullPart(FullPartArgs),

    #[command(hide = true)]
    CropBbox(CropBboxArgs),
}

#[derive(clap::Args, Debug)]
#[command(group(
    ArgGroup::new("tier")
        .args(["fast", "fastest", "ultrafast"])
        .multiple(false)
))]
struct SegmentArgs {
    /// Input CT: a NIfTI file (.nii or .nii.gz) or a DICOM directory
    #[arg(short = 'i', long = "in")]
    input: Option<PathBuf>,

    /// Output: a directory (one <structure>.nii.gz per structure, or total.nii.gz with --ml),
    /// or a .nii.gz path for a single multilabel file
    #[arg(short = 'o', long = "out")]
    output: Option<PathBuf>,

    /// Use the 3-mm model instead of the 1.5-mm models
    #[arg(long, action = ArgAction::SetTrue)]
    fast: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    fastest: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    ultrafast: bool,

    /// Write one multilabel NIfTI (total.nii.gz) instead of one file per structure
    #[arg(long, action = ArgAction::SetTrue)]
    ml: bool,

    /// Keep one or more ROI structures. Repeat the option or use commas.
    #[arg(long = "roi_subset", alias = "roi", action = ArgAction::Append, value_delimiter = ',')]
    roi: Vec<String>,

    /// Write per-structure volume (mL) and HU statistics as JSON: to the given path,
    /// to stdout with "-", or to statistics.json beside the masks when no value is given
    #[arg(long = "statistics", alias = "report", num_args = 0..=1, default_missing_value = "<auto>")]
    statistics: Option<String>,

    #[arg(long = "detections-json", num_args = 0..=1, default_missing_value = "<auto>")]
    detections_json: Option<String>,

    #[arg(long = "min-component-volume-mm3")]
    min_component_volume_mm3: Option<f64>,

    #[arg(long = "component-connectivity")]
    component_connectivity: Option<u8>,

    /// Write a PNG contact sheet of the segmentation: to the given path, or to
    /// preview_<task>.png beside the masks when no value is given
    #[arg(long, num_args = 0..=1, default_missing_value = "<auto>")]
    preview: Option<String>,

    #[arg(long = "preview-tiles", default_value_t = preview::DEFAULT_CONTACT_SHEET_SIDE)]
    preview_tiles: usize,

    #[arg(long, default_value = "total")]
    task: String,

    #[arg(long = "abd-muscle-slab")]
    abd_muscle_slab: Option<String>,

    #[arg(long = "abd-muscle-slab-margin-mm")]
    abd_muscle_slab_margin_mm: Option<f32>,

    #[arg(long = "abdominal-bbox-json", hide = true)]
    abdominal_bbox_json: Option<PathBuf>,

    #[arg(
        long = "abdominal-vertebra-localizer",
        default_value = "fastest",
        hide = true
    )]
    abdominal_vertebra_localizer: String,

    /// Execution provider: auto (DirectML when present, else CPU), cpu, openvino
    /// (required for --int8), cuda, directml
    #[arg(long, default_value = "auto")]
    device: String,

    /// When the input is a DICOM directory holding more than one series, segment
    /// the series with this SeriesInstanceUID instead of the largest one.
    #[arg(long = "dicom-series")]
    dicom_series: Option<String>,

    #[arg(long)]
    providers: Option<String>,

    /// Number of inference threads [default: all cores]
    #[arg(long, alias = "nr_thr_resamp")]
    threads: Option<usize>,

    #[arg(long = "nr_thr_saving")]
    save_threads: Option<usize>,

    #[arg(long, action = ArgAction::SetTrue)]
    fp16: bool,

    /// Use INT8 model variants (<name>_int8.onnx); requires --device openvino.
    #[arg(long, action = ArgAction::SetTrue)]
    int8: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    fp32: bool,

    /// Suppress the progress lines on stdout (with "--statistics -", stdout is then only the JSON)
    #[arg(long, action = ArgAction::SetTrue)]
    quiet: bool,

    /// Folder holding the model cards, label map and ONNX files
    /// [default: $ONNX_TOTALSEG_MODELS, else ./models]
    #[arg(long)]
    models_dir: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    model_card: Option<PathBuf>,

    /// Path to the ONNX Runtime shared library (onnxruntime.dll, libonnxruntime.so,
    /// libonnxruntime.dylib) [default: $ORT_DYLIB_PATH, else the system library]
    #[arg(long)]
    ort_lib: Option<PathBuf>,

    #[arg(long, hide = true, action = ArgAction::SetTrue)]
    profile: bool,

    #[arg(long, hide = true)]
    model: Option<String>,

    #[arg(long, hide = true)]
    resample_order: Option<usize>,
}

#[derive(clap::Args, Debug)]
struct FullPartArgs {
    #[arg(long)]
    card: PathBuf,

    #[arg(long)]
    input_raw: PathBuf,

    #[arg(long)]
    output_raw: PathBuf,

    #[arg(long, value_parser = parse_shape_arg)]
    shape: [usize; 3],

    #[arg(long, default_value = "auto")]
    device: String,

    #[arg(long)]
    providers: Option<String>,

    #[arg(long, alias = "nr_thr_resamp")]
    threads: Option<usize>,

    #[arg(long, action = ArgAction::SetTrue)]
    fp16: bool,

    /// Use INT8 model variants (<name>_int8.onnx); requires --device openvino.
    #[arg(long, action = ArgAction::SetTrue)]
    int8: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    fp32: bool,

    #[arg(long)]
    ort_lib: Option<PathBuf>,

    #[arg(long, hide = true, action = ArgAction::SetTrue)]
    cuda_low_memory: bool,

    #[arg(long, hide = true)]
    profile: bool,

    #[arg(long, hide = true, action = ArgAction::SetTrue)]
    input_cropped: bool,
}

#[derive(clap::Args, Debug)]
struct CropBboxArgs {
    #[arg(short = 'i', long = "in")]
    input: PathBuf,

    #[arg(short = 'o', long = "out")]
    output: PathBuf,

    #[arg(long)]
    model: String,

    #[arg(long)]
    crop_ids: String,

    #[arg(long)]
    margin_mm: f32,

    #[arg(long, default_value = "auto")]
    device: String,

    #[arg(long)]
    providers: Option<String>,

    #[arg(long, alias = "nr_thr_resamp")]
    threads: Option<usize>,

    #[arg(long, action = ArgAction::SetTrue)]
    fp16: bool,

    /// Use INT8 model variants (<name>_int8.onnx); requires --device openvino.
    #[arg(long, action = ArgAction::SetTrue)]
    int8: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    fp32: bool,

    #[arg(long)]
    models_dir: Option<PathBuf>,

    #[arg(long)]
    ort_lib: Option<PathBuf>,

    #[arg(long, hide = true)]
    resample_order: Option<usize>,

    #[arg(long, hide = true)]
    profile: bool,
}

#[derive(Debug)]
struct UserError(String);

impl fmt::Display for UserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for UserError {}

fn main() -> ExitCode {
    let Cli { command, segment } = Cli::parse();
    let result = match command {
        Some(Command::Segment(args)) => run(args),
        Some(Command::Serve(args)) => server::run(args),
        Some(Command::FullPart(args)) => run_full_part(args),
        Some(Command::CropBbox(args)) => run_crop_bbox(args),
        None => run(segment),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) if err.downcast_ref::<UserError>().is_some() => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
        Err(err) => {
            eprintln!("Error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_full_part(args: FullPartArgs) -> Result<()> {
    let t0 = Instant::now();
    let providers = args.providers.as_ref().map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    });
    let effective =
        resolve_provider_names(providers.as_deref(), &args.device, args.ort_lib.clone())?;
    let fp16 = (args.fp16 || provider_plan_uses_gpu(&effective)) && !args.fp32;
    let opts = RuntimeOptions {
        providers: Some(effective),
        device: args.device.clone(),
        threads: args.threads,
        fp16,
        int8: args.int8,
        ort_lib: args.ort_lib.clone(),
        profile: args.profile,
        low_memory_cpu: false,
        cuda_low_memory: args.cuda_low_memory,
        lazy_session: false,
    };
    if args.profile {
        eprintln!(
            "[profile] worker resolve_providers: {:.3}s",
            t0.elapsed().as_secs_f64()
        );
    }

    let t_read = Instant::now();
    let data = read_raw_f32_volume(&args.input_raw, args.shape)?;
    if args.profile {
        eprintln!(
            "[profile] worker read_raw: {:.3}s",
            t_read.elapsed().as_secs_f64()
        );
    }
    let t_session = Instant::now();
    let mut predictor = Segmentator::new(&args.card, &opts, None)?;
    if args.profile {
        eprintln!(
            "[profile] worker build_session: {:.3}s",
            t_session.elapsed().as_secs_f64()
        );
    }
    let target_spacing = predictor.target_spacing();
    let seg = if args.input_cropped {
        predictor.predict_cropped_array(data, target_spacing)?
    } else {
        predictor.predict_array(&data, target_spacing)?
    };
    let t_write = Instant::now();
    write_raw_u8_volume(&args.output_raw, &seg)?;
    if args.profile {
        eprintln!(
            "[profile] worker write_raw: {:.3}s",
            t_write.elapsed().as_secs_f64()
        );
    }

    // Avoid CUDA ONNX Runtime teardown in the short-lived worker. The parent
    // merges the output and the OS releases GPU memory when this process exits.
    std::mem::forget(predictor);
    process::exit(0)
}

// Decide which DICOM series to segment from a (possibly multi-series) input.
// Repoints `input_path` at a temp directory containing only the chosen series
// and returns the guard that keeps it alive; returns None (leaving the largest
// series to be picked by the reader) for non-directory input, a single series,
// or a non-interactive session with no explicit selection.
fn select_dicom_series(
    input_path: &mut PathBuf,
    explicit: Option<&str>,
) -> Result<Option<tempfile::TempDir>> {
    use onnx_totalseg_rs::dicom_io;
    use std::io::IsTerminal;

    if !input_path.is_dir() {
        return Ok(None);
    }
    // An explicit --dicom-series always wins (used by the GUI and by scripts).
    if let Some(uid) = explicit {
        let staged = dicom_io::stage_series_from_dir(input_path, uid)?;
        *input_path = staged.path().to_path_buf();
        return Ok(Some(staged));
    }
    let series = dicom_io::list_dicom_series(input_path)?;
    // Zero or one series: nothing to choose; let the reader handle it.
    if series.len() <= 1 {
        return Ok(None);
    }
    // Not an interactive terminal (piped, or spawned as a subprocess): keep the
    // default largest-series behavior rather than blocking on a prompt.
    if !std::io::stdin().is_terminal() {
        return Ok(None);
    }
    let choice = prompt_series_choice(&series)?;
    let staged = dicom_io::stage_files(&series[choice].files)?;
    *input_path = staged.path().to_path_buf();
    Ok(Some(staged))
}

// Print the series as a numbered list and read the user's choice from the
// terminal. Returns a 0-based index; an empty or unparseable reply selects the
// first (largest) series.
fn prompt_series_choice(series: &[onnx_totalseg_rs::dicom_io::SeriesInfo]) -> Result<usize> {
    use std::io::Write;
    eprintln!("\nMultiple DICOM series found — select one to segment:");
    for (i, s) in series.iter().enumerate() {
        let mut parts = vec![s.patient_display()];
        for p in [s.protocol.as_str(), s.description.as_str(), s.modality.as_str()] {
            if !p.trim().is_empty() {
                parts.push(p.trim().to_string());
            }
        }
        parts.push(format!("{} slices", s.slice_count()));
        eprintln!("  {}. {}", i + 1, parts.join(" · "));
    }
    eprint!("Enter number [1-{}] (default 1): ", series.len());
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let choice = line.trim().parse::<usize>().unwrap_or(1).clamp(1, series.len());
    Ok(choice - 1)
}

fn run_crop_bbox(args: CropBboxArgs) -> Result<()> {
    let providers = args.providers.as_ref().map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    });
    let effective =
        resolve_provider_names(providers.as_deref(), &args.device, args.ort_lib.clone())?;
    let is_gpu = provider_plan_uses_gpu(&effective);
    let fp16 = (args.fp16 || is_gpu) && !args.fp32;
    let models_dir = registry::models_dir(args.models_dir.as_deref());
    let opts = RuntimeOptions {
        providers: Some(effective),
        device: args.device.clone(),
        threads: args.threads,
        fp16,
        int8: args.int8,
        ort_lib: args.ort_lib.clone(),
        profile: args.profile,
        low_memory_cpu: registry::low_memory_cpu_model(&args.model) && !is_gpu,
        cuda_low_memory: false,
        lazy_session: false,
    };

    let (input_path, _tmp) = io::collect_nifti_or_dicom_input(&args.input)?;
    let (data, props) = io::read_image(&input_path)?;
    let mut localizer = build_predictor(&args.model, &models_dir, opts, args.resample_order)?;
    let (loc, _) = localizer.predict_loaded(&data, &props, false)?;
    let bbox = crop_bbox_from_localizer_mask(
        &loc,
        &parse_label_ids(&args.crop_ids).map_err(user_error)?,
        data.shape,
        props.spacing_zyx,
        args.margin_mm,
    );
    let text = serde_json::to_string(&bbox)?;
    fs::write(&args.output, text)
        .with_context(|| format!("failed to write {}", args.output.display()))?;

    std::mem::forget(localizer);
    process::exit(0)
}

struct CropBboxWorkerRun<'a> {
    input_path: &'a Path,
    models_dir: &'a Path,
    localizer_model: &'a str,
    crop_ids: &'a [u8],
    margin_mm: f32,
    resample_order: Option<usize>,
    args: &'a SegmentArgs,
}

fn crop_bbox_worker(run: CropBboxWorkerRun<'_>) -> Result<Option<[[usize; 2]; 3]>> {
    let temp = tempfile::Builder::new()
        .prefix("onnx_totalseg_crop_bbox_")
        .tempdir()?;
    let bbox_path = temp.path().join("bbox.json");
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("crop-bbox")
        .arg("-i")
        .arg(run.input_path)
        .arg("-o")
        .arg(&bbox_path)
        .arg("--model")
        .arg(run.localizer_model)
        .arg("--crop-ids")
        .arg(
            run.crop_ids
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )
        .arg("--margin-mm")
        .arg(run.margin_mm.to_string())
        .arg("--device")
        .arg(&run.args.device)
        .arg("--models-dir")
        .arg(run.models_dir);
    if let Some(providers) = &run.args.providers {
        cmd.arg("--providers").arg(providers);
    }
    if let Some(threads) = run.args.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }
    if run.args.fp16 {
        cmd.arg("--fp16");
    }
    if run.args.int8 {
        cmd.arg("--int8");
    }
    if run.args.fp32 {
        cmd.arg("--fp32");
    }
    if let Some(ort_lib) = &run.args.ort_lib {
        cmd.arg("--ort-lib").arg(ort_lib);
    }
    if let Some(order) = run.resample_order {
        cmd.arg("--resample-order").arg(order.to_string());
    }
    if run.args.profile {
        cmd.arg("--profile");
    }
    let output = cmd.output().context("failed to launch crop-bbox worker")?;
    if run.args.profile {
        eprint!("{}", String::from_utf8_lossy(&output.stdout));
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    if !output.status.success() {
        anyhow::bail!(
            "crop-bbox worker failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let bbox_text = fs::read_to_string(&bbox_path)
        .with_context(|| format!("failed to read {}", bbox_path.display()))?;
    serde_json::from_str(&bbox_text).context("failed to parse crop-bbox worker output")
}

struct CropWorkerRun<'a> {
    input_path: &'a Path,
    data: &'a Volume<f32>,
    props: &'a ImageProps,
    target: &'a mut onnx_totalseg_rs::runtime::Predictor,
    models_dir: &'a Path,
    localizer_model: &'a str,
    crop_ids: &'a [u8],
    margin_mm: f32,
    localizer_resample_order: Option<usize>,
    output_roi_ids: Option<&'a [u8]>,
    empty_behavior: CropEmptyBehavior,
    args: &'a SegmentArgs,
}

fn crop_with_localizer_worker(run: CropWorkerRun<'_>) -> Result<Volume<u8>> {
    let bbox = crop_bbox_worker(CropBboxWorkerRun {
        input_path: run.input_path,
        models_dir: run.models_dir,
        localizer_model: run.localizer_model,
        crop_ids: run.crop_ids,
        margin_mm: run.margin_mm,
        resample_order: run.localizer_resample_order,
        args: run.args,
    })?;
    let Some(bbox) = bbox else {
        if matches!(run.empty_behavior, CropEmptyBehavior::ReturnEmpty) {
            return Ok(Volume::<u8>::zeros(run.data.shape));
        }
        let (mut seg, _) = run.target.predict_loaded(run.data, run.props, false)?;
        if let Some(ids) = run.output_roi_ids {
            filter_roi(&mut seg, ids);
        }
        return Ok(seg);
    };

    let cropped = run.data.crop(&bbox)?;
    let (cseg, _) = run.target.predict_loaded(&cropped, run.props, false)?;
    let mut seg = Volume::<u8>::zeros(run.data.shape);
    seg.paste(&cseg, &bbox);
    if let Some(ids) = run.output_roi_ids {
        filter_roi(&mut seg, ids);
    }
    Ok(seg)
}

struct CropReleaseRun<'a> {
    data: &'a Volume<f32>,
    props: &'a ImageProps,
    target: &'a mut Predictor,
    localizer: Predictor,
    crop_ids: &'a [u8],
    margin_mm: f32,
    output_roi_ids: Option<&'a [u8]>,
    empty_behavior: CropEmptyBehavior,
}

fn crop_with_localizer_then_release(mut run: CropReleaseRun<'_>) -> Result<Volume<u8>> {
    let (loc, _) = run.localizer.predict_loaded(run.data, run.props, false)?;
    let bbox = crop_bbox_from_localizer_mask(
        &loc,
        run.crop_ids,
        run.data.shape,
        run.props.spacing_zyx,
        run.margin_mm,
    );
    drop(loc);
    drop(run.localizer);

    let Some(bbox) = bbox else {
        if matches!(run.empty_behavior, CropEmptyBehavior::ReturnEmpty) {
            return Ok(Volume::<u8>::zeros(run.data.shape));
        }
        let (mut seg, _) = run.target.predict_loaded(run.data, run.props, false)?;
        if let Some(ids) = run.output_roi_ids {
            filter_roi(&mut seg, ids);
        }
        return Ok(seg);
    };

    let cropped = run.data.crop(&bbox)?;
    let (cseg, _) = run.target.predict_loaded(&cropped, run.props, false)?;
    let mut seg = Volume::<u8>::zeros(run.data.shape);
    seg.paste(&cseg, &bbox);
    if let Some(ids) = run.output_roi_ids {
        filter_roi(&mut seg, ids);
    }
    Ok(seg)
}

fn combine_slab_z_bbox(
    mut body_bbox: [[usize; 2]; 3],
    vertebra_bbox: Option<[[usize; 2]; 3]>,
) -> [[usize; 2]; 3] {
    if let Some(vertebra_bbox) = vertebra_bbox {
        body_bbox[0] = vertebra_bbox[0];
    } else {
        eprintln!(
            "[crop] WARNING: could not localize requested vertebra slab; using body_trunc z range"
        );
    }
    body_bbox
}

fn parse_abd_muscle_slab(spec: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for part in spec
        .split([':', '-'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        out.push(vertebra_label_name(part).map_err(user_error)?);
    }
    if out.is_empty() || out.len() > 2 {
        return Err(user_error(format!(
            "invalid --abd-muscle-slab '{spec}', expected LEVEL or LEVEL:LEVEL"
        )));
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn vertebra_label_name(level: &str) -> std::result::Result<String, String> {
    let level = level.trim();
    let Some(prefix) = level.chars().next() else {
        return Err("empty vertebra level".to_string());
    };
    let prefix = prefix.to_ascii_uppercase();
    let number = level[1..]
        .parse::<u8>()
        .map_err(|_| format!("invalid vertebra level '{level}'"))?;
    let valid = match prefix {
        'C' => (1..=7).contains(&number),
        'T' => (1..=12).contains(&number),
        'L' => (1..=5).contains(&number),
        'S' => number == 1,
        _ => false,
    };
    if !valid {
        return Err(format!("invalid vertebra level '{level}'"));
    }
    Ok(format!("vertebrae_{prefix}{number}"))
}

fn default_abd_muscle_slab_margin_mm(spec: &str) -> f32 {
    if spec.contains(':') || spec.contains('-') {
        20.0
    } else {
        10.0
    }
}

fn paste_bbox_prediction(
    data: &Volume<f32>,
    props: &ImageProps,
    target: &mut Predictor,
    bbox: &[[usize; 2]; 3],
    output_roi_ids: Option<&[u8]>,
) -> Result<Volume<u8>> {
    let cropped = data.crop(bbox)?;
    let (cseg, _) = target.predict_loaded(&cropped, props, false)?;
    let mut seg = Volume::<u8>::zeros(data.shape);
    seg.paste(&cseg, bbox);
    if let Some(ids) = output_roi_ids {
        filter_roi(&mut seg, ids);
    }
    Ok(seg)
}

struct AbdominalSlabWorkerRun<'a> {
    input_path: &'a Path,
    data: &'a Volume<f32>,
    props: &'a ImageProps,
    target: &'a mut Predictor,
    models_dir: &'a Path,
    body_ids: &'a [u8],
    vertebra_model: &'a str,
    vertebra_ids: &'a [u8],
    body_margin_mm: f32,
    vertebra_margin_mm: f32,
    output_roi_ids: Option<&'a [u8]>,
    empty_behavior: CropEmptyBehavior,
    args: &'a SegmentArgs,
}

fn crop_abdominal_slab_worker(run: AbdominalSlabWorkerRun<'_>) -> Result<Volume<u8>> {
    let body_bbox = crop_bbox_worker(CropBboxWorkerRun {
        input_path: run.input_path,
        models_dir: run.models_dir,
        localizer_model: "body",
        crop_ids: run.body_ids,
        margin_mm: run.body_margin_mm,
        resample_order: None,
        args: run.args,
    })?;
    let Some(body_bbox) = body_bbox else {
        if matches!(run.empty_behavior, CropEmptyBehavior::ReturnEmpty) {
            return Ok(Volume::<u8>::zeros(run.data.shape));
        }
        let (mut seg, _) = run.target.predict_loaded(run.data, run.props, false)?;
        if let Some(ids) = run.output_roi_ids {
            filter_roi(&mut seg, ids);
        }
        return Ok(seg);
    };

    let vertebra_bbox = crop_bbox_worker(CropBboxWorkerRun {
        input_path: run.input_path,
        models_dir: run.models_dir,
        localizer_model: run.vertebra_model,
        crop_ids: run.vertebra_ids,
        margin_mm: run.vertebra_margin_mm,
        resample_order: None,
        args: run.args,
    })?;
    let bbox = combine_slab_z_bbox(body_bbox, vertebra_bbox);
    if run.args.profile {
        eprintln!("[profile] abdominal_slab_bbox: {bbox:?}");
    }
    paste_bbox_prediction(run.data, run.props, run.target, &bbox, run.output_roi_ids)
}

struct AbdominalSlabReleaseRun<'a> {
    data: &'a Volume<f32>,
    props: &'a ImageProps,
    target: &'a mut Predictor,
    body_localizer: Predictor,
    vertebra_localizer: Predictor,
    body_ids: &'a [u8],
    vertebra_ids: &'a [u8],
    body_margin_mm: f32,
    vertebra_margin_mm: f32,
    output_roi_ids: Option<&'a [u8]>,
    empty_behavior: CropEmptyBehavior,
    profile: bool,
}

fn bbox_with_localizer(
    localizer: &mut Predictor,
    data: &Volume<f32>,
    props: &ImageProps,
    crop_ids: &[u8],
    margin_mm: f32,
) -> Result<Option<[[usize; 2]; 3]>> {
    let (loc, _) = localizer.predict_loaded(data, props, false)?;
    Ok(crop_bbox_from_localizer_mask(
        &loc,
        crop_ids,
        data.shape,
        props.spacing_zyx,
        margin_mm,
    ))
}

fn crop_abdominal_slab_then_release(mut run: AbdominalSlabReleaseRun<'_>) -> Result<Volume<u8>> {
    let body_bbox = bbox_with_localizer(
        &mut run.body_localizer,
        run.data,
        run.props,
        run.body_ids,
        run.body_margin_mm,
    )?;
    drop(run.body_localizer);
    let Some(body_bbox) = body_bbox else {
        if matches!(run.empty_behavior, CropEmptyBehavior::ReturnEmpty) {
            return Ok(Volume::<u8>::zeros(run.data.shape));
        }
        let (mut seg, _) = run.target.predict_loaded(run.data, run.props, false)?;
        if let Some(ids) = run.output_roi_ids {
            filter_roi(&mut seg, ids);
        }
        return Ok(seg);
    };

    let vertebra_bbox = bbox_with_localizer(
        &mut run.vertebra_localizer,
        run.data,
        run.props,
        run.vertebra_ids,
        run.vertebra_margin_mm,
    )?;
    drop(run.vertebra_localizer);
    let bbox = combine_slab_z_bbox(body_bbox, vertebra_bbox);
    if run.profile {
        eprintln!("[profile] abdominal_slab_bbox: {bbox:?}");
    }
    paste_bbox_prediction(run.data, run.props, run.target, &bbox, run.output_roi_ids)
}

fn runtime_options_for_model(base: &RuntimeOptions, model: &str, is_gpu: bool) -> RuntimeOptions {
    let mut opts = base.clone();
    opts.low_memory_cpu = registry::low_memory_cpu_model(model) && !is_gpu;
    opts
}

fn runtime_options_for_crop_localizer(
    base: &RuntimeOptions,
    crop: registry::CropSpec,
    is_gpu: bool,
) -> RuntimeOptions {
    let mut opts = runtime_options_for_model(base, crop.localizer_model, is_gpu);
    if crop.localizer_low_memory_cpu && !is_gpu {
        opts.low_memory_cpu = true;
    }
    opts
}

fn run(args: SegmentArgs) -> Result<()> {
    let input = args
        .input
        .as_ref()
        .ok_or_else(|| user_error("--in <PATH> is required for segmentation".to_string()))?;
    let output = args
        .output
        .as_ref()
        .ok_or_else(|| user_error("--out <PATH> is required for segmentation".to_string()))?;
    if !input.exists() {
        return Err(user_error(format!("input not found: {}", input.display())));
    }
    if args.preview.is_some()
        && !(1..=preview::MAX_CONTACT_SHEET_SIDE).contains(&args.preview_tiles)
    {
        return Err(user_error(format!(
            "--preview-tiles must be between 1 and {}",
            preview::MAX_CONTACT_SHEET_SIDE
        )));
    }
    let models_dir = registry::models_dir(args.models_dir.as_deref());
    let tier = tier(&args);
    if args.model_card.is_some()
        && (args.task != "total"
            || args.fast
            || args.fastest
            || args.ultrafast
            || args.model.is_some()
            || !args.roi.is_empty()
            || args.abd_muscle_slab.is_some()
            || args.abd_muscle_slab_margin_mm.is_some()
            || args.abdominal_bbox_json.is_some())
    {
        return Err(user_error(
            "--model-card cannot be combined with TotalSegmentator task, tier, ROI, crop, or hidden model overrides"
                .to_string(),
        ));
    }
    let external_card = args
        .model_card
        .as_deref()
        .map(ModelCard::read)
        .transpose()?;
    let task_model = if args.model_card.is_some() {
        registry::TaskModel {
            model: "external",
            resample_order: None,
            lowres: false,
            crop: None,
            output_roi_names: None,
        }
    } else {
        registry::resolve_task_model(&args.task, tier).map_err(|e| UserError(e.to_string()))?
    };
    let abd_muscle_slab_labels = match &args.abd_muscle_slab {
        Some(spec) => {
            if args.task != "abdominal_muscles" {
                return Err(user_error(
                    "--abd-muscle-slab is only valid with --task abdominal_muscles".to_string(),
                ));
            }
            Some(parse_abd_muscle_slab(spec)?)
        }
        None => None,
    };
    let abd_muscle_slab_margin_mm = match (
        args.abd_muscle_slab_margin_mm,
        args.abd_muscle_slab.as_deref(),
    ) {
        (Some(margin), _) if margin < 0.0 => {
            return Err(user_error(
                "--abd-muscle-slab-margin-mm must be non-negative".to_string(),
            ));
        }
        (Some(margin), _) => Some(margin),
        (None, Some(spec)) => Some(default_abd_muscle_slab_margin_mm(spec)),
        (None, None) => None,
    };
    if args.abd_muscle_slab_margin_mm.is_some() && args.abd_muscle_slab.is_none() {
        return Err(user_error(
            "--abd-muscle-slab-margin-mm requires --abd-muscle-slab".to_string(),
        ));
    }
    if args.abdominal_bbox_json.is_some() && args.task != "abdominal_muscles" {
        return Err(user_error(
            "--abdominal-bbox-json is only valid with --task abdominal_muscles".to_string(),
        ));
    }
    if !matches!(
        args.abdominal_vertebra_localizer.as_str(),
        "fastest" | "fast"
    ) {
        return Err(user_error(
            "--abdominal-vertebra-localizer must be fastest or fast".to_string(),
        ));
    }
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| task_model.model.to_string());
    let resample_order = args.resample_order.or(task_model.resample_order);
    let lowres = task_model.lowres;
    let providers = args.providers.as_ref().map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    });
    let effective =
        resolve_provider_names(providers.as_deref(), &args.device, args.ort_lib.clone())?;
    let is_gpu = provider_plan_uses_gpu(&effective);
    let fp16 = (args.fp16 || is_gpu) && !args.fp32;
    if args.fp16 && !is_gpu {
        eprintln!("[fp16] no GPU provider requested; running fp32");
    }
    let opts = RuntimeOptions {
        providers: Some(effective),
        device: args.device.clone(),
        threads: args.threads,
        fp16,
        int8: args.int8,
        ort_lib: args.ort_lib.clone(),
        profile: args.profile,
        low_memory_cpu: registry::low_memory_cpu_model(&model) && !is_gpu,
        cuda_low_memory: false,
        lazy_session: false,
    };
    let roi_ids = if args.roi.is_empty() {
        None
    } else {
        Some(
            registry::resolve_roi_for_model(&models_dir, &model, &args.roi.join(","))
                .map_err(|e| UserError(e.to_string()))?,
        )
    };
    let task_output_roi_ids = match (roi_ids.is_none(), task_model.output_roi_names) {
        (true, Some(names)) => Some(
            registry::resolve_labels_for_model(&models_dir, &model, names)
                .map_err(|e| UserError(e.to_string()))?,
        ),
        _ => None,
    };
    let output_roi_ids = roi_ids.as_deref().or(task_output_roi_ids.as_deref());
    let task_crop_ids = match task_model.crop {
        Some(crop) => Some(
            registry::resolve_labels_for_model(&models_dir, crop.localizer_model, crop.roi_names)
                .map_err(|e| UserError(e.to_string()))?,
        ),
        None => None,
    };
    let abdominal_vertebra_ids = if let Some(labels) = &abd_muscle_slab_labels {
        let refs = labels.iter().map(String::as_str).collect::<Vec<_>>();
        Some(
            registry::resolve_labels_for_model(
                &models_dir,
                &args.abdominal_vertebra_localizer,
                &refs,
            )
            .map_err(|e| UserError(e.to_string()))?,
        )
    } else {
        None
    };
    let use_roi_cascade =
        args.task == "total" && roi_ids.is_some() && matches!(tier, "full" | "fast");
    let use_task_crop = task_model.crop.is_some();
    let use_cascade = use_roi_cascade || use_task_crop;
    let use_abd_muscle_slab = abd_muscle_slab_labels.is_some();
    let use_cached_abdominal_bbox = args.abdominal_bbox_json.is_some();
    let use_crop_worker = use_task_crop && is_gpu && !use_cached_abdominal_bbox;
    let save_threads = args.save_threads.unwrap_or_else(default_save_threads);

    let t0 = Instant::now();
    let mut target_opts = runtime_options_for_model(&opts, &model, is_gpu);
    target_opts.lazy_session = use_task_crop;
    target_opts.cuda_low_memory =
        is_gpu && ((use_roi_cascade && model == "full") || args.model_card.is_some());
    let full_roi_cards = if use_roi_cascade && model == "full" {
        Some(
            registry::full_cards_for_roi(
                &models_dir,
                roi_ids.as_deref().expect("ROI ids exist for cascade"),
            )
            .map_err(|e| UserError(e.to_string()))?,
        )
    } else {
        None
    };
    let mut predictor = if let Some(card_path) = &args.model_card {
        build_predictor_from_card(card_path, target_opts, resample_order)?
    } else {
        build_predictor_with_full_cards(
            &model,
            &models_dir,
            target_opts,
            resample_order,
            full_roi_cards,
        )?
    };
    let mut localizer = if let Some(crop) = task_model.crop {
        if use_abd_muscle_slab || use_crop_worker || use_cached_abdominal_bbox {
            None
        } else {
            Some(build_predictor(
                crop.localizer_model,
                &models_dir,
                runtime_options_for_crop_localizer(&opts, crop, is_gpu),
                None,
            )?)
        }
    } else if use_roi_cascade && !is_gpu {
        Some(build_predictor(
            "fastest",
            &models_dir,
            runtime_options_for_model(&opts, "fastest", is_gpu),
            Some(1),
        )?)
    } else {
        None
    };
    let mut abdominal_body_localizer = None;
    let mut abdominal_vertebra_localizer = None;
    if use_abd_muscle_slab && !use_crop_worker && !use_cached_abdominal_bbox {
        let crop = task_model.crop.expect("abdominal crop requires task crop");
        abdominal_body_localizer = Some(build_predictor(
            crop.localizer_model,
            &models_dir,
            runtime_options_for_crop_localizer(&opts, crop, is_gpu),
            None,
        )?);
        abdominal_vertebra_localizer = Some(build_predictor(
            &args.abdominal_vertebra_localizer,
            &models_dir,
            runtime_options_for_model(&opts, &args.abdominal_vertebra_localizer, is_gpu),
            None,
        )?);
    }
    let cold = t0.elapsed();
    let say = |msg: String| {
        if !args.quiet {
            println!("{msg}");
        }
    };
    say(format!(
        "{}  provider={}{}{}{}{}",
        args.model_card
            .as_ref()
            .map(|path| format!("model_card={}", path.display()))
            .unwrap_or_else(|| format!("task={}  tier={tier}", args.task)),
        predictor.active_provider(),
        if args.roi.is_empty() {
            String::new()
        } else {
            format!("  roi={}", args.roi.join(","))
        },
        if let Some(spec) = &args.abd_muscle_slab {
            format!(
                "  abd_muscle_slab={spec}({:.1}mm)",
                abd_muscle_slab_margin_mm.expect("slab margin exists")
            )
        } else if args.abdominal_bbox_json.is_some() {
            "  abdominal_crop=bbox".to_string()
        } else {
            String::new()
        },
        if lowres { "  (6mm output)" } else { "" },
        if use_cascade { "  (crop)" } else { "" }
    ));

    let t1 = Instant::now();
    let (mut input_path, _tmp) = io::collect_nifti_or_dicom_input(input)?;
    // Choose which DICOM series to segment when the input is a multi-series
    // directory. `input_path` is repointed at a temp directory holding only the
    // chosen series, so both this read and any crop-cascade subprocess (which
    // re-reads `input_path`) load exactly it. The guard holds the temp directory
    // for the rest of this function.
    let _series_tmp = select_dicom_series(&mut input_path, args.dicom_series.as_deref())?;
    let (data, props) =
        io::read_image_with_orientation(&input_path, predictor.preserves_input_orientation())?;
    let wants_statistics = args.statistics.is_some();
    let wants_preview = args.preview.is_some();
    let wants_input_retained = wants_statistics || wants_preview;
    if args.profile {
        eprintln!(
            "[profile] cli read_image: {:.3}s",
            t1.elapsed().as_secs_f64()
        );
    }
    let t_predict = Instant::now();
    let (mut seg, canonical_affine, retained_data) = if let Some(crop) = task_model.crop {
        let ids = task_crop_ids
            .as_deref()
            .expect("task crop requires crop ids");
        if let Some(path) = &args.abdominal_bbox_json {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let bbox: [[usize; 2]; 3] =
                serde_json::from_str(&text).context("failed to parse abdominal bbox json")?;
            (
                paste_bbox_prediction(&data, &props, &mut predictor, &bbox, output_roi_ids)?,
                None,
                wants_input_retained.then_some(data),
            )
        } else if use_abd_muscle_slab && use_crop_worker {
            (
                crop_abdominal_slab_worker(AbdominalSlabWorkerRun {
                    input_path: &input_path,
                    data: &data,
                    props: &props,
                    target: &mut predictor,
                    models_dir: &models_dir,
                    body_ids: ids,
                    vertebra_model: &args.abdominal_vertebra_localizer,
                    vertebra_ids: abdominal_vertebra_ids
                        .as_deref()
                        .expect("abdominal crop requires vertebra ids"),
                    body_margin_mm: crop.margin_mm,
                    vertebra_margin_mm: abd_muscle_slab_margin_mm.expect("slab margin exists"),
                    output_roi_ids,
                    empty_behavior: if crop.empty_is_empty {
                        CropEmptyBehavior::ReturnEmpty
                    } else {
                        CropEmptyBehavior::RunTargetFull
                    },
                    args: &args,
                })?,
                None,
                wants_input_retained.then_some(data),
            )
        } else if use_abd_muscle_slab {
            (
                crop_abdominal_slab_then_release(AbdominalSlabReleaseRun {
                    data: &data,
                    props: &props,
                    target: &mut predictor,
                    body_localizer: abdominal_body_localizer
                        .take()
                        .expect("abdominal body localizer was built"),
                    vertebra_localizer: abdominal_vertebra_localizer
                        .take()
                        .expect("abdominal vertebra localizer was built"),
                    body_ids: ids,
                    vertebra_ids: abdominal_vertebra_ids
                        .as_deref()
                        .expect("abdominal crop requires vertebra ids"),
                    body_margin_mm: crop.margin_mm,
                    vertebra_margin_mm: abd_muscle_slab_margin_mm.expect("slab margin exists"),
                    output_roi_ids,
                    empty_behavior: if crop.empty_is_empty {
                        CropEmptyBehavior::ReturnEmpty
                    } else {
                        CropEmptyBehavior::RunTargetFull
                    },
                    profile: args.profile,
                })?,
                None,
                wants_input_retained.then_some(data),
            )
        } else if use_crop_worker {
            (
                crop_with_localizer_worker(CropWorkerRun {
                    input_path: &input_path,
                    data: &data,
                    props: &props,
                    target: &mut predictor,
                    models_dir: &models_dir,
                    localizer_model: crop.localizer_model,
                    crop_ids: ids,
                    margin_mm: crop.margin_mm,
                    localizer_resample_order: None,
                    output_roi_ids,
                    empty_behavior: if crop.empty_is_empty {
                        CropEmptyBehavior::ReturnEmpty
                    } else {
                        CropEmptyBehavior::RunTargetFull
                    },
                    args: &args,
                })?,
                None,
                wants_input_retained.then_some(data),
            )
        } else {
            let localizer = localizer.take().expect("crop localizer was built");
            (
                crop_with_localizer_then_release(CropReleaseRun {
                    data: &data,
                    props: &props,
                    target: &mut predictor,
                    localizer,
                    crop_ids: ids,
                    margin_mm: crop.margin_mm,
                    output_roi_ids,
                    empty_behavior: if crop.empty_is_empty {
                        CropEmptyBehavior::ReturnEmpty
                    } else {
                        CropEmptyBehavior::RunTargetFull
                    },
                })?,
                None,
                wants_input_retained.then_some(data),
            )
        }
    } else if use_roi_cascade {
        let ids = roi_ids.as_deref().expect("cascade requires ROI ids");
        if is_gpu {
            (
                crop_with_localizer_worker(CropWorkerRun {
                    input_path: &input_path,
                    data: &data,
                    props: &props,
                    target: &mut predictor,
                    models_dir: &models_dir,
                    localizer_model: "fastest",
                    crop_ids: ids,
                    margin_mm: 20.0,
                    localizer_resample_order: Some(1),
                    output_roi_ids: Some(ids),
                    empty_behavior: CropEmptyBehavior::RunTargetFull,
                    args: &args,
                })?,
                None,
                wants_input_retained.then_some(data),
            )
        } else {
            let localizer = localizer.take().expect("cascade localizer was built");
            (
                crop_with_localizer_then_release(CropReleaseRun {
                    data: &data,
                    props: &props,
                    target: &mut predictor,
                    localizer,
                    crop_ids: ids,
                    margin_mm: 20.0,
                    output_roi_ids: Some(ids),
                    empty_behavior: CropEmptyBehavior::RunTargetFull,
                })?,
                None,
                wants_input_retained.then_some(data),
            )
        }
    } else if wants_input_retained {
        let (mut seg, canonical_affine) = predictor.predict_loaded(&data, &props, lowres)?;
        if let Some(ids) = output_roi_ids {
            filter_roi(&mut seg, ids);
        }
        (seg, canonical_affine, Some(data))
    } else {
        let (mut seg, canonical_affine) = predictor.predict_loaded_owned(data, &props, lowres)?;
        if let Some(ids) = output_roi_ids {
            filter_roi(&mut seg, ids);
        }
        (seg, canonical_affine, None)
    };
    if args.profile {
        eprintln!(
            "[profile] cli predict_model: {:.3}s",
            t_predict.elapsed().as_secs_f64()
        );
    }

    let card_postprocessing = external_card
        .as_ref()
        .and_then(|card| card.postprocessing.as_ref());
    let min_component_volume_mm3 = args.min_component_volume_mm3.or_else(|| {
        card_postprocessing
            .and_then(|post| post.min_component_volume_mm3)
            .map(f64::from)
    });
    let component_connectivity = args
        .component_connectivity
        .or_else(|| card_postprocessing.and_then(|post| post.connectivity));
    let detection_summary: Option<DetectionSummary> =
        if min_component_volume_mm3.is_some() || args.detections_json.is_some() {
            let spacing = if lowres {
                predictor.target_spacing()
            } else {
                props.spacing_zyx
            };
            Some(filter_components(
                &mut seg,
                spacing,
                min_component_volume_mm3.unwrap_or(0.0),
                component_connectivity.unwrap_or(26),
            )?)
        } else {
            None
        };

    let id_to_name = if let Some(card) = &external_card {
        card.id_to_name()
    } else {
        registry::model_label_maps(&models_dir, &model)?.1
    };
    let is_nii = is_nifti_output(output);
    let t_write = Instant::now();
    let out_loc = if args.ml || is_nii {
        let path = if is_nii {
            output.clone()
        } else {
            output.join(format!("{}.nii.gz", args.task))
        };
        io::write_seg(&seg, &path, &props, canonical_affine)?;
        path
    } else {
        io::write_label_masks(
            &seg,
            output,
            &props,
            &id_to_name,
            output_roi_ids,
            canonical_affine,
            save_threads,
        )?;
        output.clone()
    };
    if args.profile {
        eprintln!(
            "[profile] cli write_output: {:.3}s",
            t_write.elapsed().as_secs_f64()
        );
    }
    say(format!(
        "wrote {}  (cold_start={:.2}s, predict={:.2}s)",
        out_loc.display(),
        cold.as_secs_f32(),
        t1.elapsed().as_secs_f32()
    ));

    if let (Some(arg), Some(summary)) = (&args.detections_json, &detection_summary) {
        let text = serde_json::to_string_pretty(summary)?;
        if arg == "-" {
            println!("{text}");
        } else {
            let path = if arg == "<auto>" {
                detections_output_path(&out_loc)
            } else {
                PathBuf::from(arg)
            };
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            fs::write(&path, text)
                .with_context(|| format!("failed to write {}", path.display()))?;
            say(format!("wrote {}", path.display()));
        }
    }

    if let Some(preview_arg) = &args.preview {
        let data = retained_data
            .as_ref()
            .expect("preview path retains input volume");
        let path = preview_output_path(preview_arg, &out_loc, &args.task);
        let t_preview = Instant::now();
        preview::write_contact_sheet(data, &seg, &path, args.preview_tiles)?;
        if args.profile {
            eprintln!(
                "[profile] cli write_preview: {:.3}s",
                t_preview.elapsed().as_secs_f64()
            );
        }
        say(format!("wrote {}", path.display()));
    }

    if let Some(stat_arg) = &args.statistics {
        let voxel_spacing = if lowres {
            predictor.target_spacing()
        } else {
            props.spacing_zyx
        };
        let voxel_ml = voxel_spacing.iter().map(|v| *v as f64).product::<f64>() / 1000.0;
        let data = retained_data
            .as_ref()
            .expect("statistics path retains input volume");
        let mut stats = compute_statistics(data, &seg, voxel_ml, &id_to_name, true);
        // Run context for machine readers: what was segmented, how, and where
        // each structure's mask file is (null when no file was written for it).
        if let Some(obj) = stats.as_object_mut() {
            obj.insert("input".to_string(), serde_json::json!(input.display().to_string()));
            obj.insert("output".to_string(), serde_json::json!(out_loc.display().to_string()));
            obj.insert("tier".to_string(), serde_json::json!(tier));
            obj.insert("model".to_string(), serde_json::json!(model));
            obj.insert("int8".to_string(), serde_json::json!(args.int8));
            obj.insert("structures_requested".to_string(), serde_json::json!(args.roi));
            let per_file = out_loc.is_dir();
            if let Some(structures) = obj.get_mut("structures").and_then(|v| v.as_object_mut()) {
                for (name, entry) in structures.iter_mut() {
                    let mask = if per_file {
                        out_loc.join(format!("{name}.nii.gz"))
                    } else {
                        out_loc.clone()
                    };
                    let mask = mask.is_file().then(|| mask.display().to_string());
                    if let Some(entry) = entry.as_object_mut() {
                        entry.insert("mask".to_string(), serde_json::json!(mask));
                    }
                }
            }
        }
        let text = serde_json::to_string_pretty(&stats)?;
        if stat_arg == "-" {
            println!("{text}");
        } else {
            let path = if stat_arg == "<auto>" {
                let base = if out_loc.is_dir() {
                    out_loc.clone()
                } else {
                    out_loc
                        .parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(Path::new("."))
                        .to_path_buf()
                };
                base.join("statistics.json")
            } else {
                PathBuf::from(stat_arg)
            };
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            fs::write(&path, text)
                .with_context(|| format!("failed to write {}", path.display()))?;
            say(format!("wrote {}", path.display()));
        }
    }
    // CUDA ONNX Runtime builds can crash during provider/session teardown. This
    // is a short-lived CLI, so keep sessions alive until process exit.
    std::mem::forget(predictor);
    if let Some(localizer) = localizer {
        std::mem::forget(localizer);
    }
    Ok(())
}

fn tier(args: &SegmentArgs) -> &'static str {
    if args.ultrafast {
        "ultrafast"
    } else if args.fastest {
        "fastest"
    } else if args.fast {
        "fast"
    } else {
        "full"
    }
}

fn is_nifti_output(path: &Path) -> bool {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    lower.ends_with(".nii") || lower.ends_with(".nii.gz")
}

fn default_save_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(8)
}

fn preview_output_path(arg: &str, out_loc: &Path, task: &str) -> PathBuf {
    let default_name = format!("preview_{task}.png");
    if arg == "<auto>" {
        let base = if out_loc.is_dir() {
            out_loc.to_path_buf()
        } else {
            out_loc
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."))
                .to_path_buf()
        };
        return base.join(default_name);
    }
    let path = PathBuf::from(arg);
    if path.is_dir() {
        path.join(default_name)
    } else {
        path
    }
}

fn detections_output_path(out_loc: &Path) -> PathBuf {
    if out_loc.is_dir() {
        return out_loc.join("detections.json");
    }
    let name = out_loc
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("segmentation");
    let stem = name
        .strip_suffix(".nii.gz")
        .or_else(|| name.strip_suffix(".nii"))
        .unwrap_or(name);
    out_loc.with_file_name(format!("{stem}.detections.json"))
}

fn user_error(message: String) -> anyhow::Error {
    UserError(message).into()
}

fn parse_shape_arg(value: &str) -> std::result::Result<[usize; 3], String> {
    let parts = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| format!("invalid shape element '{part}'"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        return Err("shape must be z,y,x".to_string());
    }
    Ok([parts[0], parts[1], parts[2]])
}

fn parse_label_ids(value: &str) -> std::result::Result<Vec<u8>, String> {
    let ids = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u8>()
                .map_err(|_| format!("invalid label id '{part}'"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("at least one label id is required".to_string());
    }
    Ok(ids)
}

#[cfg(test)]
mod cli_tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn roi_subset_accepts_commas_and_repeated_options() {
        let cli = Cli::try_parse_from([
            "onnx-totalseg",
            "-i",
            "scan.nii.gz",
            "-o",
            "output",
            "--roi_subset",
            "spleen,liver",
            "--roi_subset",
            "pancreas",
        ])
        .unwrap();

        assert_eq!(cli.segment.roi, ["spleen", "liver", "pancreas"]);
    }
}
