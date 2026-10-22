use std::{
    collections::HashMap,
    error::Error,
    fmt, fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use crate::{
    io,
    report::compute_statistics,
    runtime::{
        CropEmptyBehavior, LocalizerCropOptions, Predictor, RuntimeOptions, crop_with_localizer,
        filter_roi, provider_plan_uses_gpu, resolve_provider_names,
    },
    tasks::totalseg::{build_predictor, crop_cascade, registry},
};
use anyhow::{Context, Result, bail};
use clap::Args as ClapArgs;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, ClapArgs, Debug)]
pub struct ServerArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8765)]
    port: u16,

    /// Execution provider: auto, cpu, openvino, cuda, directml
    #[arg(long, default_value = "auto")]
    device: String,

    /// Number of inference threads [default: all cores]
    #[arg(long)]
    threads: Option<usize>,

    #[arg(long)]
    warm: Option<String>,

    /// Folder holding the model cards, label map and ONNX files
    /// [default: $ONNX_TOTALSEG_MODELS, else ./models]
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// Path to the ONNX Runtime shared library [default: $ORT_DYLIB_PATH, else the system library]
    #[arg(long)]
    ort_lib: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    model: String,
    tier: String,
    device: String,
    threads: Option<usize>,
    fp16: bool,
    int8: bool,
    low_memory_cpu: bool,
    order_override: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct PredictRequest {
    input: String,
    output: String,
    task: Option<String>,
    tier: Option<String>,
    roi: Option<String>,
    statistics: Option<bool>,
    fp16: Option<bool>,
    int8: Option<bool>,
}

#[derive(Debug)]
struct UserError(String);

struct ServerState {
    cache: Mutex<HashMap<CacheKey, Predictor>>,
    models_dir: PathBuf,
    args: ServerArgs,
}

impl fmt::Display for UserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for UserError {}

pub fn run(args: ServerArgs) -> Result<()> {
    let models_dir = registry::models_dir(args.models_dir.as_deref());
    let mut cache: HashMap<CacheKey, Predictor> = HashMap::new();
    if let Some(warm) = &args.warm {
        for tier in warm.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let Ok(task_model) = registry::resolve_task_model("total", tier) else {
                continue;
            };
            let providers = resolve_provider_names(None, &args.device, args.ort_lib.clone())?;
            let is_gpu = provider_plan_uses_gpu(&providers);
            let fp16 = is_gpu;
            let key = CacheKey {
                model: task_model.model.to_string(),
                tier: tier.to_string(),
                device: args.device.clone(),
                threads: args.threads,
                fp16,
                int8: false,
                low_memory_cpu: registry::low_memory_cpu_model(task_model.model) && !is_gpu,
                order_override: task_model.resample_order,
            };
            get_predictor(&mut cache, &key, &models_dir, &args)?;
        }
    }
    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).with_context(|| format!("failed to bind {addr}"))?;
    let state = Arc::new(ServerState {
        cache: Mutex::new(cache),
        models_dir,
        args: args.clone(),
    });
    println!(
        "onnx-totalseg serving on http://{} (device={}, warm={})",
        addr,
        args.device,
        if state.cache.lock().is_ok_and(|cache| cache.is_empty()) {
            "lazy"
        } else {
            args.warm.as_deref().unwrap_or("lazy")
        }
    );
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    let response = handle_connection(&mut stream, &state).unwrap_or_else(error_500);
                    let _ = write_response(&mut stream, response);
                });
            }
            Err(e) => eprintln!("connection error: {e}"),
        }
    }
    Ok(())
}

fn handle_connection(stream: &mut TcpStream, state: &ServerState) -> Result<HttpResponse> {
    let request = read_http_request(stream)?;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") | ("GET", "/health/") => {
            let cache = state
                .cache
                .lock()
                .map_err(|_| anyhow::anyhow!("predictor cache lock poisoned"))?;
            Ok(json_response(
                200,
                json!({
                    "status": "ok",
                    "loaded": loaded_cache_keys(&cache),
                }),
            ))
        }
        ("POST", "/predict") | ("POST", "/predict/") => {
            let result = state
                .cache
                .lock()
                .map_err(|_| anyhow::anyhow!("predictor cache lock poisoned"))
                .and_then(|mut cache| {
                    predict(&request.body, &mut cache, &state.models_dir, &state.args)
                });
            match result {
                Ok(body) => Ok(json_response(200, body)),
                Err(e) if e.downcast_ref::<UserError>().is_some() => Ok(json_response(
                    400,
                    json!({ "status": "error", "error": e.to_string() }),
                )),
                Err(e) => Ok(json_response(
                    500,
                    json!({ "status": "error", "error": format!("{:#}", e) }),
                )),
            }
        }
        _ => Ok(json_response(
            404,
            json!({ "status": "error", "error": "not found" }),
        )),
    }
}

fn predict(
    body: &[u8],
    cache: &mut HashMap<CacheKey, Predictor>,
    models_dir: &Path,
    args: &ServerArgs,
) -> Result<Value> {
    let req: PredictRequest =
        serde_json::from_slice(body).map_err(|e| user_error(format!("invalid JSON body: {e}")))?;
    if req.input.is_empty() || req.output.is_empty() {
        return Err(user_error("'input' and 'output' are required".to_string()));
    }
    let task = req.task.as_deref().unwrap_or("total");
    let tier = req.tier.as_deref().unwrap_or("full");
    let task_model =
        registry::resolve_task_model(task, tier).map_err(|e| UserError(e.to_string()))?;
    let providers = resolve_provider_names(None, &args.device, args.ort_lib.clone())?;
    let is_gpu = provider_plan_uses_gpu(&providers);
    let fp16 = req.fp16.unwrap_or(is_gpu) && is_gpu;
    // INT8 is CPU-only (OpenVINO); opt-in per request, never auto-enabled by GPU.
    let int8 = req.int8.unwrap_or(false) && !is_gpu;
    let key = CacheKey {
        model: task_model.model.to_string(),
        tier: tier.to_string(),
        device: args.device.clone(),
        threads: args.threads,
        fp16,
        int8,
        low_memory_cpu: registry::low_memory_cpu_model(task_model.model) && !is_gpu,
        order_override: task_model.resample_order,
    };
    let roi_ids = match req.roi.as_deref() {
        Some(roi) if !roi.trim().is_empty() => Some(
            registry::resolve_roi_for_model(models_dir, task_model.model, roi)
                .map_err(|e| UserError(e.to_string()))?,
        ),
        _ => None,
    };
    let task_output_roi_ids = match (roi_ids.is_none(), task_model.output_roi_names) {
        (true, Some(names)) => Some(
            registry::resolve_labels_for_model(models_dir, task_model.model, names)
                .map_err(|e| UserError(e.to_string()))?,
        ),
        _ => None,
    };
    let output_roi_ids = roi_ids.as_deref().or(task_output_roi_ids.as_deref());
    let task_crop_ids = match task_model.crop {
        Some(crop) => Some(
            registry::resolve_labels_for_model(models_dir, crop.localizer_model, crop.roi_names)
                .map_err(|e| UserError(e.to_string()))?,
        ),
        None => None,
    };
    let t = Instant::now();
    if !Path::new(&req.input).exists() {
        return Err(user_error(format!("input not found: {}", req.input)));
    }
    let (input_path, _tmp) = io::collect_nifti_or_dicom_input(Path::new(&req.input))?;
    let preserve_input_orientation =
        get_predictor(cache, &key, models_dir, args)?.preserves_input_orientation();
    let (data, props) = io::read_image_with_orientation(&input_path, preserve_input_orientation)?;
    let lowres = task_model.lowres;
    let use_roi_cascade = task == "total" && roi_ids.is_some() && matches!(tier, "full" | "fast");
    let (seg, canonical_affine, provider, stats_spacing) = if let Some(crop) = task_model.crop {
        let ids = task_crop_ids
            .as_deref()
            .expect("task crop requires crop ids");
        let localizer_key = CacheKey {
            model: crop.localizer_model.to_string(),
            tier: "localizer".to_string(),
            device: args.device.clone(),
            threads: args.threads,
            fp16,
            int8,
            low_memory_cpu: (registry::low_memory_cpu_model(crop.localizer_model)
                || crop.localizer_low_memory_cpu)
                && !is_gpu,
            order_override: None,
        };
        get_predictor(cache, &key, models_dir, args)?;
        get_predictor(cache, &localizer_key, models_dir, args)?;

        let mut predictor = cache
            .remove(&key)
            .context("target predictor disappeared from cache")?;
        let mut localizer = cache
            .remove(&localizer_key)
            .context("localizer predictor disappeared from cache")?;
        let provider = predictor.active_provider().to_string();
        let result = crop_with_localizer(
            &data,
            &props,
            &mut predictor,
            &mut localizer,
            LocalizerCropOptions {
                crop_ids: ids,
                margin_mm: crop.margin_mm,
                output_roi_ids,
                empty_behavior: if crop.empty_is_empty {
                    CropEmptyBehavior::ReturnEmpty
                } else {
                    CropEmptyBehavior::RunTargetFull
                },
            },
        );
        cache.insert(key.clone(), predictor);
        cache.insert(localizer_key, localizer);
        (result?, None, provider, props.spacing_zyx)
    } else if use_roi_cascade {
        let ids = roi_ids.as_deref().expect("cascade requires ROI ids");
        let localizer_key = CacheKey {
            model: "fastest".to_string(),
            tier: "fastest".to_string(),
            device: args.device.clone(),
            threads: args.threads,
            fp16,
            int8,
            low_memory_cpu: registry::low_memory_cpu_model("fastest") && !is_gpu,
            order_override: Some(1),
        };
        get_predictor(cache, &key, models_dir, args)?;
        get_predictor(cache, &localizer_key, models_dir, args)?;

        let mut predictor = cache
            .remove(&key)
            .context("target predictor disappeared from cache")?;
        let mut localizer = cache
            .remove(&localizer_key)
            .context("localizer predictor disappeared from cache")?;
        let provider = predictor.active_provider().to_string();
        let result = crop_cascade(&data, &props, &mut predictor, &mut localizer, ids);
        cache.insert(key.clone(), predictor);
        cache.insert(localizer_key, localizer);
        (result?, None, provider, props.spacing_zyx)
    } else {
        let predictor = get_predictor(cache, &key, models_dir, args)?;
        let provider = predictor.active_provider().to_string();
        let stats_spacing = if lowres {
            predictor.target_spacing()
        } else {
            props.spacing_zyx
        };
        let (mut seg, canonical_affine) = predictor.predict_loaded(&data, &props, lowres)?;
        if let Some(ids) = output_roi_ids {
            filter_roi(&mut seg, ids);
        }
        (seg, canonical_affine, provider, stats_spacing)
    };
    let output = PathBuf::from(&req.output);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    io::write_seg(&seg, &output, &props, canonical_affine)?;
    let mut response = json!({
        "status": "ok",
        "output": req.output,
        "task": task,
        "tier": tier,
        "provider": provider,
        "predict_s": round3(t.elapsed().as_secs_f64()),
    });
    if req.statistics.unwrap_or(false) {
        let (_, id_to_name) = registry::model_label_maps(models_dir, task_model.model)?;
        let voxel_ml = stats_spacing.iter().map(|v| *v as f64).product::<f64>() / 1000.0;
        response["statistics"] = compute_statistics(&data, &seg, voxel_ml, &id_to_name, true);
    }
    Ok(response)
}

fn get_predictor<'a>(
    cache: &'a mut HashMap<CacheKey, Predictor>,
    key: &CacheKey,
    models_dir: &Path,
    args: &ServerArgs,
) -> Result<&'a mut Predictor> {
    if !cache.contains_key(key) {
        let providers = resolve_provider_names(None, &key.device, args.ort_lib.clone())?;
        let is_gpu = provider_plan_uses_gpu(&providers);
        let opts = RuntimeOptions {
            providers: Some(providers),
            device: key.device.clone(),
            threads: key.threads,
            fp16: key.fp16,
            int8: key.int8,
            ort_lib: args.ort_lib.clone(),
            profile: false,
            low_memory_cpu: key.low_memory_cpu && !is_gpu,
            cuda_low_memory: false,
            lazy_session: false,
        };
        let predictor = build_predictor(&key.model, models_dir, opts, key.order_override)?;
        cache.insert(key.clone(), predictor);
    }
    Ok(cache.get_mut(key).expect("cache entry exists"))
}

fn loaded_cache_keys(cache: &HashMap<CacheKey, Predictor>) -> Vec<Value> {
    cache
        .keys()
        .map(|key| {
            json!([
                key.model,
                key.tier,
                key.device,
                key.threads,
                key.fp16,
                key.int8,
                key.low_memory_cpu,
                key.order_override
            ])
        })
        .collect()
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

struct HttpResponse {
    status: u16,
    reason: &'static str,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            bail!("empty request");
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_header_end(&buf) {
            header_end = pos;
            break;
        }
        if buf.len() > 1024 * 1024 {
            bail!("HTTP header too large");
        }
    }
    let headers = std::str::from_utf8(&buf[..header_end]).context("request header is not UTF-8")?;
    let mut lines = headers.lines();
    let request_line = lines.next().context("missing request line")?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().context("missing method")?.to_string();
    let path = parts.next().context("missing path")?.to_string();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find_map(|(k, v)| k.eq_ignore_ascii_case("content-length").then(|| v.trim()))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf
        [body_start..body_start + content_length.min(buf.len().saturating_sub(body_start))]
        .to_vec();
    Ok(HttpRequest { method, path, body })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> Result<()> {
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        response.reason,
        response.body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(&response.body)?;
    Ok(())
}

fn json_response(status: u16, body: Value) -> HttpResponse {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    HttpResponse {
        status,
        reason,
        body: serde_json::to_vec(&body).unwrap_or_else(|_| b"{\"status\":\"error\"}".to_vec()),
    }
}

fn error_500(err: anyhow::Error) -> HttpResponse {
    json_response(
        500,
        json!({ "status": "error", "error": format!("{:#}", err) }),
    )
}

fn user_error(message: String) -> anyhow::Error {
    UserError(message).into()
}
