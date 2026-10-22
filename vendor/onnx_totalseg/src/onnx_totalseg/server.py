"""Warm server: load once, serve many — amortizes import + ORT-session init across calls.
On a slow machine the cold start (especially the full tier's 5 sessions) dominates per-call
cost; the server pays it once and keeps models resident.

Dependency-light: stdlib http.server only. File-handoff — the client sends paths; volumes stay
on local disk (nothing is uploaded). General/standalone (not tied to any host app).

  onnx-totalseg-serve [--host 127.0.0.1] [--port 8765] [--device auto] [--threads N]

  POST /predict  {"input","output","tier","roi","statistics","fp16"}
                 -> {"status":"ok","output","tier","provider","predict_s",["statistics"]}
  GET  /health   -> {"status":"ok","loaded":[...]}
"""
import argparse
import cgi
import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

import numpy as np

from . import io_nibabel, registry, roi as roi_mod
from .cli import CASCADE_TIERS, TIERS, _build
from .ep import resolve_providers
from .report import compute_statistics

_CACHE = {}
_LOCK = threading.Lock()


GUI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ONNX TotalSegmentator</title>
  <style>
    :root { color-scheme: light dark; font-family: Segoe UI, system-ui, sans-serif; }
    body { margin: 0; background: Canvas; color: CanvasText; }
    main { max-width: 760px; margin: 0 auto; padding: 28px; }
    h1 { font-size: 24px; margin: 0 0 18px; font-weight: 650; }
    .drop { border: 2px dashed #777; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; }
    .drop.drag { border-color: #2b7cff; background: color-mix(in srgb, #2b7cff 12%, Canvas); }
    .file { margin-top: 10px; font-size: 14px; color: #666; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 18px; }
    label { display: grid; gap: 6px; font-size: 13px; }
    select, button { font: inherit; padding: 10px 12px; border-radius: 6px; border: 1px solid #888; }
    button { margin-top: 18px; background: #1d64d8; color: white; border-color: #1d64d8; cursor: pointer; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .check { display: flex; align-items: center; gap: 8px; margin-top: 18px; }
    pre { white-space: pre-wrap; background: color-mix(in srgb, CanvasText 8%, Canvas); padding: 12px; border-radius: 6px; }
    a { color: #1d64d8; font-weight: 600; }
    @media (max-width: 640px) { main { padding: 18px; } .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>ONNX TotalSegmentator</h1>
  <div id="drop" class="drop">
    <strong>Drop a NIfTI file here</strong><br>
    <span>or click to choose .nii / .nii.gz</span>
    <input id="file" type="file" accept=".nii,.gz" hidden>
    <input id="dicomFolder" type="file" webkitdirectory directory multiple hidden>
    <div id="fileName" class="file">No file selected</div>
  </div>
  <button id="chooseDicom" type="button">Choose DICOM folder</button>

  <div class="grid">
    <label>Segmentation tier
      <select id="tier">
        <option value="full">Full quality</option>
        <option value="fast">Fast</option>
        <option value="fastest">Fastest</option>
      </select>
    </label>
    <label>ROI
      <select id="roi">
        <option value="spleen">spleen</option>
      </select>
    </label>
  </div>

  <label class="check"><input id="statistics" type="checkbox" checked> Write statistics</label>
  <button id="run" disabled>Run segmentation</button>
  <p id="status">Server is starting...</p>
  <p id="download"></p>
  <pre id="details" hidden></pre>
</main>
<script>
const drop = document.getElementById('drop');
const input = document.getElementById('file');
const dicomInput = document.getElementById('dicomFolder');
const fileName = document.getElementById('fileName');
const run = document.getElementById('run');
const chooseDicom = document.getElementById('chooseDicom');
const statusBox = document.getElementById('status');
const details = document.getElementById('details');
const download = document.getElementById('download');
let selectedFile = null;
let selectedDicomFiles = [];

function setFile(file) {
  selectedFile = file;
  selectedDicomFiles = [];
  fileName.textContent = file ? `${file.name} (${Math.round(file.size / 1048576)} MB)` : 'No file selected';
  run.disabled = !file;
  download.innerHTML = '';
  details.hidden = true;
}

drop.addEventListener('click', event => {
  event.preventDefault();
  input.click();
});
input.addEventListener('click', event => event.stopPropagation());
dicomInput.addEventListener('click', event => event.stopPropagation());
input.addEventListener('change', () => setFile(input.files[0]));
chooseDicom.addEventListener('click', event => {
  event.preventDefault();
  event.stopPropagation();
  dicomInput.click();
});
dicomInput.addEventListener('change', () => {
  selectedFile = null;
  selectedDicomFiles = Array.from(dicomInput.files || []);
  fileName.textContent = selectedDicomFiles.length
    ? `DICOM folder (${selectedDicomFiles.length} files)`
    : 'No file selected';
  run.disabled = selectedDicomFiles.length === 0;
  download.innerHTML = '';
  details.hidden = true;
});
['dragenter', 'dragover'].forEach(name => drop.addEventListener(name, event => {
  event.preventDefault(); drop.classList.add('drag');
}));
['dragleave', 'drop'].forEach(name => drop.addEventListener(name, event => {
  event.preventDefault(); drop.classList.remove('drag');
}));
drop.addEventListener('drop', event => setFile(event.dataTransfer.files[0]));

async function loadOptions() {
  try {
    const res = await fetch('/options');
    const opts = await res.json();
    const roi = document.getElementById('roi');
    roi.innerHTML = '<option value="">All structures</option>' +
      opts.rois.map(name => `<option value="${name}">${name}</option>`).join('');
    roi.value = opts.rois.includes('spleen') ? 'spleen' : '';
    statusBox.textContent = 'Ready.';
  } catch (error) {
    statusBox.textContent = 'Ready. ROI list could not be loaded.';
  }
}

run.addEventListener('click', async () => {
  if (!selectedFile && selectedDicomFiles.length === 0) return;
  run.disabled = true;
  download.innerHTML = '';
  details.hidden = true;
  statusBox.textContent = 'Running... Segmentation can take a while.';
  const form = new FormData();
  if (selectedFile) {
    form.append('file', selectedFile);
  } else {
    selectedDicomFiles.forEach(file => form.append('dicom_files', file, file.webkitRelativePath || file.name));
  }
  form.append('tier', document.getElementById('tier').value);
  form.append('roi', document.getElementById('roi').value);
  form.append('statistics', document.getElementById('statistics').checked ? '1' : '0');
  try {
    const res = await fetch('/upload_predict', { method: 'POST', body: form });
    const obj = await res.json();
    if (!res.ok || obj.status !== 'ok') throw new Error(obj.error || 'Segmentation failed');
    statusBox.textContent = `Done in ${obj.predict_s}s with ${obj.provider}.`;
    download.innerHTML = `<a href="${obj.download_url}">Download segmentation</a>` +
      (obj.statistics_url ? ` &nbsp; <a href="${obj.statistics_url}">Download statistics</a>` : '');
    if (obj.statistics) {
      details.textContent = JSON.stringify(obj.statistics, null, 2);
      details.hidden = false;
    }
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    run.disabled = !selectedFile && selectedDicomFiles.length === 0;
  }
});

loadOptions();
</script>
</body>
</html>
"""


def _provider(seg):
    return getattr(seg, "active_provider", None) or seg.parts[0].active_provider


def _get(tier, device, threads, fp16, order_override=None, device_id=None):
    key = (tier, device, threads, fp16, order_override, device_id)
    if key not in _CACHE:
        model, order, _ = TIERS[tier]
        _CACHE[key] = _build(model, order_override if order_override is not None else order,
                             None, device, threads, fp16=fp16, device_id=device_id)
    return _CACHE[key]


def _default_fp16_for_device(device):
    eff = resolve_providers(device)
    is_gpu = bool(eff) and eff[0] != "CPUExecutionProvider"
    return is_gpu


def predict(req, device, threads, device_id=None):
    tier = req.get("tier", "full")
    if tier not in TIERS:
        raise ValueError(f"unknown tier '{tier}' (full|fast|fastest|ultrafast)")
    _, _, lowres = TIERS[tier]
    # Device-matched precision by default (fp16 on GPU, fp32 on CPU); req may override with "fp16".
    is_gpu = _default_fp16_for_device(device)
    req_fp16 = req.get("fp16", None)
    fp16 = (is_gpu if req_fp16 is None else bool(req_fp16)) and is_gpu
    roi_ids = roi_mod.resolve_roi(req["roi"].split(",")) if req.get("roi") else None

    seg = _get(tier, device, threads, fp16, device_id=device_id)
    t = time.time()
    data, props = io_nibabel.read_image(req["input"])
    if roi_ids is not None and tier in CASCADE_TIERS:
        loc = _get("fastest", device, threads, fp16, order_override=1, device_id=device_id)  # order-1 localizer
        segarr, caff = roi_mod.crop_cascade(data, props, seg, loc, roi_ids), None
    else:
        segarr, caff = seg.predict_loaded(data, props, lowres=lowres)
        if roi_ids is not None:
            segarr = roi_mod.filter_roi(segarr, roi_ids)
    io_nibabel.write_seg(segarr, req["output"], props, canonical_affine=caff)

    resp = {"status": "ok", "output": req["output"], "tier": tier,
            "provider": _provider(seg), "predict_s": round(time.time() - t, 3)}
    if req.get("statistics"):
        _, id2name = roi_mod.total_label_maps()
        sp = seg.target_spacing if lowres else props["spacing"]
        vox = float(np.prod([float(x) for x in sp])) / 1000.0
        resp["statistics"] = compute_statistics(data[0], segarr, vox, id2name)
    return resp


def _safe_filename(name):
    name = os.path.basename(name or "")
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return cleaned or "input.nii"


def _is_nifti_name(name):
    low = name.lower()
    return low.endswith(".nii") or low.endswith(".nii.gz")


def _download_path(work_root, rel):
    parts = [p for p in rel.split("/") if p]
    if len(parts) != 2:
        return None
    job, filename = parts
    if not job.replace("-", "").isalnum():
        return None
    if filename not in {"segmentation.nii.gz", "statistics.json"}:
        return None
    path = (work_root / job / filename).resolve()
    try:
        path.relative_to(work_root)
    except ValueError:
        return None
    return path if path.exists() else None


def _form_items(form, name):
    if name not in form:
        return []
    value = form[name]
    return value if isinstance(value, list) else [value]


def _json_bytes(obj):
    return json.dumps(obj).encode()


def make_handler(device, threads, device_id=None, work_root=None):
    work_root = Path(work_root or "gui_runs").resolve()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = _json_bytes(obj)
            self._send_bytes(code, body, "application/json")

        def _send_bytes(self, code, body, content_type, extra_headers=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/":
                self._send_bytes(200, GUI_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/health":
                self._send(200, {"status": "ok", "loaded": [list(k) for k in _CACHE]})
            elif path == "/options":
                name2id, _ = roi_mod.total_label_maps()
                rois = sorted(k for k in name2id if k != "background")
                self._send(200, {"tiers": ["full", "fast", "fastest"], "rois": rois})
            elif path.startswith("/download/"):
                dl = _download_path(work_root, unquote(path[len("/download/"):]))
                if dl is None:
                    return self._send(404, {"status": "error", "error": "not found"})
                content_type = "application/gzip" if dl.name.endswith(".gz") else "application/json"
                headers = {"Content-Disposition": f'attachment; filename="{dl.name}"'}
                self._send_bytes(200, dl.read_bytes(), content_type, headers)
            else:
                self._send(404, {"status": "error", "error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/")
            if path == "/upload_predict":
                return self._upload_predict()
            if path != "/predict":
                return self._send(404, {"status": "error", "error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                if not req.get("input") or not req.get("output"):
                    raise ValueError("'input' and 'output' are required")
                with _LOCK:  # serialize prediction (avoid GPU contention / build races)
                    resp = predict(req, device, threads, device_id=device_id)
                self._send(200, resp)
            except (ValueError, KeyError, FileNotFoundError) as e:
                self._send(400, {"status": "error", "error": str(e)})
            except Exception as e:  # noqa
                self._send(500, {"status": "error", "error": f"{type(e).__name__}: {e}"})

        def _upload_predict(self):
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    raise ValueError("expected multipart/form-data")
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                tier = (form.getfirst("tier") or "full").strip()
                if tier not in {"full", "fast", "fastest"}:
                    raise ValueError("unknown tier")
                roi = (form.getfirst("roi") or "").strip()
                statistics = (form.getfirst("statistics") or "0") == "1"

                job = uuid4().hex
                job_dir = work_root / job
                job_dir.mkdir(parents=True, exist_ok=True)
                input_path = None
                dicom_items = [i for i in _form_items(form, "dicom_files")
                               if getattr(i, "filename", None)]
                if dicom_items:
                    dicom_dir = job_dir / "dicom"
                    dicom_dir.mkdir()
                    used = set()
                    for idx, item in enumerate(dicom_items, 1):
                        name = _safe_filename(item.filename)
                        if name in used:
                            root, ext = os.path.splitext(name)
                            name = f"{root}_{idx}{ext}"
                        used.add(name)
                        with (dicom_dir / name).open("wb") as f:
                            shutil.copyfileobj(item.file, f)
                    input_path = dicom_dir
                else:
                    item = form["file"] if "file" in form else None
                    if item is None or not getattr(item, "filename", None):
                        raise ValueError("choose a NIfTI file or DICOM folder")
                    filename = _safe_filename(item.filename)
                    if not _is_nifti_name(filename):
                        raise ValueError("input must be .nii, .nii.gz, or a DICOM folder")
                    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
                    input_path = job_dir / ("input" + suffix)
                    with input_path.open("wb") as f:
                        shutil.copyfileobj(item.file, f)
                output_path = job_dir / "segmentation.nii.gz"

                req = {
                    "input": str(input_path),
                    "output": str(output_path),
                    "tier": tier,
                    "roi": roi,
                    "statistics": statistics,
                }
                with _LOCK:
                    resp = predict(req, device, threads, device_id=device_id)

                resp["job"] = job
                resp["download_url"] = f"/download/{job}/segmentation.nii.gz"
                if statistics and "statistics" in resp:
                    stats_path = job_dir / "statistics.json"
                    stats_path.write_text(json.dumps(resp["statistics"], indent=2), encoding="utf-8")
                    resp["statistics_url"] = f"/download/{job}/statistics.json"
                self._send(200, resp)
            except (ValueError, KeyError, FileNotFoundError) as e:
                self._send(400, {"status": "error", "error": str(e)})
            except Exception as e:  # noqa
                self._send(500, {"status": "error", "error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(prog="onnx-totalseg-serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--gpu-id", "--device_id", dest="gpu_id", type=int, default=None,
                    help="GPU adapter index for DirectML/CUDA (multi-GPU boxes); default adapter 0")
    ap.add_argument("--warm", default=None,
                    help="comma-separated tiers to preload at startup (e.g. full,ultrafast)")
    ap.add_argument("--work-root", default="gui_runs",
                    help="folder for browser uploads and downloadable outputs")
    args = ap.parse_args(argv)

    if args.warm:
        fp16 = _default_fp16_for_device(args.device)
        for tier in args.warm.split(","):
            if tier in TIERS:
                _get(tier, args.device, args.threads, fp16, device_id=args.gpu_id)
    srv = ThreadingHTTPServer((args.host, args.port),
                              make_handler(args.device, args.threads, args.gpu_id, args.work_root))
    print(f"onnx-totalseg serving on http://{args.host}:{args.port} "
          f"(device={args.device}, warm={list(_CACHE) and args.warm or 'lazy'})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
