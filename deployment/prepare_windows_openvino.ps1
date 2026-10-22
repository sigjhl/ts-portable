param(
  [string]$StagingDir = ".\dist\onnx-totalseg-openvino",
  [string]$Binary = ".\target\release\onnx-totalseg.exe",
  [string]$ModelsDir = ".\models",
  # onnxruntime-openvino wheel: provides onnxruntime.dll + the OpenVINO EP glue.
  [string]$OrtOpenvinoVersion = "1.24.1",
  # openvino wheel: provides the OpenVINO runtime + CPU plugin + TBB + plugins.xml.
  # Keep aligned with the OpenVINO build bundled by the onnxruntime-openvino wheel
  # above (1.24.1 bundles OpenVINO 2025.4.1).
  [string]$OpenvinoVersion = "2025.4.1",
  [string]$PyTag = "cp312-cp312-win_amd64",
  [string]$CacheDir = ".\.pkg-cache",
  [switch]$Force
)

# Stages a self-contained Windows CPU package that runs inference through the
# OpenVINO execution provider (`--device openvino`). OpenVINO is oneDNN-backed and
# is ~1.8-2.7x faster than the legacy MLAS CPU path on the segmentation tiers,
# including on pre-AVX-512 CPUs (the win is fp32 conv kernels + graph fusion, which
# run on AVX2). The OpenVINO runtime ships as redistributable DLLs bundled here
# alongside the binary -- there is NO system install / SDK required on the target.

$ErrorActionPreference = "Stop"
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }

function Resolve-RequiredPath([string]$Path, [string]$Description) {
  if (!(Test-Path $Path)) { throw "$Description not found: $Path" }
  return (Resolve-Path $Path).Path
}
function Copy-RequiredFile([string]$Source, [string]$Destination) {
  $parent = Split-Path -Parent $Destination
  if ($parent -and !(Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
  Copy-Item -Force $Source $Destination
}
# Download a PyPI wheel (a zip) and extract it into $CacheDir, returning the extract dir.
function Get-Wheel([string]$Package, [string]$Version, [string]$Tag, [string]$Cache) {
  $wheel = "${Package}-${Version}-${Tag}.whl"
  $letter = $Package.Substring(0,1)
  $url = "https://files.pythonhosted.org/packages/py3/$letter/$Package/$wheel"
  $whlPath = Join-Path $Cache $wheel
  if (!(Test-Path $whlPath)) {
    Write-Host "downloading $Package $Version ..."
    # PyPI hashed-path fallback: resolve the real URL from the JSON API if the
    # simple path 404s (wheel URLs are content-hashed).
    try { Invoke-WebRequest -Uri $url -OutFile $whlPath }
    catch {
      $meta = Invoke-RestMethod -Uri "https://pypi.org/pypi/$Package/$Version/json"
      # Some PyPI projects insert a build tag between the version and Python
      # tag (for example, openvino-2025.4.1-20426-cp312-...). Accept either
      # form while still requiring the requested package, version, and tag.
      $file = $meta.urls | Where-Object {
        $_.filename -eq $wheel -or $_.filename -like "$Package-$Version-*-$Tag.whl"
      } | Select-Object -First 1
      if (!$file) { throw "no wheel '$wheel' on PyPI for $Package $Version (check -PyTag / versions)" }
      Invoke-WebRequest -Uri $file.url -OutFile $whlPath
    }
  }
  $extract = Join-Path $Cache "$Package-$Version"
  if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
  $zip = "$whlPath.zip"; Copy-Item -Force $whlPath $zip
  Expand-Archive -Force $zip $extract
  return $extract
}

$binaryPath = Resolve-RequiredPath $Binary "Rust CLI binary"
$modelsPath = Resolve-RequiredPath $ModelsDir "models directory"

if ((Test-Path $StagingDir) -and !$Force) { throw "staging directory already exists: $StagingDir (pass -Force to overwrite)" }
if (Test-Path $StagingDir) { Remove-Item -Recurse -Force $StagingDir }
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null
$cache = (New-Item -ItemType Directory -Force -Path $CacheDir).FullName

$stagedOrt = Join-Path $StagingDir "onnxruntime"
New-Item -ItemType Directory -Force -Path $stagedOrt | Out-Null

# 1) onnxruntime-openvino wheel -> onnxruntime.dll + provider DLLs (in onnxruntime/capi/).
$ortExtract = Get-Wheel "onnxruntime_openvino" $OrtOpenvinoVersion $PyTag $cache
$ortCapi = Resolve-RequiredPath (Join-Path $ortExtract "onnxruntime\capi") "onnxruntime capi dir"
Copy-Item -Force (Join-Path $ortCapi "*.dll") $stagedOrt
Resolve-RequiredPath (Join-Path $stagedOrt "onnxruntime.dll") "onnxruntime.dll" | Out-Null

# 2) openvino wheel -> runtime + CPU plugin + TBB (in openvino/libs/).
$ovExtract = Get-Wheel "openvino" $OpenvinoVersion $PyTag $cache
$ovLibs = Resolve-RequiredPath (Join-Path $ovExtract "openvino\libs") "openvino libs dir"
# Copy every runtime DLL and the plugins.xml (maps device -> plugin DLL). The CPU
# path needs openvino.dll, openvino_intel_cpu_plugin.dll, openvino_onnx_frontend.dll,
# tbb*.dll; copying all keeps AUTO/HETERO working. Some wheel releases include
# plugins.xml and some (including 2025.4.1) register their plugins without it.
Copy-Item -Force (Join-Path $ovLibs "*.dll") $stagedOrt
if (Test-Path (Join-Path $ovLibs "plugins.xml")) { Copy-Item -Force (Join-Path $ovLibs "plugins.xml") $stagedOrt }
foreach ($dll in @("openvino.dll", "openvino_intel_cpu_plugin.dll", "openvino_onnx_frontend.dll")) {
  if (!(Test-Path (Join-Path $stagedOrt $dll))) { Write-Warning "expected OpenVINO DLL not staged: $dll (verify -PyTag / -OpenvinoVersion)" }
}

Copy-RequiredFile $binaryPath (Join-Path $StagingDir "onnx-totalseg.exe")
Copy-Item -Recurse -Force $modelsPath (Join-Path $StagingDir "models")

$readme = @"
onnx-totalseg OpenVINO CPU package
==================================

Self-contained Windows CPU package. Inference runs through the OpenVINO execution
provider (oneDNN-backed): ~1.8-2.7x faster than the legacy MLAS CPU path on the
segmentation tiers, including on pre-AVX-512 CPUs. All OpenVINO DLLs are bundled
in .\onnxruntime\ -- no system install or OpenVINO SDK is required on this machine.

Run INT8:
  .\onnx-totalseg.exe segment -i C:\path\ct.nii.gz -o C:\path\seg.nii.gz --fast ^
    --device openvino --int8 --ort-lib .\onnxruntime\onnxruntime.dll --models-dir .\models

Run FP32:
  .\onnx-totalseg.exe segment -i C:\path\ct.nii.gz -o C:\path\seg.nii.gz --fast ^
    --device openvino --ort-lib .\onnxruntime\onnxruntime.dll --models-dir .\models

Warm server:
  .\onnx-totalseg.exe serve --device openvino --ort-lib .\onnxruntime\onnxruntime.dll ^
    --models-dir .\models --warm ultrafast

Notes:
  - --device openvino targets the CPU device. Pass --int8 for quantized model
    variants; without it the runtime uses fp32. Do NOT pass --fp16 for CPU.
  - The OpenVINO CPU plugin auto-detects the ISA (AVX2 / AVX-512 / VNNI); no
    per-CPU configuration is needed.
  - Keep openvino_intel_cpu_plugin.dll and every other staged DLL next to
    onnxruntime.dll in .\onnxruntime\. If a wheel supplies plugins.xml, keep
    that file there too; OpenVINO 2025.4.1 does not require or ship it.

Included runtime:
  onnxruntime-openvino $OrtOpenvinoVersion, OpenVINO $OpenvinoVersion ($PyTag).
  DLLs copied from the onnxruntime-openvino and openvino wheels into .\onnxruntime\.
"@
$readme | Set-Content -Encoding UTF8 (Join-Path $StagingDir "README_OPENVINO.txt")

Write-Host "staged OpenVINO CPU package: $StagingDir"
