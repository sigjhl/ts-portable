param(
  [Parameter(Mandatory=$true)]
  [string]$InputPath,

  [Parameter(Mandatory=$true)]
  [string]$OutputPath,

  [string]$Binary = ".\onnx-totalseg.exe",
  [string]$OrtLib = ".\onnxruntime\onnxruntime.dll",
  [string]$ModelsDir = ".\models",
  [ValidateSet("full", "fast", "fastest", "ultrafast")]
  [string]$Tier = "full",
  [string]$Roi = "spleen",
  # Default is INT8 (the OpenVINO-quantized path). Pass -Fp32 for the single-precision
  # OpenVINO baseline (voxelwise-identical to the MLAS CPU path, ~2.0x slower than INT8 at 1.5 mm, ~1.6x at 3 mm).
  [switch]$Fp32,
  [switch]$Statistics
)

# Runs the crop-cascade spleen biomarker (default) through the OpenVINO CPU execution
# provider. INT8 by default; the OpenVINO DLLs staged by prepare_windows_openvino.ps1
# sit in .\onnxruntime\ next to onnxruntime.dll (no system OpenVINO install required).

$ErrorActionPreference = "Stop"

function Resolve-PackagePath([string]$Path) {
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $PSScriptRoot $Path
}

$BinaryPath = Resolve-PackagePath $Binary
$OrtLibPath = Resolve-PackagePath $OrtLib
$ModelsDirPath = Resolve-PackagePath $ModelsDir

if (!(Test-Path $BinaryPath))    { throw "Rust binary not found: $BinaryPath" }
if (!(Test-Path $OrtLibPath))    { throw "ONNX Runtime DLL not found: $OrtLibPath" }
if (!(Test-Path $ModelsDirPath)) { throw "models directory not found: $ModelsDirPath" }

# The OpenVINO CPU plugin + TBB DLLs must be discoverable next to onnxruntime.dll.
$ortDir = Split-Path -Parent (Resolve-Path $OrtLibPath)
$env:PATH = "$ortDir;$env:PATH"

$args = @(
  "-i", $InputPath,
  "-o", $OutputPath,
  "--device", "openvino",
  "--ort-lib", $OrtLibPath,
  "--models-dir", $ModelsDirPath
)

if (-not $Fp32) { $args += "--int8" }   # INT8 is the default; -Fp32 selects single precision

switch ($Tier) {
  "fast"     { $args += "--fast" }
  "fastest"  { $args += "--fastest" }
  "ultrafast"{ $args += "--ultrafast" }
  # "full" => no tier flag (whole 1.5-mm path / full-resolution crop-cascade)
}

if ($Roi)        { $args += @("--roi_subset", $Roi) }
if ($Statistics) { $args += "--statistics" }

& $BinaryPath @args
exit $LASTEXITCODE
