<#
.SYNOPSIS
  TS-Portable release build (v0.1.0-rc4) on the PACS reading station: identity check against the
  July build, then the 29 Duke Spleen CT series from NIfTI and from DICOM.

  Arms, run one after another (arm-major), all with runtimes\int8-openvino-rc4\onnx-totalseg.exe:
    gate         July exe and rc4 exe on a few AMOS cases (NIfTI); spleen masks compared byte for byte
                 after decompression. Not a timing arm.
    amos_nifti   rc4 on the 100 AMOS cases. Runs ONLY if the gate finds a difference (or -ForceAmos):
                 when the masks are identical, the July AMOS results stand for the release build.
    duke_nifti   rc4 on the 29 Duke CT series converted to NIfTI.
    duke_dicom   rc4 on the same 29 series as distributed (DICOM folders; JPEG Lossless, JPEG 2000 and
                 uncompressed files).

  Same protocol as Run-PacsAmos100.ps1: one unmeasured warm-up per arm, sequential runs, 250 ms
  process-tree sampling, 5 s cooldown, no --force_split, INT8 remainder pinned to f32, roi spleen.
  Afterwards each Duke case's DICOM-input mask is compared with its NIfTI-input mask.

.EXAMPLE
  # Quick check (about 2 minutes): gate on 1 case, then 3 Duke cases (JPEG 2000, uncompressed, JPEG Lossless) from NIfTI and DICOM.
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-PacsDuke29.ps1 -GateCases 1 -DukeCaseList .\DUKE3_CASES.txt -CooldownSeconds 1

.EXAMPLE
  # Full run (about 15 minutes if the gate passes; about 35 if it does not):
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-PacsDuke29.ps1

.EXAMPLE
  # Resume an interrupted run:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-PacsDuke29.ps1 -Resume .\results\duke29_run_20261020_200000
#>
[CmdletBinding()]
param(
    [string[]]$Arms = @('gate', 'duke_nifti', 'duke_dicom'),
    [int]$GateCases = 3,
    [switch]$ForceAmos,
    [string]$DukeCaseList = "",
    [string]$StartAt = "",
    [string]$Resume = "",
    [int]$WarmupRuns = 1,
    [int]$SampleMilliseconds = 250,
    [int]$CooldownSeconds = 5,
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

# Some managed launchers pass both Path and PATH; PS 5 Start-Process then fails copying the env.
$inheritedPath = $env:Path
Remove-Item Env:PATH -ErrorAction SilentlyContinue
$env:Path = $inheritedPath

$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path

# powershell.exe -File passes "-Arms a,b" as one string; normalize to an array and validate.
$Arms = @($Arms | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$KnownArms = @('gate', 'amos_nifti', 'duke_nifti', 'duke_dicom')
foreach ($a in $Arms) { if ($KnownArms -notcontains $a) { throw "Unknown arm '$a'. Valid: $($KnownArms -join ', ')" } }
if ($Arms.Count -eq 0) { throw "No arms requested." }
if ($ForceAmos -and ($Arms -notcontains 'amos_nifti')) { $Arms = @($Arms) + 'amos_nifti' }

# ---------------------------------------------------------------- verify the pack by hash
$ExpectedSha = @{}
$sumsPath = Join-Path $Root 'PACK_SHA256SUMS.txt'
if (!(Test-Path -LiteralPath $sumsPath)) { throw "PACK_SHA256SUMS.txt is missing next to this script; this is not an assembled pack." }
foreach ($line in Get-Content -LiteralPath $sumsPath) {
    if ($line -match '^\s*([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$') { $ExpectedSha[($Matches[2] -replace '/', '\')] = $Matches[1].ToUpperInvariant() }
}
$mustVerify = @(
    'runtimes\int8-openvino-rc4\onnx-totalseg.exe',
    'runtimes\int8-openvino-rc4\onnxruntime\onnxruntime.dll',
    'runtimes\int8-openvino-rc4\models\total_6mm_298_int8.onnx',
    'runtimes\int8-openvino-rc4\models\total_1p5mm_291_organs_int8.onnx')
if ($Arms -contains 'gate') { $mustVerify += 'runtimes\int8-openvino\onnx-totalseg.exe' }
foreach ($rel in $mustVerify) {
    if (!$ExpectedSha.ContainsKey($rel)) { throw "PACK_SHA256SUMS.txt has no entry for $rel" }
    $p = Join-Path $Root $rel
    if (!(Test-Path -LiteralPath $p)) { throw "Missing $rel" }
    $h = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($h -ne $ExpectedSha[$rel]) { throw "Hash mismatch for $rel; re-copy the pack." }
}

$NewExe    = Join-Path $Root 'runtimes\int8-openvino-rc4\onnx-totalseg.exe'
$NewOrt    = Join-Path $Root 'runtimes\int8-openvino-rc4\onnxruntime\onnxruntime.dll'
$NewModels = Join-Path $Root 'runtimes\int8-openvino-rc4\models'
$OldExe    = Join-Path $Root 'runtimes\int8-openvino\onnx-totalseg.exe'
$OldOrt    = Join-Path $Root 'runtimes\int8-openvino\onnxruntime\onnxruntime.dll'
$OldModels = Join-Path $Root 'runtimes\int8-openvino\models'

# ---------------------------------------------------------------- inputs
function Read-CaseList([string]$Path) { return @(Get-Content -LiteralPath $Path | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }

$AmosDir = Join-Path $Root 'input-nifti-amos100'
$AmosCases = Read-CaseList (Join-Path $Root 'AMOS100_CASES.txt')
$DukeNiftiDir = Join-Path $Root 'input-nifti-duke29'
$DukeDicomDir = Join-Path $Root 'input-dicom-duke29'
if (!$DukeCaseList) { $DukeCaseList = Join-Path $Root 'DUKE29_CASES.txt' }
$DukeCases = Read-CaseList $DukeCaseList

# Copying through a Mac can leave AppleDouble "._*" files beside each DICOM; they are not DICOM.
$junk = @(Get-ChildItem -LiteralPath $DukeDicomDir -Recurse -Force -File -Filter '._*' -ErrorAction SilentlyContinue)
foreach ($j in $junk) { Remove-Item -LiteralPath $j.FullName -Force }

function Get-ArmInputs([string]$Arm) {
    $list = @()
    switch ($Arm) {
        'amos_nifti' { foreach ($c in $AmosCases) { $list += [pscustomobject]@{ case = $c; path = (Join-Path $AmosDir "$c.nii.gz") } } }
        'duke_nifti' { foreach ($c in $DukeCases) { $list += [pscustomobject]@{ case = $c; path = (Join-Path $DukeNiftiDir "$c.nii.gz") } } }
        'duke_dicom' { foreach ($c in $DukeCases) { $list += [pscustomobject]@{ case = $c; path = (Join-Path $DukeDicomDir $c) } } }
    }
    foreach ($i in $list) { if (!(Test-Path -LiteralPath $i.path)) { throw "Input missing for arm ${Arm}: $($i.path)" } }
    return $list
}

if ($StartAt) {
    $target = [datetime]::ParseExact($StartAt, 'HH:mm', $null)
    if ($target -lt (Get-Date)) { $target = $target.AddDays(1) }
    Write-Host ("Waiting until {0:yyyy-MM-dd HH:mm} to start (leave this window open; screen lock is fine)." -f $target)
    while ((Get-Date) -lt $target) { Start-Sleep -Seconds 30 }
}

# Keep the workstation awake for the duration (no admin needed).
try {
    Add-Type -Namespace Native -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
    [void][Native.Power]::SetThreadExecutionState([uint32]0x80000003)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
} catch {}

# ---------------------------------------------------------------- run folder (new or resumed)
if ($Resume) { $RunRoot = (Resolve-Path -LiteralPath $Resume).Path }
else { $RunRoot = Join-Path $Root ('results\duke29_run_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$OutputRoot = Join-Path $RunRoot 'masks'
$LogRoot    = Join-Path $RunRoot 'logs'
$WarmupRoot = Join-Path $RunRoot 'warmup-discard'
$CsvPath    = Join-Path $RunRoot 'measurements.csv'
$JsonlPath  = Join-Path $RunRoot 'measurements.jsonl'
$WarmupJsonlPath = Join-Path $RunRoot 'warmup.jsonl'
$GateJsonlPath = Join-Path $RunRoot 'gate_runs.jsonl'
$GateCsvPath = Join-Path $RunRoot 'gate.csv'
$GateDecisionPath = Join-Path $RunRoot 'gate.json'
$PairCsvPath = Join-Path $RunRoot 'duke_dicom_vs_nifti.csv'
$SummaryPath = Join-Path $RunRoot 'summary.csv'
$DriverLog  = Join-Path $RunRoot 'driver.log'
New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot, $WarmupRoot | Out-Null

$Done = @{}
if (Test-Path -LiteralPath $JsonlPath) {
    foreach ($line in Get-Content -LiteralPath $JsonlPath) {
        if (!$line.Trim()) { continue }
        try { $r = $line | ConvertFrom-Json } catch { continue }
        if ($r.status -eq 'ok') { $Done[("{0}|{1}" -f $r.arm, $r.case)] = $true }
    }
}

function Write-DriverLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $DriverLog -Value $line -Encoding UTF8
    Write-Host $line
}
function Quote-Argument([string]$Value) {
    if ($null -eq $Value -or $Value.Length -eq 0) { return '""' }
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}
function Get-SystemUsedMb {
    try { $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop; return [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1024.0, 1) } catch { return 0.0 }
}
function Get-CimSafe([string]$ClassName) { try { return @(Get-CimInstance $ClassName -ErrorAction Stop) } catch { return @() } }
function Get-ProcessTreeSample([int]$RootPid) {
    try {
        $all = @(Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId, ParentProcessId, WorkingSetSize, PrivatePageCount, Name)
        $ids = New-Object 'System.Collections.Generic.HashSet[int]'
        [void]$ids.Add($RootPid)
        $changed = $true
        while ($changed) {
            $changed = $false
            foreach ($proc in $all) {
                if ($ids.Contains([int]$proc.ParentProcessId) -and -not $ids.Contains([int]$proc.ProcessId)) { [void]$ids.Add([int]$proc.ProcessId); $changed = $true }
            }
        }
        $members = @($all | Where-Object { $ids.Contains([int]$_.ProcessId) })
        $ws = 0L; $pv = 0L
        foreach ($m in $members) { if ($null -ne $m.WorkingSetSize) { $ws += [int64]$m.WorkingSetSize }; if ($null -ne $m.PrivatePageCount) { $pv += [int64]$m.PrivatePageCount } }
        return [pscustomobject]@{ working_set_mb = [math]::Round($ws / 1MB, 1); private_mb = [math]::Round($pv / 1MB, 1); process_count = $members.Count; process_names = (($members | ForEach-Object { $_.Name } | Sort-Object -Unique) -join ';') }
    } catch {
        try { $p = Get-Process -Id $RootPid -ErrorAction Stop; return [pscustomobject]@{ working_set_mb = [math]::Round($p.WorkingSet64 / 1MB, 1); private_mb = [math]::Round($p.PrivateMemorySize64 / 1MB, 1); process_count = 1; process_names = $p.ProcessName } }
        catch { return [pscustomobject]@{ working_set_mb = 0.0; private_mb = 0.0; process_count = 0; process_names = '' } }
    }
}
function Stop-ProcessTree([int]$RootPid) { try { & taskkill.exe /PID $RootPid /T /F 2>$null | Out-Null } catch {} }
function Add-JsonLine([string]$Path, $Object) { $Object | ConvertTo-Json -Compress -Depth 6 | Add-Content -LiteralPath $Path -Encoding UTF8 }
function Add-CsvRow([string]$Path, $Row) {
    if (Test-Path -LiteralPath $Path) { $Row | Export-Csv -LiteralPath $Path -NoTypeInformation -Append } else { $Row | Export-Csv -LiteralPath $Path -NoTypeInformation }
}
function Get-FileSha([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant() }
function Read-Gunzip([string]$Path) {
    $fs = [IO.File]::OpenRead($Path)
    try {
        $gz = New-Object IO.Compression.GZipStream($fs, [IO.Compression.CompressionMode]::Decompress)
        $ms = New-Object IO.MemoryStream
        $gz.CopyTo($ms)
        return , $ms.ToArray()
    } finally { $fs.Dispose() }
}
# Same mask in two NIfTI-1 files? Voxel bytes, dim/datatype, and the sform rows compared as numbers
# (the DICOM path can write -0.0 where the NIfTI path writes 0.0 in off-diagonal affine terms).
function Compare-NiftiMask([string]$A, [string]$B) {
    $x = Read-Gunzip $A; $y = Read-Gunzip $B
    $offX = [int][BitConverter]::ToSingle($x, 108); $offY = [int][BitConverter]::ToSingle($y, 108)
    $sha = [Security.Cryptography.SHA256]::Create()
    $voxX = [BitConverter]::ToString($sha.ComputeHash($x, $offX, $x.Length - $offX)) -replace '-', ''
    $voxY = [BitConverter]::ToString($sha.ComputeHash($y, $offY, $y.Length - $offY)) -replace '-', ''
    $geom = $true
    for ($i = 40; $i -lt 74; $i++) { if ($x[$i] -ne $y[$i]) { $geom = $false } }   # dim[8], intent, datatype, bitpix
    for ($i = 0; $i -lt 12; $i++) { if ([BitConverter]::ToSingle($x, 280 + 4 * $i) -ne [BitConverter]::ToSingle($y, 280 + 4 * $i)) { $geom = $false } }
    $hdrSame = ($offX -eq $offY)
    if ($hdrSame) { for ($i = 0; $i -lt $offX; $i++) { if ($x[$i] -ne $y[$i]) { $hdrSame = $false; break } } }
    return [pscustomobject]@{ voxels_identical = ($voxX -eq $voxY); geometry_identical = $geom; header_bytes_identical = $hdrSame; voxels_a_sha256 = $voxX; voxels_b_sha256 = $voxY }
}
# SHA-256 of the decompressed .nii.gz: NIfTI header plus voxels, independent of gzip framing.
function Get-GunzipSha([string]$Path) {
    $fs = [IO.File]::OpenRead($Path)
    try {
        $gz = New-Object IO.Compression.GZipStream($fs, [IO.Compression.CompressionMode]::Decompress)
        $sha = [Security.Cryptography.SHA256]::Create()
        return ([BitConverter]::ToString($sha.ComputeHash($gz)) -replace '-', '')
    } finally { $fs.Dispose() }
}

# Build: 'new' = rc4 release, 'old' = July build (gate only).
function Invoke-OneRun([string]$Arm, [string]$Build, [string]$CaseName, [string]$InputPath, [string]$OutputPath, [string]$Kind, [int]$WarmupIndex) {
    $safeName = (($Arm + '_' + $Build + '_' + $CaseName + $(if ($Kind -eq 'warmup') { "_warmup_$WarmupIndex" } else { '' })) -replace '[^A-Za-z0-9_.-]', '_')
    $stdoutPath = Join-Path $LogRoot ($safeName + '.stdout.log')
    $stderrPath = Join-Path $LogRoot ($safeName + '.stderr.log')
    if ($Build -eq 'old') { $exe = $OldExe; $ort = $OldOrt; $models = $OldModels } else { $exe = $NewExe; $ort = $NewOrt; $models = $NewModels }
    $argList = @('segment', '--in', $InputPath, '--out', $OutputPath, '--device', 'openvino', '--int8', '--roi_subset', 'spleen', '--ort-lib', $ort, '--models-dir', $models)
    $argumentText = (($argList | ForEach-Object { Quote-Argument $_ }) -join ' ')

    $baselineSystem = Get-SystemUsedMb; $peakSystem = $baselineSystem
    $peakWorkingSet = 0.0; $peakPrivate = 0.0; $maxProcessCount = 0; $processNames = ''; $timedOut = $false
    $started = Get-Date
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $exe -ArgumentList $argumentText -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    try {
        while (!$process.HasExited) {
            $s = Get-ProcessTreeSample $process.Id
            $peakWorkingSet = [math]::Max($peakWorkingSet, $s.working_set_mb)
            $peakPrivate = [math]::Max($peakPrivate, $s.private_mb)
            $peakSystem = [math]::Max($peakSystem, (Get-SystemUsedMb))
            $maxProcessCount = [math]::Max($maxProcessCount, $s.process_count)
            if ($s.process_names) { $processNames = $s.process_names }
            if ($TimeoutSeconds -gt 0 -and $watch.Elapsed.TotalSeconds -ge $TimeoutSeconds) { $timedOut = $true; Stop-ProcessTree $process.Id; break }
            Start-Sleep -Milliseconds $SampleMilliseconds
            $process.Refresh()
        }
        $process.WaitForExit()
    } finally { $watch.Stop() }
    $process.Refresh()
    $rawExit = $process.ExitCode
    $exitCode = if ($timedOut) { -1 } elseif ($null -ne $rawExit) { $rawExit } elseif (Test-Path -LiteralPath $OutputPath) { 0 } else { -2 }
    $status = if ($timedOut) { 'timeout' } elseif ($exitCode -ne 0) { 'failed' } elseif (!(Test-Path -LiteralPath $OutputPath)) { 'missing_output' } else { 'ok' }

    $row = [pscustomobject][ordered]@{
        timestamp = $started.ToString('s'); host = $env:COMPUTERNAME; arm = $Arm; build = $Build; case = $CaseName
        warmup = ($Kind -eq 'warmup'); warmup_index = $(if ($Kind -eq 'warmup') { $WarmupIndex } else { '' }); status = $status
        wall_seconds = [math]::Round($watch.Elapsed.TotalSeconds, 3)
        baseline_system_used_mb = $baselineSystem; peak_system_used_mb = $peakSystem
        delta_system_used_mb = [math]::Round([math]::Max(0.0, $peakSystem - $baselineSystem), 1)
        peak_tree_working_set_mb = $peakWorkingSet; peak_tree_private_mb = $peakPrivate
        max_process_count = $maxProcessCount; process_names = $processNames; exit_code = $exitCode
        input = $InputPath; output = $OutputPath; stdout_log = $stdoutPath; stderr_log = $stderrPath
        command = (Quote-Argument $exe) + ' ' + $argumentText
    }
    switch ($Kind) {
        'warmup'  { Add-JsonLine $WarmupJsonlPath $row }
        'gate'    { Add-JsonLine $GateJsonlPath $row }
        default   { Add-JsonLine $JsonlPath $row; Add-CsvRow $CsvPath $row }
    }
    Write-DriverLog ("{0} arm={1} build={2} case={3} status={4} wall={5}s peak_ws={6}MB" -f $Kind.ToUpperInvariant(), $Arm, $Build, $CaseName, $status, $row.wall_seconds, $row.peak_tree_working_set_mb)
    return $row
}

# ---------------------------------------------------------------- machine record
$machine = [ordered]@{
    generated_at = (Get-Date).ToString('o'); computer_name = $env:COMPUTERNAME; pack_root = $Root; resumed = [bool]$Resume
    windows = @(Get-CimSafe 'Win32_OperatingSystem' | Select-Object Caption, Version, BuildNumber, OSArchitecture)
    cpu = @(Get-CimSafe 'Win32_Processor' | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed)
    memory = @(Get-CimSafe 'Win32_PhysicalMemory' | Select-Object Manufacturer, PartNumber, Capacity, Speed)
    disks = @(Get-CimSafe 'Win32_DiskDrive' | Select-Object Model, MediaType, InterfaceType, Size)
    powershell = $PSVersionTable.PSVersion.ToString()
    new_exe_sha256 = (Get-FileSha $NewExe); old_exe_sha256 = $(if (Test-Path -LiteralPath $OldExe) { Get-FileSha $OldExe } else { '' })
    removed_appledouble_files = $junk.Count
    settings = [ordered]@{ arms = $Arms; gate_cases = $GateCases; force_amos = [bool]$ForceAmos; duke_case_list = $DukeCaseList; duke_cases = $DukeCases.Count; warmup_runs = $WarmupRuns; sample_milliseconds = $SampleMilliseconds; cooldown_seconds = $CooldownSeconds; timeout_seconds = $TimeoutSeconds; roi = 'spleen'; force_split = $false; order = 'arm-major'; openvino_inference_precision = 'f32' }
}
$machine | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RunRoot ('machine' + $(if ($Resume) { '_resume_' + (Get-Date -Format 'yyyyMMdd_HHmmss') } else { '' }) + '.json')) -Encoding UTF8

# ---------------------------------------------------------------- run
$oldPrecision = $env:ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION
$env:ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION = 'f32'
$failed = $false
$gateIdentical = $null
try {
    Write-DriverLog "Pack root (hash-verified): $Root"
    Write-DriverLog ("Arms: {0}; Duke cases: {1}; already ok: {2}; removed {3} AppleDouble files" -f ($Arms -join ','), $DukeCases.Count, $Done.Count, $junk.Count)

    # ---- gate: July exe vs rc4 exe on the first GateCases AMOS cases
    if ($Arms -contains 'gate') {
        if (Test-Path -LiteralPath $GateDecisionPath) {
            $gateIdentical = [bool]((Get-Content -LiteralPath $GateDecisionPath -Raw | ConvertFrom-Json).identical)
            Write-DriverLog "Gate already decided in this run folder: identical=$gateIdentical"
        } else {
            $gateDir = Join-Path $OutputRoot 'gate'
            New-Item -ItemType Directory -Force -Path $gateDir | Out-Null
            $gateList = @($AmosCases | Select-Object -First $GateCases)
            $gateIdentical = $true
            foreach ($c in $gateList) {
                $in = Join-Path $AmosDir "$c.nii.gz"
                $outs = @{}
                $gateRunsOk = $true
                foreach ($b in @('old', 'new')) {
                    $o = Join-Path $gateDir ("{0}_{1}_spleen.nii.gz" -f $c, $b)
                    if (Test-Path -LiteralPath $o) { Remove-Item -LiteralPath $o -Force }
                    $r = Invoke-OneRun 'gate' $b $c $in $o 'gate' 0
                    if ($r.status -ne 'ok') { $gateRunsOk = $false; $failed = $true; Write-DriverLog "Gate run failed (build=$b case=$c); counting as a difference. See $($r.stderr_log)" }
                    $outs[$b] = $o
                }
                if (!$gateRunsOk) { $gateIdentical = $false; continue }
                $oldRaw = Get-FileSha $outs['old']; $newRaw = Get-FileSha $outs['new']
                $oldNii = Get-GunzipSha $outs['old']; $newNii = Get-GunzipSha $outs['new']
                $same = ($oldNii -eq $newNii)
                if (!$same) { $gateIdentical = $false }
                Add-CsvRow $GateCsvPath ([pscustomobject][ordered]@{ case = $c; identical_nifti = $same; identical_file = ($oldRaw -eq $newRaw); old_nifti_sha256 = $oldNii; new_nifti_sha256 = $newNii; old_file_sha256 = $oldRaw; new_file_sha256 = $newRaw })
                Write-DriverLog ("GATE case={0} identical={1}" -f $c, $same)
            }
            [pscustomobject][ordered]@{ identical = $gateIdentical; cases = $gateList; old_exe_sha256 = $machine.old_exe_sha256; new_exe_sha256 = $machine.new_exe_sha256 } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $GateDecisionPath -Encoding UTF8
        }
        if ($gateIdentical) { Write-DriverLog ("GATE PASSED: rc4 masks are byte-identical to the July build; " + $(if ($ForceAmos) { "-ForceAmos: running AMOS 100 anyway." } else { "AMOS 100 is not re-run." })) }
        else {
            Write-DriverLog "GATE FAILED: rc4 masks differ from the July build; adding the AMOS 100 arm."
            if ($Arms -notcontains 'amos_nifti') { $Arms = @($Arms) + 'amos_nifti' }
        }
    }

    # ---- timed arms (amos_nifti first if it runs, then Duke NIfTI, then Duke DICOM)
    foreach ($arm in @('amos_nifti', 'duke_nifti', 'duke_dicom')) {
        if ($Arms -notcontains $arm) { continue }
        $inputs = Get-ArmInputs $arm
        $pending = @($inputs | Where-Object { -not $Done.ContainsKey("$arm|$($_.case)") })
        if ($pending.Count -eq 0) { Write-DriverLog "arm=$arm already complete; skipping"; continue }
        $warmOk = $true
        for ($i = 1; $i -le $WarmupRuns; $i++) {
            $wo = Join-Path $WarmupRoot ("{0}_{1}_{2}.nii.gz" -f $arm, $inputs[0].case, $i)
            $r = Invoke-OneRun $arm 'new' $inputs[0].case $inputs[0].path $wo 'warmup' $i
            if ($r.status -ne 'ok') { $warmOk = $false }
        }
        if (!$warmOk) { Write-DriverLog "WARM-UP FAILED for $arm; skipping this arm. See logs\."; $failed = $true; continue }
        $armOutputDir = Join-Path $OutputRoot $arm
        New-Item -ItemType Directory -Force -Path $armOutputDir | Out-Null
        foreach ($input in $pending) {
            $out = Join-Path $armOutputDir ($input.case + '_spleen.nii.gz')
            if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
            $r = Invoke-OneRun $arm 'new' $input.case $input.path $out 'measure' 0
            if ($r.status -ne 'ok') { $failed = $true }
            if ($CooldownSeconds -gt 0) { Start-Sleep -Seconds $CooldownSeconds }
        }
    }
} finally {
    $env:ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION = $oldPrecision
    try { [void][Native.Power]::SetThreadExecutionState([uint32]0x80000000) } catch {}
}

# ---------------------------------------------------------------- Duke: DICOM-input vs NIfTI-input masks
$niftiMaskDir = Join-Path $OutputRoot 'duke_nifti'
$dicomMaskDir = Join-Path $OutputRoot 'duke_dicom'
if ((Test-Path -LiteralPath $niftiMaskDir) -and (Test-Path -LiteralPath $dicomMaskDir)) {
    if (Test-Path -LiteralPath $PairCsvPath) { Remove-Item -LiteralPath $PairCsvPath -Force }
    $nSame = 0; $nPairs = 0
    foreach ($c in $DukeCases) {
        $a = Join-Path $niftiMaskDir "${c}_spleen.nii.gz"; $b = Join-Path $dicomMaskDir "${c}_spleen.nii.gz"
        if (!(Test-Path -LiteralPath $a) -or !(Test-Path -LiteralPath $b)) { continue }
        $cmp = Compare-NiftiMask $a $b; $nPairs++
        $same = ($cmp.voxels_identical -and $cmp.geometry_identical)
        if ($same) { $nSame++ }
        Add-CsvRow $PairCsvPath ([pscustomobject][ordered]@{ case = $c; identical_mask = $same; voxels_identical = $cmp.voxels_identical; geometry_identical = $cmp.geometry_identical; header_bytes_identical = $cmp.header_bytes_identical; nifti_input_voxels_sha256 = $cmp.voxels_a_sha256; dicom_input_voxels_sha256 = $cmp.voxels_b_sha256 })
    }
    Write-DriverLog "Duke DICOM-input vs NIfTI-input masks: $nSame of $nPairs identical (voxels, dimensions and affine)."
}

# ---------------------------------------------------------------- summary (ok rows, one row per arm/case)
if (Test-Path -LiteralPath $CsvPath) {
    $rows = @(Import-Csv -LiteralPath $CsvPath)
    $summary = foreach ($g in ($rows | Group-Object arm)) {
        $ok = @($g.Group | Where-Object { $_.status -eq 'ok' } | Group-Object case | ForEach-Object { $_.Group | Select-Object -Last 1 })
        $t = @($ok | ForEach-Object { [double]$_.wall_seconds } | Sort-Object)
        $median = if ($t.Count -eq 0) { 0.0 } elseif ($t.Count % 2 -eq 1) { $t[[int][math]::Floor($t.Count / 2)] } else { ($t[$t.Count / 2 - 1] + $t[$t.Count / 2]) / 2.0 }
        [pscustomobject][ordered]@{
            arm = $g.Name; cases_ok = $ok.Count; cases_expected = @(Get-ArmInputs $g.Name).Count
            mean_wall_seconds = $(if ($t.Count) { [math]::Round(($t | Measure-Object -Average).Average, 3) } else { 0.0 })
            median_wall_seconds = [math]::Round($median, 3)
            max_wall_seconds = $(if ($t.Count) { [math]::Round(($t | Measure-Object -Maximum).Maximum, 3) } else { 0.0 })
            max_tree_working_set_mb = $(if ($ok.Count) { [math]::Round(($ok | Measure-Object peak_tree_working_set_mb -Maximum).Maximum, 1) } else { 0.0 })
            max_tree_private_mb = $(if ($ok.Count) { [math]::Round(($ok | Measure-Object peak_tree_private_mb -Maximum).Maximum, 1) } else { 0.0 })
        }
    }
    $summary | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation
    $summary | Format-Table -AutoSize
}
Remove-Item -LiteralPath $WarmupRoot -Recurse -Force -ErrorAction SilentlyContinue
Write-DriverLog "Complete. Copy this folder back: $RunRoot"
if ($failed) { Write-Host "Some runs did not finish ok. Re-run with -Resume `"$RunRoot`" to fill them in."; exit 1 }
