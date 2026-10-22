<#
.SYNOPSIS
  Unattended 100-case AMOS spleen crop-cascade benchmark on the PACS reading station.

  Three arms, run one after another (arm-major so INT8 finishes first):
    int8_openvino   onnx-totalseg.exe, OpenVINO CPU, NNCF INT8 + FP32 remainder
    stock_ts_cpu    TotalSegmentator 2.4.0 PyInstaller package, --device cpu
    stock_ts_cuda   TotalSegmentator 2.4.0 PyInstaller package, --device gpu (Quadro P400)

  Same protocol as the July 20 AMOS-20 run (Run-Experiment.ps1): one unmeasured warm-up per arm,
  sequential runs, 250 ms process-tree sampling, 5 s cooldown, no --force_split, INT8 remainder
  pinned to f32. Output schema matches measurements.jsonl from that run, plus the GPU fields the
  CUDA-only run recorded (baseline/peak/delta GPU MB, cpu_fallback_detected).

.DESCRIPTION
  Finds the correct experiment pack by hash, not by folder name: it looks for a folder containing
  runtimes\int8-openvino\onnx-totalseg.exe whose exe and two INT8 models match INT8_SHA256SUMS.txt
  from the July delivery. Pass -PackRoot to skip the search.

  Resumable: re-running with -Resume <results folder> skips every (arm, case) already recorded ok.
  A per-run timeout (default 15 min) records a failure and moves on instead of hanging all night.

.EXAMPLE
  # From the folder that holds input-nifti-amos100\ and AMOS100_CASES.txt, start at 20:00 tonight:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-PacsAmos100.ps1 -StartAt 20:00

.EXAMPLE
  # Register a one-shot Task Scheduler job instead of leaving a window open:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-PacsAmos100.ps1 -RegisterTask 20:00

.EXAMPLE
  # Resume an interrupted run:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-PacsAmos100.ps1 -Resume .\results\amos100_run_20260905_200000
#>
[CmdletBinding()]
param(
    [string]$PackRoot = "",
    [string]$InputDir = "",
    [string]$CaseList = "",
    [string[]]$Arms = @('int8_openvino', 'stock_ts_cpu', 'stock_ts_cuda'),
    [string]$StartAt = "",
    [string]$RegisterTask = "",
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

$Here = (Resolve-Path -LiteralPath $PSScriptRoot).Path

# powershell.exe -File passes "-Arms a,b" as one string; normalize to an array and validate.
$Arms = @($Arms | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$KnownArms = @('int8_openvino', 'stock_ts_cpu', 'stock_ts_cuda')
foreach ($a in $Arms) { if ($KnownArms -notcontains $a) { throw "Unknown arm '$a'. Valid: $($KnownArms -join ', ')" } }
if ($Arms.Count -eq 0) { throw "No arms requested." }

# ---------------------------------------------------------------- expected INT8 artifact hashes
# Read from PACK_SHA256SUMS.txt next to this script (written when the pack was assembled), so the
# pack verifies itself. The two model hashes are the July delivery's (INT8_SHA256SUMS.txt); the exe
# hash is whichever build was packed, and is recorded in the results' machine.json.
$ExpectedSha = @{}
$sumsPath = Join-Path $Here 'PACK_SHA256SUMS.txt'
if (!(Test-Path -LiteralPath $sumsPath)) { throw "PACK_SHA256SUMS.txt is missing next to this script; this is not an assembled pack." }
foreach ($line in Get-Content -LiteralPath $sumsPath) {
    if ($line -match '^\s*([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$') { $ExpectedSha[($Matches[2] -replace '/', '\')] = $Matches[1].ToUpperInvariant() }
}
foreach ($must in @('runtimes\int8-openvino\onnx-totalseg.exe', 'runtimes\int8-openvino\models\total_6mm_298_int8.onnx', 'runtimes\int8-openvino\models\total_1p5mm_291_organs_int8.onnx')) {
    if (!$ExpectedSha.ContainsKey($must)) { throw "PACK_SHA256SUMS.txt has no entry for $must" }
}

$NeedInt8  = [bool]($Arms -contains 'int8_openvino')
$NeedStock = [bool](($Arms -contains 'stock_ts_cpu') -or ($Arms -contains 'stock_ts_cuda'))
function Test-PackRoot([string]$Candidate) {
    # Verify only what the requested arms will execute.
    foreach ($rel in $ExpectedSha.Keys) {
        if ($rel -notlike 'runtimes\*') { continue }   # the sums file also lists this script; not a runtime check
        $isInt8 = $rel -like 'runtimes\int8-openvino\*'
        $isStock = $rel -like 'runtimes\stock-ts\*'
        if (($isInt8 -and -not $NeedInt8) -or ($isStock -and -not $NeedStock)) { continue }
        $p = Join-Path $Candidate $rel
        if (!(Test-Path -LiteralPath $p)) { return "missing $rel" }
        $h = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($h -ne $ExpectedSha[$rel]) { return "hash mismatch for $rel" }
    }
    if ($NeedStock -and !(Test-Path -LiteralPath (Join-Path $Candidate 'runtimes\stock-ts\run.bat'))) { return "missing runtimes\stock-ts\run.bat" }
    return $null
}

$script:PackTried = @()
$script:PackSeen = @{}
function Probe-PackRoot([string]$Candidate) {
    if (!$Candidate -or $script:PackSeen.ContainsKey($Candidate)) { return $null }
    $script:PackSeen[$Candidate] = $true
    $why = Test-PackRoot $Candidate
    if ($null -eq $why) { return $Candidate }
    $script:PackTried += "  $Candidate -> $why"
    return $null
}

function Find-PackRoot {
    # Try the cheap locations first and return as soon as one verifies; only then widen the search.
    if ($PackRoot) { $r = Probe-PackRoot ((Resolve-Path -LiteralPath $PackRoot).Path); if ($r) { return $r } }
    $r = Probe-PackRoot $Here; if ($r) { return $r }
    $searchRoots = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('UserProfile'), 'C:\', 'D:\') | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    foreach ($root in $searchRoots) {
        Write-Host "Searching $root for the experiment pack..."
        $hits = @(Get-ChildItem -LiteralPath $root -Recurse -Depth 4 -Filter 'onnx-totalseg.exe' -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\runtimes\\int8-openvino\\onnx-totalseg\.exe$' })
        foreach ($hit in $hits) {
            $r = Probe-PackRoot (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $hit.FullName)))
            if ($r) { return $r }
        }
    }
    throw ("No experiment pack matched the July INT8 hashes. Tried:`n" + ($script:PackTried -join "`n") + "`nPass -PackRoot <folder containing runtimes\>.")
}

# ---------------------------------------------------------------- optional scheduling
if ($RegisterTask) {
    $psExe = (Get-Command powershell.exe).Source
    $argLine = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    foreach ($kv in @(@('PackRoot', $PackRoot), @('InputDir', $InputDir), @('CaseList', $CaseList))) {
        if ($kv[1]) { $argLine += " -$($kv[0]) `"$($kv[1])`"" }
    }
    $argLine += " -Arms " + ($Arms -join ',')
    $taskName = 'PacsAmos100Benchmark'
    & schtasks.exe /Create /F /SC ONCE /ST $RegisterTask /TN $taskName /TR "`"$psExe`" $argLine" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "schtasks failed (exit $LASTEXITCODE). Use -StartAt instead and leave this window open." }
    Write-Host "Registered task '$taskName' for $RegisterTask today. Check with: schtasks /Query /TN $taskName"
    exit 0
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

# ---------------------------------------------------------------- locate everything
$Root = Find-PackRoot
$Int8Exe   = Join-Path $Root 'runtimes\int8-openvino\onnx-totalseg.exe'
$OrtDll    = Join-Path $Root 'runtimes\int8-openvino\onnxruntime\onnxruntime.dll'
$Int8Models = Join-Path $Root 'runtimes\int8-openvino\models'
$TsLauncher = Join-Path $Root 'runtimes\stock-ts\run.bat'
if (!(Test-Path -LiteralPath $OrtDll)) { throw "ONNX Runtime DLL missing: $OrtDll" }

if (!$InputDir) {
    foreach ($c in @((Join-Path $Here 'input-nifti-amos100'), (Join-Path $Root 'input-nifti-amos100'))) { if (Test-Path -LiteralPath $c) { $InputDir = $c; break } }
    if (!$InputDir) { throw "input-nifti-amos100 not found next to this script or the pack. Pass -InputDir." }
}
$InputDir = (Resolve-Path -LiteralPath $InputDir).Path
if (!$CaseList) {
    foreach ($c in @((Join-Path $Here 'AMOS100_CASES.txt'), (Join-Path $InputDir 'AMOS100_CASES.txt'), (Join-Path (Split-Path -Parent $InputDir) 'AMOS100_CASES.txt'))) { if (Test-Path -LiteralPath $c) { $CaseList = $c; break } }
}
$Discovered = @(Get-ChildItem -LiteralPath $InputDir -File | Where-Object { $_.Name -match '(?i)\.nii(?:\.gz)?$' } | Sort-Object Name)
if ($Discovered.Count -eq 0) { throw "No NIfTI inputs in $InputDir" }
if ($CaseList) {
    $Inputs = @()
    foreach ($caseName in @(Get-Content -LiteralPath $CaseList | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $m = @($Discovered | Where-Object { $_.Name -ieq ($caseName + '.nii.gz') -or $_.Name -ieq ($caseName + '.nii') })
        if ($m.Count -ne 1) { throw "Expected exactly one input for $caseName in $InputDir; found $($m.Count)" }
        $Inputs += $m[0]
    }
} else { $Inputs = $Discovered }

$NvidiaSmi = $null
foreach ($c in @("$env:SystemRoot\System32\nvidia-smi.exe", "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe")) { if (Test-Path -LiteralPath $c) { $NvidiaSmi = $c; break } }
if (($Arms -contains 'stock_ts_cuda') -and -not $NvidiaSmi) { Write-Warning "nvidia-smi not found; CUDA arm will run without GPU memory sampling." }

# ---------------------------------------------------------------- run folder (new or resumed)
if ($Resume) {
    $RunRoot = (Resolve-Path -LiteralPath $Resume).Path
} else {
    $RunRoot = Join-Path $Root ('results\amos100_run_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$OutputRoot = Join-Path $RunRoot 'masks'
$LogRoot    = Join-Path $RunRoot 'logs'
$WarmupRoot = Join-Path $RunRoot 'warmup-discard'
$CsvPath    = Join-Path $RunRoot 'measurements.csv'
$JsonlPath  = Join-Path $RunRoot 'measurements.jsonl'
$WarmupJsonlPath = Join-Path $RunRoot 'warmup.jsonl'
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
function Get-CaseStem([string]$Path) {
    $name = [IO.Path]::GetFileName($Path)
    if ($name -match '^(.*)\.nii\.gz$') { return $Matches[1] }
    if ($name -match '^(.*)\.nii$') { return $Matches[1] }
    return [IO.Path]::GetFileNameWithoutExtension($Path)
}
function Get-SystemUsedMb {
    try { $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop; return [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1024.0, 1) } catch { return 0.0 }
}
function Get-CimSafe([string]$ClassName) { try { return @(Get-CimInstance $ClassName -ErrorAction Stop) } catch { return @() } }
function Get-GpuUsedMb {
    if (!$NvidiaSmi) { return $null }
    try {
        $v = (& $NvidiaSmi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if ($v -match '^\s*(\d+)') { return [int]$Matches[1] }
    } catch {}
    return $null
}
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
function Save-Measurement($Row) {
    Add-JsonLine $JsonlPath $Row
    if (Test-Path -LiteralPath $CsvPath) { $Row | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Append } else { $Row | Export-Csv -LiteralPath $CsvPath -NoTypeInformation }
}

function Invoke-OneRun([string]$Arm, [string]$CaseName, [string]$InputPath, [string]$OutputPath, [bool]$IsWarmup, [int]$WarmupIndex) {
    $safeName = (($Arm + '_' + $CaseName + $(if ($IsWarmup) { "_warmup_$WarmupIndex" } else { '' })) -replace '[^A-Za-z0-9_.-]', '_')
    $stdoutPath = Join-Path $LogRoot ($safeName + '.stdout.log')
    $stderrPath = Join-Path $LogRoot ($safeName + '.stderr.log')

    switch ($Arm) {
        'int8_openvino' {
            $filePath = $Int8Exe
            $args = @('segment', '--in', $InputPath, '--out', $OutputPath, '--device', 'openvino', '--int8', '--roi_subset', 'spleen', '--ort-lib', $OrtDll, '--models-dir', $Int8Models)
        }
        'stock_ts_cpu' {
            $filePath = $env:ComSpec
            $ts = ((@($TsLauncher, '-i', $InputPath, '-o', $OutputPath, '--roi_subset', 'spleen', '--ml', '--device', 'cpu') | ForEach-Object { Quote-Argument $_ }) -join ' ')
            $args = @('/d', '/s', '/c', $ts)
        }
        'stock_ts_cuda' {
            $filePath = $env:ComSpec
            $ts = ((@($TsLauncher, '-i', $InputPath, '-o', $OutputPath, '--roi_subset', 'spleen', '--ml', '--device', 'gpu') | ForEach-Object { Quote-Argument $_ }) -join ' ')
            $args = @('/d', '/s', '/c', $ts)
        }
        default { throw "Unknown arm: $Arm" }
    }
    $isCuda = ($Arm -eq 'stock_ts_cuda')
    $argumentText = (($args | ForEach-Object { Quote-Argument $_ }) -join ' ')
    $displayCommand = (Quote-Argument $filePath) + ' ' + $argumentText

    $baselineSystem = Get-SystemUsedMb; $peakSystem = $baselineSystem
    $baselineGpu = $(if ($isCuda) { Get-GpuUsedMb } else { $null }); $peakGpu = $baselineGpu
    $peakWorkingSet = 0.0; $peakPrivate = 0.0; $maxProcessCount = 0; $processNames = ''; $timedOut = $false
    $started = Get-Date
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $filePath -ArgumentList $argumentText -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    try {
        while (!$process.HasExited) {
            $s = Get-ProcessTreeSample $process.Id
            $peakWorkingSet = [math]::Max($peakWorkingSet, $s.working_set_mb)
            $peakPrivate = [math]::Max($peakPrivate, $s.private_mb)
            $peakSystem = [math]::Max($peakSystem, (Get-SystemUsedMb))
            $maxProcessCount = [math]::Max($maxProcessCount, $s.process_count)
            if ($s.process_names) { $processNames = $s.process_names }
            if ($isCuda -and $null -ne $baselineGpu) { $g = Get-GpuUsedMb; if ($null -ne $g) { $peakGpu = [math]::Max($peakGpu, $g) } }
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

    $fallback = $false
    if ($isCuda) {
        $text = ''
        foreach ($p in @($stdoutPath, $stderrPath)) { if (Test-Path -LiteralPath $p) { $text += (Get-Content -LiteralPath $p -Raw -ErrorAction SilentlyContinue) } }
        if ($text -match '(?i)no gpu|running on cpu|cuda is not available|falling back to cpu') { $fallback = $true; if ($status -eq 'ok') { $status = 'cpu_fallback' } }
    }

    $row = [ordered]@{
        timestamp = $started.ToString('s'); host = $env:COMPUTERNAME; arm = $Arm; case = $CaseName
        warmup = $IsWarmup; warmup_index = $(if ($IsWarmup) { $WarmupIndex } else { '' }); status = $status
        wall_seconds = [math]::Round($watch.Elapsed.TotalSeconds, 3)
        baseline_system_used_mb = $baselineSystem; peak_system_used_mb = $peakSystem
        delta_system_used_mb = [math]::Round([math]::Max(0.0, $peakSystem - $baselineSystem), 1)
        peak_tree_working_set_mb = $peakWorkingSet; peak_tree_private_mb = $peakPrivate
        max_process_count = $maxProcessCount; process_names = $processNames; exit_code = $exitCode
    }
    if ($isCuda) {
        $row['baseline_gpu_mb'] = $baselineGpu; $row['peak_gpu_mb'] = $peakGpu
        $row['delta_gpu_mb'] = $(if ($null -ne $baselineGpu -and $null -ne $peakGpu) { [math]::Max(0, $peakGpu - $baselineGpu) } else { $null })
        $row['cpu_fallback_detected'] = $fallback
    }
    $row['input'] = $InputPath; $row['output'] = $OutputPath; $row['stdout_log'] = $stdoutPath; $row['stderr_log'] = $stderrPath; $row['command'] = $displayCommand
    $row = [pscustomobject]$row
    if ($IsWarmup) { Add-JsonLine $WarmupJsonlPath $row } else { Save-Measurement $row }
    Write-DriverLog ("{0} arm={1} case={2} status={3} wall={4}s peak_ws={5}MB{6}" -f $(if ($IsWarmup) { 'WARMUP' } else { 'MEASURE' }), $Arm, $CaseName, $status, $row.wall_seconds, $row.peak_tree_working_set_mb, $(if ($isCuda) { " peak_gpu=$($peakGpu)MB" } else { '' }))
    return $row
}

# ---------------------------------------------------------------- machine record
$machine = [ordered]@{
    generated_at = (Get-Date).ToString('o'); computer_name = $env:COMPUTERNAME; pack_root = $Root; input_dir = $InputDir; resumed = [bool]$Resume
    windows = @(Get-CimSafe 'Win32_OperatingSystem' | Select-Object Caption, Version, BuildNumber, OSArchitecture)
    cpu = @(Get-CimSafe 'Win32_Processor' | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed)
    memory = @(Get-CimSafe 'Win32_PhysicalMemory' | Select-Object Manufacturer, PartNumber, Capacity, Speed)
    gpu = $(if ($NvidiaSmi) { @(& $NvidiaSmi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>$null) } else { @() })
    powershell = $PSVersionTable.PSVersion.ToString()
    int8_model_format = 'NNCF accuracy-controlled QDQ INT8 with FP32 remainder'; stock_ts_version = '2.4.0'
    settings = [ordered]@{ arms = $Arms; warmup_runs = $WarmupRuns; sample_milliseconds = $SampleMilliseconds; cooldown_seconds = $CooldownSeconds; timeout_seconds = $TimeoutSeconds; roi = 'spleen'; force_split = $false; order = 'arm-major' }
}
$machine | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RunRoot ('machine' + $(if ($Resume) { '_resume_' + (Get-Date -Format 'yyyyMMdd_HHmmss') } else { '' }) + '.json')) -Encoding UTF8

# ---------------------------------------------------------------- run
$oldPrecision = $env:ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION
$env:ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION = 'f32'
$failed = $false
try {
    Write-DriverLog "Pack root (hash-verified): $Root"
    Write-DriverLog ("Inputs: {0} cases from {1}; arms: {2}; already ok: {3}" -f $Inputs.Count, $InputDir, ($Arms -join ','), $Done.Count)
    Write-DriverLog "Sequential, arm-major, no --force_split, INT8 remainder pinned to f32."
    $warmupInput = $Inputs[0]; $warmupCase = Get-CaseStem $warmupInput.FullName
    foreach ($arm in $Arms) {
        $pending = @($Inputs | Where-Object { -not $Done.ContainsKey("$arm|" + (Get-CaseStem $_.FullName)) })
        if ($pending.Count -eq 0) { Write-DriverLog "arm=$arm already complete; skipping"; continue }
        for ($i = 1; $i -le $WarmupRuns; $i++) {
            $wo = Join-Path $WarmupRoot ("{0}_{1}_{2}.nii.gz" -f $arm, $warmupCase, $i)
            $r = Invoke-OneRun $arm $warmupCase $warmupInput.FullName $wo $true $i
            if ($r.status -ne 'ok') { Write-DriverLog "WARM-UP FAILED for $arm; skipping this arm. See $($r.stderr_log)"; $failed = $true; continue }
        }
        $armOutputDir = Join-Path $OutputRoot $arm
        New-Item -ItemType Directory -Force -Path $armOutputDir | Out-Null
        foreach ($input in $pending) {
            $caseName = Get-CaseStem $input.FullName
            $out = Join-Path $armOutputDir ($caseName + '_spleen.nii.gz')
            if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
            $r = Invoke-OneRun $arm $caseName $input.FullName $out $false 0
            if ($r.status -ne 'ok') { $failed = $true }
            if ($CooldownSeconds -gt 0) { Start-Sleep -Seconds $CooldownSeconds }
        }
    }
} finally {
    $env:ONNX_TOTALSEG_OPENVINO_INFERENCE_PRECISION = $oldPrecision
    try { [void][Native.Power]::SetThreadExecutionState([uint32]0x80000000) } catch {}
}

# ---------------------------------------------------------------- summary (ok rows, one row per arm/case)
$rows = @(Import-Csv -LiteralPath $CsvPath)
$summary = foreach ($g in ($rows | Group-Object arm)) {
    $ok = @($g.Group | Where-Object { $_.status -eq 'ok' } | Group-Object case | ForEach-Object { $_.Group | Select-Object -Last 1 })
    $t = @($ok | ForEach-Object { [double]$_.wall_seconds } | Sort-Object)
    $median = if ($t.Count -eq 0) { 0.0 } elseif ($t.Count % 2 -eq 1) { $t[[int][math]::Floor($t.Count / 2)] } else { ($t[$t.Count / 2 - 1] + $t[$t.Count / 2]) / 2.0 }
    [pscustomobject][ordered]@{
        arm = $g.Name; cases_ok = $ok.Count; cases_expected = $Inputs.Count
        mean_wall_seconds = $(if ($t.Count) { [math]::Round(($t | Measure-Object -Average).Average, 3) } else { 0.0 })
        median_wall_seconds = [math]::Round($median, 3)
        max_wall_seconds = $(if ($t.Count) { [math]::Round(($t | Measure-Object -Maximum).Maximum, 3) } else { 0.0 })
        max_tree_working_set_mb = $(if ($ok.Count) { [math]::Round(($ok | Measure-Object peak_tree_working_set_mb -Maximum).Maximum, 1) } else { 0.0 })
        max_tree_private_mb = $(if ($ok.Count) { [math]::Round(($ok | Measure-Object peak_tree_private_mb -Maximum).Maximum, 1) } else { 0.0 })
    }
}
$summary | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation
Remove-Item -LiteralPath $WarmupRoot -Recurse -Force -ErrorAction SilentlyContinue
Write-DriverLog "Complete. Copy this folder back: $RunRoot"
$summary | Format-Table -AutoSize
if ($failed) { Write-Host "Some runs did not finish ok. Re-run with -Resume `"$RunRoot`" to fill them in."; exit 1 }
