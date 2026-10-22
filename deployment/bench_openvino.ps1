<#
  Time a command and sample the working set/private memory of its process tree.

  Example:
    .\bench_openvino.ps1 -Label int8_volume-0 .\run_openvino.ps1 `
      -InputPath .\data\volume-0.nii.gz -OutputPath .\out\i8_volume-0.nii.gz

  One machine-readable record is appended to benchmark_results.jsonl by default.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$Label,

  [string]$ResultsPath = ".\benchmark_results.jsonl",

  [ValidateRange(100, 5000)]
  [int]$SampleMs = 500,

  [Parameter(Mandatory=$true, Position=0, ValueFromRemainingArguments=$true)]
  [string[]]$Command
)

$ErrorActionPreference = "Stop"

function Quote-ProcessArgument([string]$Value) {
  if ($Value -notmatch '[\s"]') { return $Value }
  return '"' + ($Value -replace '"', '\"') + '"'
}

function Get-ProcessTreeMemory([int]$RootPid) {
  try {
    $all = Get-CimInstance Win32_Process -ErrorAction Stop |
      Select-Object ProcessId, ParentProcessId, WorkingSetSize, PrivatePageCount
  } catch {
    # Some locked-down reading workstations deny Win32_Process queries. This
    # package runs cases serially, so fall back to the wrapper plus any active
    # onnx-totalseg process without requiring administrator privileges.
    $members = @()
    try { $members += Get-Process -Id $RootPid -ErrorAction Stop } catch {}
    try { $members += Get-Process -Name "onnx-totalseg" -ErrorAction Stop } catch {}
    $workingSet = 0L
    $private = 0L
    foreach ($member in @($members | Sort-Object Id -Unique)) {
      $workingSet += [int64]$member.WorkingSet64
      $private += [int64]$member.PrivateMemorySize64
    }
    return [pscustomobject]@{ WorkingSetBytes = $workingSet; PrivateBytes = $private }
  }

  $ids = New-Object 'System.Collections.Generic.HashSet[int]'
  [void]$ids.Add($RootPid)
  $changed = $true
  while ($changed) {
    $changed = $false
    foreach ($proc in $all) {
      $processId = [int]$proc.ProcessId
      if ($ids.Contains([int]$proc.ParentProcessId) -and -not $ids.Contains($processId)) {
        [void]$ids.Add($processId)
        $changed = $true
      }
    }
  }

  $workingSet = 0L
  $private = 0L
  foreach ($proc in $all) {
    if ($ids.Contains([int]$proc.ProcessId)) {
      if ($null -ne $proc.WorkingSetSize) { $workingSet += [int64]$proc.WorkingSetSize }
      if ($null -ne $proc.PrivatePageCount) { $private += [int64]$proc.PrivatePageCount }
    }
  }
  return [pscustomobject]@{ WorkingSetBytes = $workingSet; PrivateBytes = $private }
}

if (!$Command -or $Command.Count -eq 0) { throw "no command supplied" }

$program = $Command[0]
$childArgs = @()
if ($Command.Count -gt 1) { $childArgs = @($Command[1..($Command.Count - 1)]) }
if ([IO.Path]::GetExtension($program) -ieq ".ps1") {
  $childArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $program) + $childArgs
  $program = "powershell.exe"
}
$argumentLine = ($childArgs | ForEach-Object { Quote-ProcessArgument $_ }) -join " "

$startedAt = (Get-Date).ToUniversalTime()
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $program
$startInfo.Arguments = $argumentLine
$startInfo.UseShellExecute = $false
$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo
if (!$process.Start()) { throw "failed to start benchmark command: $program" }
$peakWorkingSet = 0L
$peakPrivate = 0L

while (!$process.HasExited) {
  $sample = Get-ProcessTreeMemory $process.Id
  if ($sample.WorkingSetBytes -gt $peakWorkingSet) { $peakWorkingSet = $sample.WorkingSetBytes }
  if ($sample.PrivateBytes -gt $peakPrivate) { $peakPrivate = $sample.PrivateBytes }
  Start-Sleep -Milliseconds $SampleMs
  $process.Refresh()
}
$stopwatch.Stop()

$record = [ordered]@{
  label = $Label
  timestamp_utc = $startedAt.ToString("o")
  command = @($Command)
  wall_s = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
  peak_working_set_mb = [math]::Round($peakWorkingSet / 1MB, 1)
  peak_private_mb = [math]::Round($peakPrivate / 1MB, 1)
  peak_vram_mb = 0
  exit_code = $process.ExitCode
}

$parent = Split-Path -Parent $ResultsPath
if ($parent -and !(Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$record | ConvertTo-Json -Compress | Add-Content -Encoding UTF8 $ResultsPath

Write-Host ("label={0} wall_s={1:N3} peak_working_set_mb={2:N1} peak_private_mb={3:N1} exit={4}" -f `
  $record.label, $record.wall_s, $record.peak_working_set_mb, $record.peak_private_mb, $record.exit_code)
exit $process.ExitCode
