$ErrorActionPreference = 'SilentlyContinue'
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Detect GPU name & total VRAM once at startup
$gpuName = ''
$gpuTotalMb = 0
try {
  $nv = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
  if ($nv -and $nv -match '^([^,]+),\s*(\d+)') {
    $gpuName = $Matches[1].Trim()
    $gpuTotalMb = [int]$Matches[2]
  }
} catch {}

# Warm CIM GPU performance counter provider once so polls take ~16 ms
$null = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory
$pidCache = @{}

function Get-VramSnapshot {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $rows = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
    Where-Object { $_.DedicatedUsage -gt 2097152 }

  $byPid = @{}
  foreach ($r in $rows) {
    if ($r.Name -match 'pid_(\d+)_') {
      $p = [int]$Matches[1]
      $byPid[$p] = [double]$byPid[$p] + [double]$r.DedicatedUsage
    }
  }

  # Refresh metadata only for newly seen PIDs
  foreach ($p in $byPid.Keys) {
    if (-not $pidCache.ContainsKey($p)) {
      $pr = Get-Process -Id $p -ErrorAction SilentlyContinue
      $pn = if ($pr) { $pr.ProcessName } else { '' }
      $role = ''
      if ($pn -match '^(chrome|msedge|Antigravity|electron)$') {
        $cmd = (Get-CimInstance Win32_Process -Filter ("ProcessId=" + $p) -ErrorAction SilentlyContinue).CommandLine
        if ($cmd -match 'OnDeviceModelService') { $role = 'nano' }
        elseif ($cmd -match '--type=gpu-process') { $role = 'gpu' }
        else { $role = 'other' }
      }
      $pidCache[$p] = @{ name = $pn; role = $role }
    }
  }

  # Prune dead PIDs periodically
  if ($pidCache.Count -gt 256) {
    $alive = @{}
    foreach ($p in $byPid.Keys) { if ($pidCache.ContainsKey($p)) { $alive[$p] = $pidCache[$p] } }
    $pidCache = $alive
  }

  $chromeGpu = 0.0
  $chromeNano = 0.0
  $chromeOther = 0.0
  $antigravity = 0.0
  $edge = 0.0

  foreach ($kv in $byPid.GetEnumerator()) {
    $meta = $pidCache[$kv.Key]
    if (-not $meta) { continue }
    $mb = $kv.Value / 1048576.0
    switch ($meta.name) {
      'chrome' {
        if ($meta.role -eq 'nano') { $chromeNano += $mb }
        elseif ($meta.role -eq 'gpu') { $chromeGpu += $mb }
        else { $chromeOther += $mb }
      }
      'Antigravity' { $antigravity += $mb }
      'msedge'      { $edge += $mb }
    }
  }

  $adapterUsedMb = 0.0
  try {
    $adapters = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory
    foreach ($a in $adapters) {
      $mb = [double]$a.DedicatedUsage / 1048576.0
      if ($mb -gt $adapterUsedMb) { $adapterUsedMb = $mb }
    }
  } catch {}

  $sw.Stop()
  return @{
    ok               = $true
    gpuName          = $gpuName
    gpuTotalMb       = $gpuTotalMb
    gpuUsedMb        = [math]::Round($adapterUsedMb, 1)
    chromeGpuMb      = [math]::Round($chromeGpu, 1)
    chromeNanoMb     = [math]::Round($chromeNano, 1)
    chromeOtherMb    = [math]::Round($chromeOther, 1)
    chromeTotalMb    = [math]::Round($chromeGpu + $chromeNano + $chromeOther, 1)
    antigravityGpuMb = [math]::Round($antigravity, 1)
    edgeGpuMb        = [math]::Round($edge, 1)
    ms               = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
  }
}

Write-Output (@{ ready = $true; gpuName = $gpuName; gpuTotalMb = $gpuTotalMb } | ConvertTo-Json -Compress)

while ($true) {
  $line = [Console]::In.ReadLine()
  if ($null -eq $line) { break }
  $line = $line.Trim()
  if (-not $line) { continue }
  try {
    $req = $line | ConvertFrom-Json
    $snap = Get-VramSnapshot
    $snap['id'] = $req.id
    Write-Output ($snap | ConvertTo-Json -Compress)
  } catch {
    Write-Output (@{ id = 0; ok = $false; error = $_.Exception.Message } | ConvertTo-Json -Compress)
  }
}
