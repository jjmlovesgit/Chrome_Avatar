@echo off
setlocal
set "WEB=%~dp0web"
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command ^
  "$pidFile = Join-Path '%WEB%' '.server.pid';" ^
  "if (Test-Path $pidFile) {" ^
  "  $id = [int](Get-Content $pidFile -Raw).Trim();" ^
  "  taskkill /PID $id /T /F 2>$null | Out-Null;" ^
  "  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue;" ^
  "}" ^
  "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" | Where-Object { $_.CommandLine -match 'server\.mjs' } | ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null | Out-Null }"
exit /b 0
