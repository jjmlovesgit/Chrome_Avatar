@echo off
setlocal
set "ROOT=%~dp0"
set "WEB=%ROOT%web"
set "URL=http://127.0.0.1:8765/"

REM First-run dependency install (only shows briefly if node_modules is missing)
if not exist "%WEB%\node_modules" (
  echo First run detected - installing LiteRT.js and WebLLM packages...
  pushd "%WEB%"
  call npm install
  popd
)

REM Launch hidden node server.mjs (unless port 8765 is already listening) and open Google Chrome
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command ^
  "$web = '%WEB%'; $url = '%URL%'; $pidFile = Join-Path $web '.server.pid';" ^
  "$up = $false; try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', 8765); $c.Close(); $up = $true } catch {}" ^
  "if (-not $up) {" ^
  "  $p = Start-Process -FilePath 'node' -ArgumentList 'server.mjs' -WorkingDirectory $web -WindowStyle Hidden -PassThru;" ^
  "  if ($p) { Set-Content -Path $pidFile -Value $p.Id -Encoding ASCII }" ^
  "  for ($i = 0; $i -lt 40; $i++) {" ^
  "    Start-Sleep -Milliseconds 250;" ^
  "    try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', 8765); $c.Close(); break } catch {}" ^
  "  }" ^
  "}" ^
  "$chrome = @(\"$env:ProgramFiles\Google\Chrome\Application\chrome.exe\", \"${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe\", \"$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe\") | Where-Object { Test-Path $_ } | Select-Object -First 1;" ^
  "if ($chrome) { Start-Process -FilePath $chrome -ArgumentList $url } else { Start-Process $url }"

exit /b 0
