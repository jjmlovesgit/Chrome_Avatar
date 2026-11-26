@echo off
setlocal
set "ROOT=%~dp0"
set "WEB=%ROOT%web"
set "URL=http://127.0.0.1:8765/"
set "REL_BASE=https://github.com/jjmlovesgit/Chrome_Avatar/releases/download/v1.0.0"

REM 1. Auto-download any missing FP16-weight .tflite models from GitHub Release v1.0.0 (first run only)
for %%F in (wav2vec2_2s_fp16w.tflite fm_step_fp16w.tflite student_renderer_512_fp16w.tflite student_renderer_256_fp16w.tflite renderer_decode_fp16w.tflite) do (
  if not exist "%ROOT%%%F" (
    echo Downloading %%F from GitHub Release v1.0.0...
    curl.exe -L --fail --progress-bar -o "%ROOT%%%F.part" "%REL_BASE%/%%F"
    if errorlevel 1 (
      del /q "%ROOT%%%F.part" 2>nul
      echo.
      echo ERROR: Failed to download %%F. Please check your internet connection or download manually from:
      echo   %REL_BASE%/%%F
      pause
      exit /b 1
    )
    move /y "%ROOT%%%F.part" "%ROOT%%%F" >nul
  )
)

REM 2. First-run npm install (only runs if web\node_modules is missing)
if not exist "%WEB%\node_modules" (
  echo First run detected - installing LiteRT.js and WebLLM packages...
  pushd "%WEB%"
  call npm install
  popd
)

REM 3. Launch hidden node server.mjs (unless port 8765 is already listening) and open Google Chrome
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
