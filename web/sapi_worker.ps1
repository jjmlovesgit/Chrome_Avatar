# Persistent warm Windows SAPI 5 COM TTS worker for server.mjs (zero Python, ~50ms warm synthesis).
# Protocol (NDJSON over stdin/stdout):
#   stdin:  {"id":1,"voice":"Andrew (Natural HD)","text":"Hello world","out":"C:\\...\\tts.wav"}
#   stdout: {"id":1,"ok":true,"ms":54.2,"voice":"Microsoft Andrew (Natural HD) ..."}

$ErrorActionPreference = 'Stop'
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$speaker = New-Object -ComObject SAPI.SpVoice
$allVoices = @()
$voiceMeta = @()
$vList = $speaker.GetVoices()
for ($i = 0; $i -lt $vList.Count; $i++) {
    $v = $vList.Item($i)
    $desc = $v.GetDescription()
    $gender = ''
    $lang = ''
    try { $gender = $v.GetAttribute('Gender') } catch {}
    try { $lang = $v.GetAttribute('Language') } catch {}
    $allVoices += [PSCustomObject]@{ Desc = $desc; Obj = $v }
    $voiceMeta += [PSCustomObject]@{ name = $desc; gender = $gender; lang = $lang }
}

function Select-SapiVoice([string]$query) {
    if ([string]::IsNullOrWhiteSpace($query)) { return $speaker.Voice }
    $qLower = $query.ToLowerInvariant()
    foreach ($item in $allVoices) {
        if ($item.Desc.ToLowerInvariant() -eq $qLower) {
            $speaker.Voice = $item.Obj
            return $item.Obj
        }
    }
    foreach ($item in $allVoices) {
        if ($item.Desc.ToLowerInvariant().Contains($qLower)) {
            $speaker.Voice = $item.Obj
            return $item.Obj
        }
    }
    return $speaker.Voice
}

# Warm up both default voices into a scratch file so first live turn is fast
$warmPath = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "litert_sapi_warm.wav")
foreach ($vName in @("Andrew (Natural HD)", "Jenny(Natural)")) {
    try {
        Select-SapiVoice $vName | Out-Null
        $st = New-Object -ComObject SAPI.SpFileStream
        $fmt = New-Object -ComObject SAPI.SpAudioFormat
        $fmt.Type = 18  # SAFT16kHz16BitMono
        $st.Format = $fmt
        $st.Open($warmPath, 3) # SSFMCreateForWrite
        $speaker.AudioOutputStream = $st
        $speaker.Speak("Hi") | Out-Null
        $st.Close()
    } catch {}
}

$readyObj = @{ ready = $true; voiceCount = $allVoices.Count; voices = $voiceMeta } | ConvertTo-Json -Compress -Depth 4
[Console]::Out.WriteLine($readyObj)
[Console]::Out.Flush()

while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) { break }
    $line = $line.Trim()
    if ($line.Length -eq 0) { continue }
    $id = 0
    try {
        $req = $line | ConvertFrom-Json
        $id = $req.id
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $chosen = Select-SapiVoice $req.voice
        $st = New-Object -ComObject SAPI.SpFileStream
        $fmt = New-Object -ComObject SAPI.SpAudioFormat
        $fmt.Type = 18  # SAFT16kHz16BitMono
        $st.Format = $fmt
        $st.Open($req.out, 3)
        $speaker.AudioOutputStream = $st
        $speaker.Speak([string]$req.text) | Out-Null
        $st.Close()
        $sw.Stop()
        $resp = @{
            id    = $id
            ok    = $true
            ms    = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
            voice = $chosen.GetDescription()
        } | ConvertTo-Json -Compress
        [Console]::Out.WriteLine($resp)
        [Console]::Out.Flush()
    } catch {
        $resp = @{
            id    = $id
            ok    = $false
            error = $_.Exception.Message
        } | ConvertTo-Json -Compress
        [Console]::Out.WriteLine($resp)
        [Console]::Out.Flush()
    }
}
