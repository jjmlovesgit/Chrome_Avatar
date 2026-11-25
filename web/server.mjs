import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..'); // litert_poc/
const port = Number(process.env.PORT || 8765);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.wav': 'audio/wav',
  '.tflite': 'application/octet-stream',
  '.bin': 'application/octet-stream',
};

const DEFAULT_VOICES = {
  puck: 'Andrew (Natural HD)',
  gretchen: 'Jenny(Natural)',
};

// ============================================================================
// Persistent Warm Windows SAPI 5 COM Worker (~50-75 ms per utterance)
// ============================================================================
class SapiWorker {
  constructor() {
    this.nextId = 1;
    this.pending = new Map();
    this.queue = Promise.resolve();
    this.ready = false;
    this.readyPromise = new Promise((resolve) => { this._resolveReady = resolve; });
    this._start();
  }

  _start() {
    const scriptPath = path.join(__dirname, 'sapi_worker.ps1');
    this.proc = spawn('powershell.exe', [
      '-NoLogo', '-NoProfile', '-NonInteractive',
      '-ExecutionPolicy', 'Bypass',
      '-File', scriptPath,
    ], { stdio: ['pipe', 'pipe', 'pipe'] });

    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on('line', (line) => {
      const s = line.trim();
      if (!s) return;
      try {
        const msg = JSON.parse(s);
        if (msg.ready) {
          this.ready = true;
          console.log(`SAPI worker ready (${msg.voiceCount} voices installed)`);
          this._resolveReady();
          return;
        }
        const entry = this.pending.get(msg.id);
        if (entry) {
          this.pending.delete(msg.id);
          if (msg.ok) entry.resolve(msg);
          else entry.reject(new Error(msg.error || 'SAPI synthesis failed'));
        }
      } catch (e) {
        console.warn('[SAPI stdout]', s);
      }
    });

    this.proc.stderr.on('data', (chunk) => {
      const txt = chunk.toString().trim();
      if (txt) console.warn('[SAPI stderr]', txt);
    });

    this.proc.on('exit', (code) => {
      console.warn(`SAPI worker exited (code=${code}); restarting on next request.`);
      for (const [, entry] of this.pending) {
        entry.reject(new Error('SAPI worker exited unexpectedly'));
      }
      this.pending.clear();
      this.proc = null;
      this.ready = false;
    });
  }

  async synthesize(text, voiceQuery) {
    const run = async () => {
      if (!this.proc) {
        this.readyPromise = new Promise((resolve) => { this._resolveReady = resolve; });
        this._start();
      }
      await this.readyPromise;
      const id = this.nextId++;
      const outPath = path.join(os.tmpdir(), `litert_sapi_${process.pid}_${id}.wav`);
      const res = await new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        const payload = JSON.stringify({ id, voice: voiceQuery, text, out: outPath }) + '\n';
        this.proc.stdin.write(payload, 'utf8');
      });
      const wavBuf = await fs.promises.readFile(outPath);
      fs.promises.unlink(outPath).catch(() => {});
      return { wavBuf, ms: res.ms, voice: res.voice };
    };
    // Serialize COM requests cleanly
    const p = this.queue.then(run, run);
    this.queue = p.catch(() => {});
    return p;
  }
}

const sapi = new SapiWorker();

// ============================================================================
// Persistent Warm Windows GPU VRAM Worker (~16 ms per snapshot!)
// ============================================================================
class VramWorker {
  constructor() {
    this.nextId = 1;
    this.pending = new Map();
    this.ready = false;
    this.readyPromise = new Promise((resolve) => { this._resolveReady = resolve; });
    this.lastSnap = null;
    this.lastSnapAt = 0;
    this.inFlight = null;
    this._start();
  }

  _start() {
    const scriptPath = path.join(__dirname, 'vram_worker.ps1');
    this.proc = spawn('powershell.exe', [
      '-NoLogo', '-NoProfile', '-NonInteractive',
      '-ExecutionPolicy', 'Bypass',
      '-File', scriptPath,
    ], { stdio: ['pipe', 'pipe', 'pipe'] });

    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on('line', (line) => {
      const s = line.trim();
      if (!s) return;
      try {
        const msg = JSON.parse(s);
        if (msg.ready) {
          this.ready = true;
          console.log(`VRAM worker ready (${msg.gpuName || 'GPU'}, ${msg.gpuTotalMb || '?'} MB total)`);
          this._resolveReady();
          return;
        }
        const entry = this.pending.get(msg.id);
        if (entry) {
          this.pending.delete(msg.id);
          if (msg.ok) entry.resolve(msg);
          else entry.reject(new Error(msg.error || 'VRAM poll failed'));
        }
      } catch {}
    });

    this.proc.on('exit', () => {
      for (const [, entry] of this.pending) entry.reject(new Error('VRAM worker exited'));
      this.pending.clear();
      this.proc = null;
      this.ready = false;
    });
  }

  async snapshot(force = false) {
    const now = Date.now();
    if (!force && this.lastSnap && (now - this.lastSnapAt) < 800) {
      return this.lastSnap;
    }
    if (this.inFlight) return this.inFlight;
    this.inFlight = (async () => {
      if (!this.proc) {
        this.readyPromise = new Promise((resolve) => { this._resolveReady = resolve; });
        this._start();
      }
      await this.readyPromise;
      const id = this.nextId++;
      const snap = await new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        this.proc.stdin.write(JSON.stringify({ id }) + '\n', 'utf8');
      });
      this.lastSnap = { ...snap, updatedAt: Date.now() };
      this.lastSnapAt = Date.now();
      return this.lastSnap;
    })();
    try {
      return await this.inFlight;
    } finally {
      this.inFlight = null;
    }
  }
}

const vram = new VramWorker();

function cleanSpokenText(raw) {
  return String(raw || '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[*_`#~<>]/g, '')
    .replace(/\[[^\]]*\]\([^)]*\)/g, (m) => m.replace(/\[([^\]]*)\]\([^)]*\)/, '$1'))
    .replace(/^(Puck|Gretchen)\s*:\s*/i, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function readBodyJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}'));
      } catch (e) {
        reject(e);
      }
    });
    req.on('error', reject);
  });
}

const COMMON_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
  'Cache-Control': 'no-cache',
};

const server = http.createServer(async (req, res) => {
  const urlPath = (req.url || '/').split('?')[0];

  // GET /api/vram -> Live Windows per-process GPU VRAM breakdown (~16 ms warm!)
  if (req.method === 'GET' && urlPath === '/api/vram') {
    try {
      const force = (req.url || '').includes('force=1');
      const snap = await vram.snapshot(force);
      res.writeHead(200, { ...COMMON_HEADERS, 'Content-Type': 'application/json' });
      res.end(JSON.stringify(snap));
    } catch (e) {
      res.writeHead(500, { ...COMMON_HEADERS, 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: String(e?.message || e) }));
    }
    return;
  }

  // POST /api/tts -> Windows SAPI 16kHz 16-bit mono WAV
  if (req.method === 'POST' && urlPath === '/api/tts') {
    try {
      const body = await readBodyJson(req);
      const avatar = (body.avatar || 'puck').toLowerCase();
      const voiceQuery = body.voice || DEFAULT_VOICES[avatar] || DEFAULT_VOICES.puck;
      const text = cleanSpokenText(body.text);
      if (!text) {
        res.writeHead(400, { ...COMMON_HEADERS, 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Empty text' }));
        return;
      }
      const { wavBuf, ms, voice } = await sapi.synthesize(text, voiceQuery);
      res.writeHead(200, {
        ...COMMON_HEADERS,
        'Content-Type': 'audio/wav',
        'Content-Length': String(wavBuf.length),
        'X-TTS-Ms': String(ms),
        'X-TTS-Voice': encodeURIComponent(voice || voiceQuery),
      });
      res.end(wavBuf);
    } catch (e) {
      res.writeHead(500, { ...COMMON_HEADERS, 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: String(e?.message || e) }));
    }
    return;
  }

  // POST /api/llm -> Optional local fallback proxy to LM Studio / llama-server (port 1234)
  if (req.method === 'POST' && urlPath === '/api/llm') {
    try {
      const body = await readBodyJson(req);
      const endpoint = body.endpoint || 'http://127.0.0.1:1234/v1/chat/completions';
      const upstream = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: body.model || 'dolphin3.0-llama3.1-8b-abliterated',
          messages: body.messages || [],
          temperature: body.temperature ?? 0.7,
          max_tokens: body.max_tokens ?? 140,
          stream: false,
        }),
      });
      const txt = await upstream.text();
      res.writeHead(upstream.status, { ...COMMON_HEADERS, 'Content-Type': 'application/json' });
      res.end(txt);
    } catch (e) {
      res.writeHead(502, { ...COMMON_HEADERS, 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: `Local LLM unreachable (${e?.message || e})` }));
    }
    return;
  }

  // POST /api/telemetry -> Optional autotest log sink for browser verification
  if (req.method === 'POST' && urlPath === '/api/telemetry') {
    try {
      const body = await readBodyJson(req);
      console.log('=== CLIENT TELEMETRY ===\n' + JSON.stringify(body, null, 2) + '\n=== END TELEMETRY ===');
      res.writeHead(200, { ...COMMON_HEADERS, 'Content-Type': 'application/json' });
      res.end('{"ok":true}');
    } catch (e) {
      res.writeHead(400, { ...COMMON_HEADERS, 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: String(e?.message || e) }));
    }
    return;
  }

  // Static files under litert_poc/
  const relPath = urlPath === '/' ? '/web/demo.html' : urlPath;
  const fullPath = path.resolve(rootDir, '.' + decodeURIComponent(relPath));
  if (!fullPath.startsWith(rootDir) || !fs.existsSync(fullPath) || fs.statSync(fullPath).isDirectory()) {
    res.writeHead(404, { ...COMMON_HEADERS, 'Content-Type': 'text/plain' });
    res.end('Not found: ' + relPath);
    return;
  }
  const ext = path.extname(fullPath).toLowerCase();
  res.writeHead(200, {
    ...COMMON_HEADERS,
    'Content-Type': MIME[ext] || 'application/octet-stream',
  });
  fs.createReadStream(fullPath).pipe(res);
});

server.listen(port, '127.0.0.1', () => {
  console.log(`LiteRT.js Studio server running at http://127.0.0.1:${port}/`);
});
