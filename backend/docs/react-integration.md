# React integration guide

For the frontend developer. Everything here runs in the browser; **no secrets** are needed client-side
(Auth0 domain, client ID and audience are public identifiers).

## 1. Auth0 login and access tokens

```bash
npm install @auth0/auth0-react
```

```jsx
// main.jsx
import { Auth0Provider } from "@auth0/auth0-react";

<Auth0Provider
  domain={import.meta.env.VITE_AUTH0_DOMAIN}          // your-tenant.us.auth0.com
  clientId={import.meta.env.VITE_AUTH0_CLIENT_ID}     // SPA application client ID
  authorizationParams={{
    redirect_uri: window.location.origin,
    audience: import.meta.env.VITE_AUTH0_AUDIENCE,    // MUST equal backend AUTH0_AUDIENCE
    scope: "openid profile email",
  }}
  cacheLocation="memory"
  useRefreshTokens
>
  <App />
</Auth0Provider>
```

- Login with Google or email/password: `loginWithRedirect()`. Sign-up screen: `loginWithRedirect({ authorizationParams: { screen_hint: "signup" } })`.
- Logout: `logout({ logoutParams: { returnTo: window.location.origin } })`.
- Without `audience`, Auth0 issues a token the backend rejects (401 "invalid audience").

## 2. API client

```js
// api.js
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export function makeApi(getAccessTokenSilently) {
  async function request(path, options = {}) {
    const token = await getAccessTokenSilently();
    const res = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: { Authorization: `Bearer ${token}`, ...(options.headers ?? {}) },
    });
    if (res.status === 204) return null;
    const body = await res.json().catch(() => null);
    if (!res.ok) {
      const err = new Error(body?.error?.message ?? `HTTP ${res.status}`);
      err.code = body?.error?.code;           // e.g. "model_unavailable", "rate_limited"
      err.status = res.status;
      err.retryAfter = Number(res.headers.get("Retry-After")) || null;
      throw err;
    }
    return body;
  }
  return {
    me: () => request("/api/users/me"),
    soundTypes: () => fetch(`${API_URL}/api/sound-types`).then((r) => r.json()),
    getPreferences: () => request("/api/preferences"),
    putPreferences: (preferences) =>
      request("/api/preferences", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preferences }),
      }),
    history: (params = {}) => request(`/api/alert-history?${new URLSearchParams(params)}`),
    clearHistory: () => request("/api/alert-history", { method: "DELETE" }),
    detect: (wavBlob) => {
      const form = new FormData();
      form.append("audio", wavBlob, "chunk.wav");    // do NOT set Content-Type; the browser adds the boundary
      return request("/api/detect-sound", { method: "POST", body: form });
    },
  };
}
```

Call `api.me()` once after login. It creates the user's profile row.

## 3. Capturing audio: 16 kHz mono WAV chunks

`MediaRecorder` produces WebM/Opus (Chrome) or MP4/AAC (Safari), which need ffmpeg on the server.
Capture raw PCM with the Web Audio API and send WAV instead. That is smaller, needs no server
dependencies, and needs no resampling.

```js
// useSoundMonitor.js
import { useCallback, useRef, useState } from "react";

const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 2;

const workletCode = `
class Capture extends AudioWorkletProcessor {
  process(inputs) { if (inputs[0][0]) this.port.postMessage(inputs[0][0].slice(0)); return true; }
}
registerProcessor("capture", Capture);`;

function encodeWav(float32, sampleRate) {
  const buf = new ArrayBuffer(44 + float32.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + float32.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data");
  v.setUint32(40, float32.length * 2, true);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

export function useSoundMonitor(api, onDetection) {
  const [active, setActive] = useState(false);
  const [error, setError] = useState(null);
  const ctxRef = useRef(null);
  const streamRef = useRef(null);
  const busyRef = useRef(false);

  const start = useCallback(async () => {
    setError(null);
    const stream = await navigator.mediaDevices.getUserMedia({
      // Turn OFF voice processing: it suppresses exactly the sounds we want to detect.
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    const url = URL.createObjectURL(new Blob([workletCode], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(url);
    const node = new AudioWorkletNode(ctx, "capture");
    ctx.createMediaStreamSource(stream).connect(node);

    const target = ctx.sampleRate * CHUNK_SECONDS;   // ctx.sampleRate is 16000 if the browser honoured it
    let parts = [], length = 0;
    node.port.onmessage = async ({ data }) => {
      parts.push(data); length += data.length;
      if (length < target) return;
      const chunk = new Float32Array(length);
      let off = 0; for (const p of parts) { chunk.set(p, off); off += p.length; }
      parts = []; length = 0;
      if (busyRef.current) return;                    // drop a chunk rather than queue up latency
      busyRef.current = true;
      try {
        const result = await api.detect(encodeWav(chunk, ctx.sampleRate));
        if (result.detected && result.is_new_event) onDetection(result);
      } catch (e) {
        setError(e);                                  // network drop / 503 while model loads: keep listening
      } finally {
        busyRef.current = false;
      }
    };
    ctxRef.current = ctx; streamRef.current = stream; setActive(true);
  }, [api, onDetection]);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close();
    setActive(false);
  }, []);

  return { active, error, start, stop };
}
```

## 4. Show the alert with the user's own colour

The backend returns only the sound type. Colours come from the user's preferences.

```jsx
const prefs = await api.getPreferences();                     // load once, refresh after PUT
const byType = Object.fromEntries(prefs.preferences.map((p) => [p.sound_type, p]));

function onDetection(result) {
  const pref = byType[result.sound_type];
  if (!pref?.enabled) return;
  showFlash({ color: pref.color, flashesPerSecond: pref.flash_speed, label: result.sound_type });
}
```

- `flash_speed` is in flashes per second and is limited to 1–3 (WCAG 2.3.1). Do not flash faster.
- Respect `prefers-reduced-motion`: offer a solid colour plus a text banner instead of flashing.
- Always show **text** (for example "Fire alarm detected") as well as the colour. Colour alone fails colour-blind users.
- On `is_new_event: false`, keep the current alert visible. Do not restart it.

## 5. Response reference: `POST /api/detect-sound`

```json
{
  "detected": true,
  "sound_type": "fire_alarm",
  "confidence": 0.93,
  "timestamp": "2026-11-07T12:00:00Z",
  "reason": "detected",
  "is_new_event": true,
  "event_id": "1b0c…",
  "event_started_at": "2026-11-07T12:00:00Z",
  "candidates": [{"label": "fire_alarm", "confidence": 0.93}],
  "model_labels": [{"label": "Fire alarm", "confidence": 0.93}],
  "inference": {"provider": "local_yamnet", "model": "…", "preprocess_ms": 1.0, "inference_ms": 15.2, "total_ms": 16.5}
}
```

| `reason` | `detected` | `sound_type` |
|---|---|---|
| `detected` | true | category id |
| `below_threshold` | false | `"unknown"` |
| `silence` | false | `null` |

Errors always look like `{"error": {"code": "...", "message": "..."}}`. These matter for monitoring:
- `503 model_unavailable`: the model is still loading at startup. Retry. Show "Starting up…".
- `429 rate_limited`: wait `Retry-After` seconds.
- `422 invalid_audio`: the chunk was too short, corrupted or in an unsupported format.

## 6. Limitations to tell users

A web page cannot guarantee continuous listening. Browsers throttle or suspend background tabs.
Mobile operating systems stop the microphone when the screen locks or the app is backgrounded. Show
a clear "Monitoring active / paused" indicator. Use the Screen Wake Lock API
(`navigator.wakeLock.request("screen")`) while monitoring. **Say plainly that this app does not
replace a certified alerting device** (strobe smoke alarms, bed shakers).
