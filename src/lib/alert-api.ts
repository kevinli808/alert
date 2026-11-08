// Browser client for the FastAPI backend in /backend. No secrets live here: the URL is public config.

// Empty means same-origin requests, routed to FastAPI by Vercel's /api and /health rewrites.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL || '').replace(/\/$/, '')

export type BackendHealth = {
  status: 'ok' | 'degraded'
  model: { provider: string; status: string }
  auth_configured: boolean
  database_configured: boolean
  anonymous_detection?: boolean
}

export type DetectionResult = {
  detected: boolean
  sound_type: string | null
  confidence: number
  timestamp: string
  reason: 'detected' | 'below_threshold' | 'silence'
  is_new_event: boolean
  event_id: string | null
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public retryAfter?: number,
  ) {
    super(message)
  }
}

/** Returns an Auth0 access token once login is wired up; null means "not logged in". */
export type TokenGetter = () => Promise<string | null>

async function request<T>(
  path: string,
  init: RequestInit = {},
  getToken?: TokenGetter,
): Promise<T> {
  const token = getToken ? await getToken().catch(() => null) : null
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...init.headers },
  })
  const body = res.status === 204 ? null : await res.json().catch(() => null)
  if (!res.ok) {
    throw new ApiError(
      body?.error?.message ?? `HTTP ${res.status}`,
      res.status,
      body?.error?.code,
      Number(res.headers.get('Retry-After')) || undefined,
    )
  }
  return body as T
}

export const getHealth = () => request<BackendHealth>('/health')

export function detectSound(wav: Blob, getToken?: TokenGetter, signal?: AbortSignal) {
  const form = new FormData()
  form.append('audio', wav, 'chunk.wav') // let the browser set the multipart boundary
  return request<DetectionResult>(
    '/api/detect-sound',
    { method: 'POST', body: form, signal },
    getToken,
  )
}

// The backend's database records users only (Databricks users table). Calling this after login
// creates the user on first login and updates last_login_at afterwards. Settings stay in the browser.
export type UserRecord = {
  id: string
  email: string | null
  username: string | null
  created_at: string
  last_login_at: string
}

export const recordLogin = (getToken: TokenGetter) => request<UserRecord>('/api/users/me', {}, getToken)

// --------------------------------------------------------------- login (Auth0) ---------
// Public identifiers only (no secrets). Set them in .env.local; audience must equal the backend's
// AUTH0_AUDIENCE. While any is missing, the app falls back to its local preview account.
export const AUTH0_CONFIG = {
  domain: process.env.NEXT_PUBLIC_AUTH0_DOMAIN || '',
  clientId: process.env.NEXT_PUBLIC_AUTH0_CLIENT_ID || '',
  audience: process.env.NEXT_PUBLIC_AUTH0_AUDIENCE || '',
}
export const AUTH0_ENABLED = Boolean(
  AUTH0_CONFIG.domain && AUTH0_CONFIG.clientId && AUTH0_CONFIG.audience,
)

let tokenGetter: TokenGetter = async () => null
/** Called by the Auth0 bridge in AlertApp once the user's login state is known. */
export const setAccessTokenGetter = (getter: TokenGetter) => {
  tokenGetter = getter
}
/** The logged-in user's access token, or null (not logged in / no Auth0). */
export const getAccessToken: TokenGetter = () => tokenGetter().catch(() => null)

/**
 * Backend sound category -> frontend rule id. Keep in sync with
 * backend/app/services/sound_categories.py (GET /api/sound-types lists them).
 */
export const BACKEND_SOUND_TO_RULE: Record<string, string> = {
  fire_alarm: 'fire-alarm',
  smoke_detector: 'smoke-alarm',
  emergency_siren: 'siren',
  glass_breaking: 'glass-breaking',
  baby_crying: 'baby-crying',
  car_alarm: 'car-alarm',
  alarm: 'alarm',
  doorbell: 'doorbell',
  door_knock: 'door-knock',
  phone_ringing: 'phone-ringing',
  alarm_clock: 'alarm-clock',
  dog_barking: 'dog-barking',
}
export const LIVE_RULE_IDS = new Set(Object.values(BACKEND_SOUND_TO_RULE))

// --------------------------------------------------------------- audio capture ---------

// Collects mono samples inside the audio thread and posts one Float32Array per full chunk.
const CAPTURE_WORKLET = `
class ChunkCapture extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.size = options.processorOptions.size
    this.buf = new Float32Array(this.size)
    this.len = 0
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (ch) {
      let i = 0
      while (i < ch.length) {
        const n = Math.min(ch.length - i, this.size - this.len)
        this.buf.set(ch.subarray(i, i + n), this.len)
        this.len += n
        i += n
        if (this.len === this.size) {
          this.port.postMessage(this.buf, [this.buf.buffer])
          this.buf = new Float32Array(this.size)
          this.len = 0
        }
      }
    }
    return true
  }
}
registerProcessor('chunk-capture', ChunkCapture)
`

/** Calls onChunk with ~`seconds` of mono audio at context.sampleRate. Stops when the context closes. */
export async function startChunkCapture(
  context: AudioContext,
  source: AudioNode,
  seconds: number,
  onChunk: (samples: Float32Array) => void,
): Promise<AudioWorkletNode> {
  const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: 'application/javascript' }))
  try {
    await context.audioWorklet.addModule(url)
  } finally {
    URL.revokeObjectURL(url)
  }
  const node = new AudioWorkletNode(context, 'chunk-capture', {
    processorOptions: { size: Math.round(context.sampleRate * seconds) },
  })
  node.port.onmessage = (event: MessageEvent<Float32Array>) => onChunk(event.data)
  // Route through a muted gain so the graph keeps pulling the node without playing the mic back.
  const mute = context.createGain()
  mute.gain.value = 0
  source.connect(node)
  node.connect(mute).connect(context.destination)
  return node
}

/** 16-bit PCM mono WAV. The backend resamples to 16 kHz itself. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
  }
  write(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  write(8, 'WAVE')
  write(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  write(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}
