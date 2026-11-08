'use client'

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react'
import { Auth0Provider, useAuth0 } from '@auth0/auth0-react'
import {
  AUTH0_CONFIG,
  AUTH0_ENABLED,
  ApiError,
  BACKEND_SOUND_TO_RULE,
  LIVE_RULE_IDS,
  detectSound,
  encodeWav,
  getAccessToken,
  getHealth,
  wakeModel,
  recordLogin,
  setAccessTokenGetter,
  startChunkCapture,
  type DetectionResult,
} from '../lib/alert-api'

const CHUNK_SECONDS = 2

type ListeningState = 'paused' | 'listening' | 'unavailable'
type BackendStatus = 'checking' | 'ready' | 'loading' | 'offline' | 'login-required' | 'error'
type View = 'home' | 'configure'
type DialogName = 'help' | 'account' | 'custom-state' | null
type Priority = 0 | 1 | 2
type VisualState = {
  id: string
  name: string
  color: string
  priority: Priority
  duration: number
  system?: boolean
}
type SoundRule = {
  id: string
  label: string
  kind: 'sound' | 'phrase'
  enabled: boolean
  effects: string[]
  ttsResponse?: string
  lightColor?: string
}
type AlertRequest = {
  stateId: string
  soundNames: string[]
}
type ActiveAlert = AlertRequest & {
  startedAt: number
  expiresAt: number
}
type Detection = {
  id: string
  label: string
  expiresAt: number
  critical: boolean
  live: boolean
  confidence?: number
}
type StoredSettings = {
  simple: boolean
  rules: SoundRule[]
  customStates: VisualState[]
  stateOrder: string[]
  selectedRuleId: string
  priorityCollapsed: boolean
  introComplete: boolean
}

const STORAGE_KEY = 'alert-settings-v2'
const ACCOUNT_KEY = 'alert-account-v1'

const SYSTEM_STATES: VisualState[] = [
  { id: 'default', name: 'Default', color: '#2dd4bf', priority: 0, duration: 0, system: true },
  {
    id: 'warning',
    name: 'Yellow — Warning',
    color: '#d4a72c',
    priority: 1,
    duration: 3000,
    system: true,
  },
  {
    id: 'urgent',
    name: 'Red — Urgent',
    color: '#e05252',
    priority: 2,
    duration: 5000,
    system: true,
  },
]
const SYSTEM_STATE_ORDER = ['urgent', 'warning', 'default']

const DEFAULT_RULES: SoundRule[] = [
  { id: 'dog-barking', label: 'Dog barking', kind: 'sound', enabled: true, effects: ['warning'] },
  { id: 'human-speech', label: 'Human speech', kind: 'sound', enabled: true, effects: ['default'] },
  { id: 'footsteps', label: 'Footsteps', kind: 'sound', enabled: true, effects: ['default'] },
  {
    id: 'doorbell',
    label: 'Doorbell',
    kind: 'sound',
    enabled: true,
    effects: ['warning', 'vibration'],
  },
  {
    id: 'smoke-alarm',
    label: 'Smoke alarm',
    kind: 'sound',
    enabled: true,
    effects: ['urgent', 'vibration'],
  },
  { id: 'siren', label: 'Siren', kind: 'sound', enabled: true, effects: ['urgent'] },
  {
    id: 'glass-breaking',
    label: 'Glass breaking',
    kind: 'sound',
    enabled: true,
    effects: ['urgent'],
  },
  {
    id: 'vehicle-passing',
    label: 'Vehicle passing',
    kind: 'sound',
    enabled: true,
    effects: ['default'],
  },
  { id: 'baby-crying', label: 'Baby crying', kind: 'sound', enabled: true, effects: ['warning'] },
  {
    id: 'phone-ringing',
    label: 'Phone ringing',
    kind: 'sound',
    enabled: true,
    effects: ['warning', 'vibration'],
  },
  {
    id: 'fire-alarm',
    label: 'Fire alarm',
    kind: 'sound',
    enabled: true,
    effects: ['urgent', 'vibration'],
  },
  {
    id: 'alarm',
    label: 'Alarm / buzzer',
    kind: 'sound',
    enabled: true,
    effects: ['urgent', 'vibration'],
  },
  { id: 'car-alarm', label: 'Car alarm', kind: 'sound', enabled: true, effects: ['warning'] },
  {
    id: 'door-knock',
    label: 'Door knock',
    kind: 'sound',
    enabled: true,
    effects: ['warning', 'vibration'],
  },
  { id: 'alarm-clock', label: 'Alarm clock', kind: 'sound', enabled: true, effects: ['warning'] },
  {
    id: 'phrase-help',
    label: 'Help',
    kind: 'phrase',
    enabled: true,
    effects: ['urgent', 'vibration'],
  },
]

const ADDITIONAL_EFFECTS = [
  { id: 'vibration', name: 'Vibration', category: 'Physical feedback' },
  { id: 'tts', name: 'Text-to-speech', category: 'Additional effects' },
]

const FUTURE_EFFECTS = [
  { id: 'smart-lights', name: 'Turn lights — chosen colour', category: 'Future effects' },
  { id: 'watch-haptics', name: 'Smartwatch haptics', category: 'Future effects' },
  { id: 'screen-flash', name: 'Screen flash', category: 'Future effects' },
  {
    id: 'persistent-notifications',
    name: 'Persistent notifications',
    category: 'Future effects',
  },
  { id: 'escalating-alerts', name: 'Escalating alerts', category: 'Future effects' },
  {
    id: 'emergency-notifications',
    name: 'Emergency notifications',
    category: 'Future effects',
  },
]
const FUTURE_EFFECT_IDS = new Set(FUTURE_EFFECTS.map((effect) => effect.id))

const API_STATUS_LABEL: Record<BackendStatus, string> = {
  checking: 'CONNECTING TO DETECTOR…',
  ready: 'LIVE DETECTION READY',
  loading: 'DETECTION MODEL STARTING…',
  offline: 'DETECTOR OFFLINE · DEMO ONLY',
  'login-required': 'LOG IN FOR LIVE DETECTION',
  error: 'DETECTOR ERROR · DEMO ONLY',
}
const API_STATUS_HINT: Record<BackendStatus, string> = {
  checking: 'connecting to the detection server',
  ready: 'connected',
  loading: 'the sound model is still loading',
  offline: 'detection server not reachable, so only simulated sounds work',
  'login-required': 'the detection server requires a login',
  error: 'the detection server reported an error',
}

const DEFAULT_SETTINGS: StoredSettings = {
  simple: true,
  rules: DEFAULT_RULES,
  customStates: [],
  stateOrder: SYSTEM_STATE_ORDER,
  selectedRuleId: DEFAULT_RULES[0].id,
  priorityCollapsed: false,
  introComplete: false,
}

function loadSettings(storageKey: string): StoredSettings {
  if (typeof window === 'undefined') return DEFAULT_SETTINGS
  try {
    let serialized = localStorage.getItem(storageKey)
    if (!serialized) {
      const legacySettings = localStorage.getItem(STORAGE_KEY)
      const guestSettings = localStorage.getItem(`${STORAGE_KEY}:guest`)
      serialized = legacySettings || guestSettings
      if (serialized) {
        localStorage.setItem(storageKey, serialized)
        if (legacySettings) localStorage.removeItem(STORAGE_KEY)
        if (guestSettings && storageKey !== `${STORAGE_KEY}:guest`)
          localStorage.removeItem(`${STORAGE_KEY}:guest`)
      }
    }
    return normalizeSettings(JSON.parse(serialized || '{}'))
  } catch {
    return DEFAULT_SETTINGS
  }
}

/** Repairs settings from any source (this browser or the backend) into a consistent shape. */
function normalizeSettings(saved: Partial<StoredSettings>): StoredSettings {
  try {
    const customStates = (saved.customStates ?? []).map((state, index) => {
      // Drop the legacy `label` key so the backend (which allowlists fields) accepts the state.
      const { label: legacyLabel, ...current } = state as VisualState & { label?: string }
      return {
        ...current,
        name:
          (typeof state.name === 'string' && state.name.trim()) ||
          (typeof legacyLabel === 'string' && legacyLabel.trim()) ||
          `Custom state ${index + 1}`,
      }
    })
    const allStateIds = [...SYSTEM_STATE_ORDER, ...customStates.map((state) => state.id)]
    const restoredOrder = [
      ...(saved.stateOrder ?? SYSTEM_STATE_ORDER).filter((id) => allStateIds.includes(id)),
      ...allStateIds.filter((id) => !(saved.stateOrder ?? SYSTEM_STATE_ORDER).includes(id)),
    ]
    const stateOrder = [
      'urgent',
      ...restoredOrder.filter((id) => id !== 'urgent' && id !== 'default'),
      'default',
    ]
    const visualStateIds = new Set(allStateIds)
    const savedRules = saved.rules?.length ? saved.rules : DEFAULT_RULES
    // Settings saved before a sound was added (e.g. new backend categories) still get that sound.
    const missingSounds = DEFAULT_RULES.filter(
      (rule) => rule.kind === 'sound' && !savedRules.some((savedRule) => savedRule.id === rule.id),
    )
    const rules = [...savedRules, ...missingSounds].map((rule) => {
      const assignedStates = rule.effects.filter((effect) => visualStateIds.has(effect))
      const assignedState =
        stateOrder.find((stateId) => assignedStates.includes(stateId)) ?? 'default'
      return {
        ...rule,
        effects: [assignedState, ...rule.effects.filter((effect) => !visualStateIds.has(effect))],
      }
    })
    return {
      simple: saved.simple ?? DEFAULT_SETTINGS.simple,
      rules,
      customStates,
      stateOrder,
      selectedRuleId:
        typeof saved.selectedRuleId === 'string' ? saved.selectedRuleId : DEFAULT_RULES[0].id,
      priorityCollapsed: saved.priorityCollapsed ?? false,
      introComplete: saved.introComplete ?? false,
    }
  } catch {
    return DEFAULT_SETTINGS
  }
}

function Icon({ name }: { name: 'help' | 'menu' | 'user' | 'close' | 'back' | 'plus' | 'search' }) {
  const paths: Record<string, ReactNode> = {
    help: (
      <>
        <circle cx="8" cy="8" r="6.25" />
        <path d="M6.55 6.05A1.55 1.55 0 0 1 8.1 4.7c.86 0 1.55.59 1.55 1.35 0 1.1-1.65 1.32-1.65 2.42" />
        <circle cx="8" cy="11.15" r=".45" fill="currentColor" stroke="none" />
      </>
    ),
    menu: <path d="M2.5 4h11M2.5 8h11M2.5 12h11" />,
    user: (
      <>
        <circle cx="8" cy="5.4" r="2.4" />
        <path d="M3.4 13.1c.45-2.18 2.15-3.35 4.6-3.35s4.15 1.17 4.6 3.35" />
      </>
    ),
    close: <path d="m3.5 3.5 9 9m0-9-9 9" />,
    back: <path d="m9.75 3.25-4.5 4.75 4.5 4.75M5.5 8h7" />,
    plus: <path d="M8 3v10M3 8h10" />,
    search: (
      <>
        <circle cx="7" cy="7" r="4.25" />
        <path d="m10.3 10.3 3.1 3.1" />
      </>
    ),
  }
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.25"
      strokeLinecap="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  )
}

function AlertLogo() {
  return (
    <svg
      className="alert-logo"
      width="24"
      height="20"
      viewBox="0 0 24 20"
      fill="none"
      aria-hidden="true"
    >
      <rect x="1" y="7" width="3" height="6" rx="1.5" fill="currentColor" />
      <rect x="6" y="4" width="3" height="12" rx="1.5" fill="currentColor" />
      <rect x="11" y="1" width="3" height="18" rx="1.5" fill="currentColor" />
      <rect x="16" y="4" width="3" height="12" rx="1.5" fill="currentColor" />
      <rect x="21" y="7" width="3" height="6" rx="1.5" fill="currentColor" />
    </svg>
  )
}

function GoogleIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 18 18" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M17.64 9.205c0-.638-.057-1.252-.164-1.841H9v3.481h4.844a4.14 4.14 0 0 1-1.797 2.715v2.258h2.909c1.703-1.568 2.684-3.878 2.684-6.613Z"
      />
      <path
        fill="#34A853"
        d="M9 18c2.43 0 4.468-.806 5.956-2.182l-2.91-2.258c-.805.54-1.834.859-3.046.859-2.344 0-4.328-1.585-5.037-3.714H.956v2.332A8.998 8.998 0 0 0 9 18Z"
      />
      <path
        fill="#FBBC05"
        d="M3.963 10.705A5.41 5.41 0 0 1 3.682 9c0-.592.102-1.168.281-1.705V4.963H.956A9.003 9.003 0 0 0 0 9c0 1.452.347 2.827.956 4.037l3.007-2.332Z"
      />
      <path
        fill="#EA4335"
        d="M9 3.58c1.321 0 2.507.454 3.44 1.345l2.582-2.581C13.464.892 11.427 0 9 0A8.998 8.998 0 0 0 .956 4.963l3.007 2.332C4.672 5.166 6.656 3.58 9 3.58Z"
      />
    </svg>
  )
}

function Toggle({
  checked,
  onChange,
  label,
  disabled = false,
}: {
  checked: boolean
  onChange: () => void
  label: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={onChange}
      className={`toggle ${checked ? 'toggle-on' : ''}`}
    >
      <span />
    </button>
  )
}

function Dialog({
  title,
  onClose,
  wide = false,
  children,
}: {
  title: string
  onClose: () => void
  wide?: boolean
  children: ReactNode
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="dialog-backdrop" onMouseDown={onClose}>
      <section
        className={`dialog ${wide ? 'dialog-wide' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="dialog-header">
          <h2 id="dialog-title">{title}</h2>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close">
            <Icon name="close" />
          </button>
        </header>
        {children}
      </section>
    </div>
  )
}

function Waveform({
  level,
  active,
  color,
  samplesRef,
}: {
  level: number
  active: boolean
  color: string
  samplesRef: { current: Uint8Array | null }
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stateRef = useRef({ level, active, color })
  useEffect(() => {
    stateRef.current = { level, active, color }
  }, [level, active, color])
  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context) return
    let frame = 0
    let phase = 0
    let smooth = 0.02
    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = rect.width * ratio
      canvas.height = rect.height * ratio
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
    }
    const draw = () => {
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      const target = stateRef.current.active ? Math.max(stateRef.current.level, 0.04) : 0.015
      smooth += (target - smooth) * 0.07
      phase += 0.015 + smooth * 0.03
      context.clearRect(0, 0, width, height)
      context.beginPath()
      context.strokeStyle = stateRef.current.active
        ? stateRef.current.color
        : 'rgba(113,113,122,.58)'
      context.lineWidth = 1.25
      const samples = samplesRef.current
      if (samples && stateRef.current.active) {
        const sliceWidth = width / Math.max(samples.length - 1, 1)
        for (let index = 0; index < samples.length; index += 1) {
          const normalized = (samples[index] - 128) / 128
          const x = index * sliceWidth
          const y = height / 2 + normalized * height * 0.72
          index ? context.lineTo(x, y) : context.moveTo(x, y)
        }
      } else {
        const amplitude = Math.min(height * 0.38, 5 + smooth * height * 0.65)
        for (let x = 0; x <= width; x += 2) {
          const n = x / Math.max(width, 1)
          const envelope = Math.pow(Math.sin(n * Math.PI), 0.68)
          const organic =
            Math.sin(n * 38 + phase * 2.1) * 0.38 +
            Math.sin(n * 83 - phase * 1.4) * 0.24 +
            Math.sin(n * 151 + phase) * 0.13
          const y =
            height / 2 +
            (stateRef.current.active
              ? organic * amplitude * envelope
              : Math.sin(n * 25 + phase) * 1.2)
          x ? context.lineTo(x, y) : context.moveTo(x, y)
        }
      }
      context.stroke()
      frame = requestAnimationFrame(draw)
    }
    resize()
    window.addEventListener('resize', resize)
    frame = requestAnimationFrame(draw)
    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(frame)
    }
  }, [])
  return (
    <canvas
      ref={canvasRef}
      className="wave-canvas"
      role="img"
      aria-label="Live environmental audio waveform"
    />
  )
}

function HelpContent() {
  return (
    <div className="help-content">
      <section>
        <h3>What is Alert?</h3>
        <p>
          Alert detects sounds in your surroundings and translates them into visual subtitles,
          helping you become aware of environmental sounds without relying on hearing.
        </p>
      </section>
      <section>
        <h3>Understanding the three states</h3>
        <dl className="state-guide">
          <div>
            <dt>
              <i className="state-swatch default" />
              Default
            </dt>
            <dd>Normal monitoring. No warning state is currently active.</dd>
          </div>
          <div>
            <dt>
              <i className="state-swatch warning" />
              Yellow — Warning
            </dt>
            <dd>An assigned sound was detected. The warning remains active for 3 seconds.</dd>
          </div>
          <div>
            <dt>
              <i className="state-swatch urgent" />
              Red — Urgent
            </dt>
            <dd>An urgent sound was detected. The urgent state remains active for 5 seconds.</dd>
          </div>
        </dl>
        <p className="help-note">
          Urgent alerts always take priority. A warning received during an urgent alert waits and
          activates after the urgent alert finishes.
        </p>
      </section>
      <section>
        <h3>Sound detection</h3>
        <p>
          Alert identifies the presence of sounds. It does not determine their direction or physical
          location.
        </p>
      </section>
      <section>
        <h3>Customisation</h3>
        <p>
          Use Menu to connect sounds to colour states, vibration, and text-to-speech responses.
          Speech recognition and classification require a compatible detection integration; demo
          tests are always labeled.
        </p>
      </section>
    </div>
  )
}

function SearchBox({
  value,
  onChange,
  placeholder,
}: {
  value: string
  onChange: (value: string) => void
  placeholder: string
}) {
  return (
    <label className="search-box">
      <Icon name="search" />
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
      />
    </label>
  )
}

type EditorProps = {
  rules: SoundRule[]
  states: VisualState[]
  selectedId: string
  setSelectedId: (id: string) => void
  updateRule: (id: string, update: Partial<SoundRule>) => void
  changeEffect: (ruleId: string, effectId: string, visual: boolean) => void
  removeRule: (id: string) => void
  onAddPhrase: () => void
  onAddState: () => void
  onDeleteState: (id: string) => void
  onTest: (id: string) => void
}

function NodeEditor(props: EditorProps) {
  const [search, setSearch] = useState('')
  const workspaceRef = useRef<HTMLDivElement>(null)
  const selectedNodeRef = useRef<HTMLButtonElement | null>(null)
  const effectRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const [connectionPaths, setConnectionPaths] = useState<
    Array<{
      id: string
      d: string
      startX: number
      startY: number
      endX: number
      endY: number
      color: string
    }>
  >([])
  const selected = props.rules.find((rule) => rule.id === props.selectedId) ?? props.rules[0]
  const filtered = props.rules.filter((rule) =>
    rule.label.toLowerCase().includes(search.toLowerCase()),
  )
  const effects = [
    ...props.states.map((state) => ({ id: state.id, name: state.name, category: 'Visual state' })),
    ...ADDITIONAL_EFFECTS,
    ...FUTURE_EFFECTS,
  ]

  const updateConnections = useCallback(() => {
    const workspace = workspaceRef.current
    const source = selectedNodeRef.current
    if (!workspace || !source || !selected) {
      setConnectionPaths([])
      return
    }
    const workspaceRect = workspace.getBoundingClientRect()
    const sourceRect = source.getBoundingClientRect()
    const startX = sourceRect.right - workspaceRect.left
    const startY = sourceRect.top + sourceRect.height / 2 - workspaceRect.top
    const paths = selected.effects.flatMap((effectId) => {
      const target = effectRefs.current.get(effectId)
      if (!target) return []
      const targetRect = target.getBoundingClientRect()
      const endX = targetRect.left - workspaceRect.left
      const endY = targetRect.top + targetRect.height / 2 - workspaceRect.top
      const controlOffset = Math.max(34, (endX - startX) * 0.48)
      const visualState = props.states.find((state) => state.id === effectId)
      return [
        {
          id: effectId,
          d: `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`,
          startX,
          startY,
          endX,
          endY,
          color:
            visualState?.color ?? (FUTURE_EFFECT_IDS.has(effectId) ? '#71717a' : 'var(--accent)'),
        },
      ]
    })
    setConnectionPaths(paths)
  }, [props.states, selected])

  useEffect(() => {
    const workspace = workspaceRef.current
    if (!workspace) return
    let frame = requestAnimationFrame(updateConnections)
    const scheduleUpdate = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(updateConnections)
    }
    const observer = new ResizeObserver(scheduleUpdate)
    observer.observe(workspace)
    workspace.addEventListener('scroll', scheduleUpdate, true)
    window.addEventListener('resize', scheduleUpdate)
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      workspace.removeEventListener('scroll', scheduleUpdate, true)
      window.removeEventListener('resize', scheduleUpdate)
    }
  }, [search, updateConnections])

  return (
    <div className="node-workspace" ref={workspaceRef}>
      <svg className="connection-overlay" width="100%" height="100%" aria-hidden="true">
        {connectionPaths.map((path) => (
          <g key={path.id}>
            <path d={path.d} stroke={path.color} />
            <circle cx={path.startX} cy={path.startY} r="3" fill={path.color} />
            <circle cx={path.endX} cy={path.endY} r="3" fill={path.color} />
          </g>
        ))}
      </svg>
      <section className="node-panel sounds-panel">
        <div className="panel-title">
          <div>
            <span>01</span>
            <h2>Sounds</h2>
          </div>
          <button type="button" onClick={props.onAddPhrase}>
            <Icon name="plus" /> Phrase
          </button>
        </div>
        <SearchBox value={search} onChange={setSearch} placeholder="Search recognized sounds" />
        <div className="node-list">
          {filtered.map((rule) => (
            <button
              key={rule.id}
              ref={selected?.id === rule.id ? selectedNodeRef : undefined}
              type="button"
              className={`sound-node ${selected?.id === rule.id ? 'selected' : ''}`}
              onClick={() => props.setSelectedId(rule.id)}
            >
              <span className="node-port output" />
              <span>
                <b>{rule.label}</b>
                <small>
                  {rule.kind === 'phrase'
                    ? 'Recognized phrase · integration required'
                    : LIVE_RULE_IDS.has(rule.id)
                      ? 'Environmental sound · live detection'
                      : 'Environmental sound · simulation only'}
                </small>
              </span>
              <i className={rule.enabled ? 'enabled' : ''} />
            </button>
          ))}
        </div>
      </section>
      <div className="connection-channel" aria-hidden="true">
        <b>Sound</b>
        <b>Effect</b>
      </div>
      <section className="node-panel effects-panel">
        <div className="panel-title">
          <div>
            <span>02</span>
            <h2>Effects</h2>
          </div>
          <button type="button" onClick={props.onAddState}>
            <Icon name="plus" /> State
          </button>
        </div>
        <div className="selected-rule-bar">
          <span>
            Editing <b>{selected?.label}</b>
          </span>
          {selected?.kind === 'phrase' && (
            <div>
              <button type="button" onClick={() => props.removeRule(selected.id)}>
                Remove
              </button>
            </div>
          )}
        </div>
        <div className="effect-groups">
          {['Visual state', 'Physical feedback', 'Additional effects', 'Future effects'].map(
            (category) => (
              <div className="effect-group" key={category}>
                <h3>{category}</h3>
                {effects
                  .filter((effect) => effect.category === category)
                  .map((effect) => {
                    const checked = selected?.effects.includes(effect.id) ?? false
                    const visualState = props.states.find((state) => state.id === effect.id)
                    const custom = visualState && !visualState.system ? visualState : undefined
                    return (
                      <div
                        className={`effect-node ${checked ? 'connected' : ''} ${effect.id === 'urgent' ? 'effect-urgent' : ''}`}
                        key={effect.id}
                        ref={(node) => {
                          if (node) effectRefs.current.set(effect.id, node)
                          else effectRefs.current.delete(effect.id)
                        }}
                      >
                        <span className="node-port input" />
                        {visualState && (
                          <i
                            className="effect-color"
                            style={{ backgroundColor: visualState.color }}
                          />
                        )}
                        <button
                          type="button"
                          onClick={() =>
                            selected &&
                            props.changeEffect(selected.id, effect.id, category === 'Visual state')
                          }
                        >
                          <span>{effect.name}</span>
                          <small>
                            {checked
                              ? visualState
                                ? 'Assigned state'
                                : category === 'Future effects'
                                  ? 'Configured · integration required'
                                  : 'Connected'
                              : visualState
                                ? `${visualState.duration / 1000 || 0}s display`
                                : category === 'Future effects'
                                  ? 'Future integration'
                                  : 'Click to connect'}
                          </small>
                        </button>
                        {custom && (
                          <button
                            className="remove-mini"
                            type="button"
                            aria-label={`Delete ${custom.name}`}
                            onClick={() => props.onDeleteState(custom.id)}
                          >
                            ×
                          </button>
                        )}
                      </div>
                    )
                  })}
              </div>
            ),
          )}
        </div>
        {selected?.effects.includes('tts') && (
          <label className="tts-field">
            Outgoing spoken response
            <input
              value={selected.ttsResponse ?? ''}
              maxLength={300}
              onChange={(event) =>
                props.updateRule(selected.id, { ttsResponse: event.target.value })
              }
              placeholder="Text Alert should speak"
            />
            <small>Generated after a compatible detector recognizes this input.</small>
          </label>
        )}
        {selected?.effects.includes('smart-lights') && (
          <label className="light-colour-field">
            Light colour
            <span>
              <input
                type="color"
                value={selected.lightColor ?? '#2dd4bf'}
                onChange={(event) =>
                  props.updateRule(selected.id, { lightColor: event.target.value })
                }
              />
              <b>{(selected.lightColor ?? '#2dd4bf').toUpperCase()}</b>
            </span>
            <small>Requires a compatible smart-light integration.</small>
          </label>
        )}
        <div className="node-test-action">
          <button
            className="primary-action"
            type="button"
            onClick={() => selected && props.onTest(selected.id)}
          >
            Test selected sound
          </button>
        </div>
      </section>
    </div>
  )
}

function SimpleEditor(props: EditorProps) {
  const [search, setSearch] = useState('')
  const selected = props.rules.find((rule) => rule.id === props.selectedId) ?? props.rules[0]
  const visible = props.rules.filter((rule) =>
    rule.label.toLowerCase().includes(search.toLowerCase()),
  )
  return (
    <div className="simple-editor">
      <section className="simple-index">
        <div className="panel-title">
          <div>
            <span>01</span>
            <h2>Sound index</h2>
          </div>
          <button type="button" onClick={props.onAddPhrase}>
            <Icon name="plus" /> Phrase
          </button>
        </div>
        <SearchBox value={search} onChange={setSearch} placeholder="Search sounds" />
        <div className="index-list">
          {visible.map((rule) => (
            <button
              key={rule.id}
              type="button"
              className={selected?.id === rule.id ? 'selected' : ''}
              onClick={() => props.setSelectedId(rule.id)}
            >
              <span>{String(props.rules.indexOf(rule) + 1).padStart(2, '0')}.</span>
              <b>{rule.label}</b>
              <small>{rule.kind === 'phrase' ? 'Phrase' : 'Sound'}</small>
            </button>
          ))}
        </div>
      </section>
      <section className="simple-effects">
        <div className="panel-title">
          <div>
            <span>02</span>
            <h2>Effects</h2>
          </div>
          <button type="button" onClick={props.onAddState}>
            <Icon name="plus" /> State
          </button>
        </div>
        <div className="simple-selection">
          <p>Selected sound</p>
          <h3>{selected?.label}</h3>
          {selected?.kind === 'phrase' && (
            <p className="integration-note">
              Incoming phrase recognition requires a speech-processing integration.
            </p>
          )}
        </div>
        <fieldset>
          <legend>Assigned visual state</legend>
          {props.states.map((state) => {
            const checked = selected?.effects.includes(state.id) ?? false
            return (
              <label key={state.id} className={state.id === 'urgent' ? 'simple-urgent-state' : ''}>
                <input
                  type="radio"
                  name={`visual-state-${selected?.id}`}
                  checked={checked}
                  onChange={() => selected && props.changeEffect(selected.id, state.id, true)}
                />
                <i className="simple-state-colour" style={{ backgroundColor: state.color }} />
                <span>{state.name}</span>
              </label>
            )
          })}
          <legend className="secondary-legend">Additional effects</legend>
          {ADDITIONAL_EFFECTS.map((effect) => {
            const checked = selected?.effects.includes(effect.id) ?? false
            return (
              <label key={effect.id}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => selected && props.changeEffect(selected.id, effect.id, false)}
                />
                <span>{effect.name}</span>
              </label>
            )
          })}
          <details className="future-effects">
            <summary>More integrations (not available yet)</summary>
            {FUTURE_EFFECTS.map((effect) => {
              const checked = selected?.effects.includes(effect.id) ?? false
              return (
                <label key={effect.id} className="future-effect-option">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => selected && props.changeEffect(selected.id, effect.id, false)}
                  />
                  <span>{effect.name}</span>
                </label>
              )
            })}
          </details>
        </fieldset>
        {selected?.effects.includes('tts') && (
          <label className="tts-field">
            Outgoing text-to-speech response
            <input
              value={selected.ttsResponse ?? ''}
              maxLength={300}
              onChange={(event) =>
                props.updateRule(selected.id, { ttsResponse: event.target.value })
              }
              placeholder="Enter response"
            />
          </label>
        )}
        {selected?.effects.includes('smart-lights') && (
          <label className="light-colour-field">
            Turn lights this colour
            <span>
              <input
                type="color"
                value={selected.lightColor ?? '#2dd4bf'}
                onChange={(event) =>
                  props.updateRule(selected.id, { lightColor: event.target.value })
                }
              />
              <b>{(selected.lightColor ?? '#2dd4bf').toUpperCase()}</b>
            </span>
            <small>Saved now; hardware integration is required to control lights.</small>
          </label>
        )}
        <div className="simple-actions">
          <button
            type="button"
            className="primary-action"
            onClick={() => selected && props.onTest(selected.id)}
          >
            Test selected sound
          </button>
          {selected?.kind === 'phrase' && (
            <button type="button" onClick={() => props.removeRule(selected.id)}>
              Remove phrase
            </button>
          )}
        </div>
      </section>
    </div>
  )
}

function StatePriorityPanel({
  states,
  collapsed,
  onToggle,
  onReorder,
  onEdit,
  onDelete,
}: {
  states: VisualState[]
  collapsed: boolean
  onToggle: () => void
  onReorder: (draggedId: string, targetId: string) => void
  onEdit: (state: VisualState) => void
  onDelete: (id: string) => void
}) {
  const [draggedId, setDraggedId] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  return (
    <aside
      className={`priority-panel ${collapsed ? 'priority-panel-collapsed' : ''}`}
      aria-labelledby={collapsed ? undefined : 'priority-title'}
      aria-label={collapsed ? 'State priority' : undefined}
    >
      <div className="priority-heading">
        {!collapsed && (
          <div>
            <span>State priority</span>
            <h2 id="priority-title">Highest first</h2>
          </div>
        )}
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={!collapsed}
          aria-label={collapsed ? 'Expand state priority' : 'Minimize state priority'}
          title={collapsed ? 'Expand state priority' : 'Minimize state priority'}
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path
              d={collapsed ? 'm5 3 4 4-4 4' : 'm9 3-4 4 4 4'}
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      {!collapsed && <p>Drag states to determine which alert takes precedence.</p>}
      <ol className="priority-list">
        {states.map((state, index) => (
          <li
            key={state.id}
            title={collapsed ? state.name : undefined}
            draggable={state.id !== 'urgent' && state.id !== 'default'}
            className={draggedId === state.id ? 'dragging' : ''}
            onDragStart={(event) => {
              if (state.id === 'urgent' || state.id === 'default') {
                event.preventDefault()
                return
              }
              setDraggedId(state.id)
              event.dataTransfer.effectAllowed = 'move'
              event.dataTransfer.setData('text/plain', state.id)
            }}
            onDragOver={(event) => {
              event.preventDefault()
              event.dataTransfer.dropEffect = 'move'
            }}
            onDrop={(event) => {
              event.preventDefault()
              const source = event.dataTransfer.getData('text/plain') || draggedId
              if (source && source !== state.id) onReorder(source, state.id)
              setDraggedId(null)
            }}
            onDragEnd={() => setDraggedId(null)}
          >
            <span className="priority-number">{String(index + 1).padStart(2, '0')}</span>
            <span
              className={`drag-handle ${state.id === 'urgent' || state.id === 'default' ? 'fixed' : ''}`}
              aria-label={
                state.id === 'urgent' || state.id === 'default'
                  ? `${state.name} has fixed priority`
                  : `Drag ${state.name}`
              }
              title={
                state.id === 'urgent' || state.id === 'default'
                  ? 'Fixed priority'
                  : 'Drag to reorder'
              }
            >
              <i />
              <i />
              <i />
            </span>
            <i className="priority-colour" style={{ backgroundColor: state.color }} />
            <span className="priority-name">
              <b>{state.name}</b>
              <small>{state.system ? 'System state' : `${state.duration / 1000}s duration`}</small>
            </span>
            {!state.system && (
              <span className="priority-actions">
                <button type="button" onClick={() => onEdit(state)}>
                  Edit
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (confirmDeleteId === state.id) {
                      onDelete(state.id)
                      setConfirmDeleteId(null)
                    } else {
                      setConfirmDeleteId(state.id)
                    }
                  }}
                >
                  {confirmDeleteId === state.id ? 'Confirm' : 'Delete'}
                </button>
              </span>
            )}
          </li>
        ))}
      </ol>
    </aside>
  )
}

/** Real login (Auth0). null = Auth0 not configured, so the local preview account is used. */
type RealAuth = {
  loading: boolean
  user: { id: string; name: string; email: string } | null
  login: (mode: 'login' | 'create' | 'google') => void
  logout: () => void
}

function AlertApp({ auth }: { auth: RealAuth | null }) {
  const [view, setView] = useState<View>('home')
  const [simple, setSimple] = useState(DEFAULT_SETTINGS.simple)
  const [rules, setRules] = useState<SoundRule[]>(DEFAULT_SETTINGS.rules)
  const [customStates, setCustomStates] = useState<VisualState[]>(DEFAULT_SETTINGS.customStates)
  const [stateOrder, setStateOrder] = useState(DEFAULT_SETTINGS.stateOrder)
  const [introComplete, setIntroComplete] = useState(DEFAULT_SETTINGS.introComplete)
  const [dialog, setDialog] = useState<DialogName>(null)
  const [introHelp, setIntroHelp] = useState(false)
  const [localAccount, setLocalAccount] = useState<{ name: string; email: string } | null>(null)
  const account = auth ? auth.user : localAccount
  const accountKey = auth ? (auth.user?.id ?? null) : (localAccount?.email ?? null)
  const settingsStorageKey = `${STORAGE_KEY}:${accountKey ? encodeURIComponent(accountKey) : 'guest'}`
  const [accountReady, setAccountReady] = useState(false)
  const [loadedSettingsKey, setLoadedSettingsKey] = useState<string | null>(null)
  const settingsReady = accountReady && loadedSettingsKey === settingsStorageKey
  const [accountMode, setAccountMode] = useState<'login' | 'create'>('login')
  const [listening, setListening] = useState<ListeningState>('paused')
  const [audioLevel, setAudioLevel] = useState(0)
  const [detections, setDetections] = useState<Detection[]>([])
  const [clock, setClock] = useState(() => Date.now())
  const [activeAlert, setActiveAlert] = useState<ActiveAlert | null>(null)
  const [pendingAlerts, setPendingAlerts] = useState<AlertRequest[]>([])
  const [selectedId, setSelectedId] = useState(DEFAULT_SETTINGS.selectedRuleId)
  const [priorityCollapsed, setPriorityCollapsed] = useState(DEFAULT_SETTINGS.priorityCollapsed)
  const [saveStatus, setSaveStatus] = useState<'saved' | 'error'>('saved')
  const [phraseDraft, setPhraseDraft] = useState('')
  const [showPhraseForm, setShowPhraseForm] = useState(false)
  const [editingStateId, setEditingStateId] = useState<string | null>(null)
  const [stateDraft, setStateDraft] = useState({
    name: '',
    color: '#4f7fd8',
    duration: 4,
  })
  const audioRef = useRef<{ context: AudioContext; stream: MediaStream; frame: number } | null>(
    null,
  )
  const waveformSamplesRef = useRef<Uint8Array | null>(null)
  const simpleRef = useRef(simple)
  const activeRef = useRef<ActiveAlert | null>(null)
  const pendingRef = useRef<AlertRequest[]>([])
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking')
  const triggerRuleRef = useRef<(id: string, live?: { confidence: number }) => void>(() => {})
  const sustainRuleRef = useRef<(id: string, live: { confidence: number }) => void>(() => {})
  const detectBusyRef = useRef(false)
  const detectPausedUntilRef = useRef(0)
  const micSessionRef = useRef(0)
  // True while the server says the model is still starting (e.g. Databricks waking from scale-to-zero).
  // /health cannot see that, so it must not flip the status back to "ready" until a chunk succeeds.
  const modelWakingRef = useRef(false)

  const checkBackend = useCallback(async () => {
    try {
      const health = await getHealth()
      // With Auth0 on the server, detection needs a logged-in user's access token.
      const allowed =
        health.anonymous_detection || (health.auth_configured && (await getAccessToken()))
      if (allowed) {
        const model = health.model.status
        if (model === 'ready' && modelWakingRef.current) setBackendStatus('loading')
        else setBackendStatus(model === 'ready' ? 'ready' : model === 'failed' ? 'error' : 'loading')
      } else {
        setBackendStatus('login-required')
      }
    } catch {
      setBackendStatus('offline')
    }
  }, [])
  useEffect(() => {
    void checkBackend()
  }, [checkBackend, accountKey])
  useEffect(() => {
    // Keep checking while the model warms up or the server is down, so the status recovers by itself.
    if (backendStatus !== 'loading' && backendStatus !== 'offline') return
    const interval = window.setInterval(() => void checkBackend(), 4000)
    return () => window.clearInterval(interval)
  }, [backendStatus, checkBackend])

  const handleDetection = useCallback((result: DetectionResult) => {
    modelWakingRef.current = false
    setBackendStatus('ready')
    if (!result.detected || !result.sound_type) return
    const ruleId = BACKEND_SOUND_TO_RULE[result.sound_type]
    if (!ruleId) return
    // The first chunk of an event triggers the alert (vibration included). Later chunks of the same
    // ongoing sound keep it on screen, so a fire alarm that keeps ringing keeps the screen red.
    if (result.is_new_event) triggerRuleRef.current(ruleId, { confidence: result.confidence })
    else sustainRuleRef.current(ruleId, { confidence: result.confidence })
  }, [])

  const sendChunk = useCallback(
    async (samples: Float32Array, sampleRate: number, session: number) => {
      // Drop chunks rather than queue them: a late alert is worse than a skipped chunk.
      if (detectBusyRef.current || Date.now() < detectPausedUntilRef.current) return
      detectBusyRef.current = true
      try {
        const result = await detectSound(encodeWav(samples, sampleRate), getAccessToken)
        if (session === micSessionRef.current) handleDetection(result)
      } catch (error) {
        if (!(error instanceof ApiError)) {
          setBackendStatus('offline')
          detectPausedUntilRef.current = Date.now() + 4000
        } else if (error.status === 401) {
          setBackendStatus('login-required')
          detectPausedUntilRef.current = Date.now() + 10000
        } else if (error.code === 'model_unavailable') {
          // Model still starting: keep sending chunks; they keep the endpoint awake once it is up.
          modelWakingRef.current = true
          setBackendStatus('loading')
        } else if (error.status === 429) {
          detectPausedUntilRef.current = Date.now() + (error.retryAfter ?? 5) * 1000
        } else if (error.status >= 500) {
          setBackendStatus('error')
        }
        // 422 invalid_audio (e.g. a glitchy chunk) is ignored; the next chunk is tried normally.
      } finally {
        detectBusyRef.current = false
      }
    },
    [handleDetection],
  )

  const states = useMemo(() => {
    const available = [...SYSTEM_STATES, ...customStates]
    return [
      ...stateOrder
        .map((id) => available.find((state) => state.id === id))
        .filter((state): state is VisualState => Boolean(state)),
      ...available.filter((state) => !stateOrder.includes(state.id)),
    ]
  }, [customStates, stateOrder])
  const currentState = states.find((state) => state.id === activeAlert?.stateId) ?? SYSTEM_STATES[0]
  const previewState: VisualState =
    dialog === 'custom-state' && !activeAlert
      ? {
          id: 'preview',
          name: stateDraft.name || 'Custom state',
          color: stateDraft.color,
          priority: 1,
          duration: stateDraft.duration * 1000,
        }
      : currentState
  const isUrgent = previewState.priority === 2
  const isWarning = previewState.priority === 1

  useEffect(() => {
    activeRef.current = activeAlert
  }, [activeAlert])
  useEffect(() => {
    simpleRef.current = simple
    if (simple) setAudioLevel(0)
  }, [simple])
  useEffect(() => {
    pendingRef.current = pendingAlerts
  }, [pendingAlerts])
  useEffect(() => {
    if (auth) {
      if (!auth.loading) setAccountReady(true)
      return
    }
    try {
      const saved = JSON.parse(localStorage.getItem(ACCOUNT_KEY) || 'null') as {
        name: string
        email: string
      } | null
      setLocalAccount(saved)
    } catch {
      setLocalAccount(null)
    } finally {
      setAccountReady(true)
    }
  }, [auth])

  useEffect(() => {
    if (!accountReady) {
      setLoadedSettingsKey(null)
      return
    }
    const restored = loadSettings(settingsStorageKey)
    setSimple(restored.simple)
    setRules(restored.rules)
    setCustomStates(restored.customStates)
    setStateOrder(restored.stateOrder)
    setSelectedId(
      restored.rules.some((rule) => rule.id === restored.selectedRuleId)
        ? restored.selectedRuleId
        : (restored.rules[0]?.id ?? ''),
    )
    setPriorityCollapsed(restored.priorityCollapsed)
    setIntroComplete(restored.introComplete)
    setLoadedSettingsKey(settingsStorageKey)
  }, [accountReady, settingsStorageKey])

  const settings = useMemo<StoredSettings>(
    () => ({
      simple,
      rules,
      customStates,
      stateOrder,
      selectedRuleId: selectedId,
      priorityCollapsed,
      introComplete,
    }),
    [customStates, introComplete, priorityCollapsed, rules, selectedId, simple, stateOrder],
  )
  useEffect(() => {
    // On a real Auth0 login, record the user in the backend's users table (Databricks).
    // Best-effort: login and the app keep working if the backend or database is unavailable.
    if (!auth || !accountKey) return
    void (async () => {
      try {
        if (!(await getAccessToken())) return
        if (!(await getHealth()).database_configured) return
        await recordLogin(getAccessToken)
      } catch {
        /* user is still logged in; the row is written on the next login */
      }
    })()
  }, [auth, accountKey])

  useEffect(() => {
    if (!accountReady || loadedSettingsKey !== settingsStorageKey) return
    try {
      localStorage.setItem(settingsStorageKey, JSON.stringify(settings))
      setSaveStatus('saved')
    } catch {
      setSaveStatus('error')
    }
  }, [accountReady, loadedSettingsKey, settings, settingsStorageKey])

  const stateDuration = useCallback(
    (stateId: string) => {
      return states.find((state) => state.id === stateId)?.duration ?? 3000
    },
    [states],
  )
  const statePriority = useCallback(
    (stateId: string) => {
      const index = states.findIndex((state) => state.id === stateId)
      return index === -1 ? 0 : states.length - index
    },
    [states],
  )

  const activate = useCallback(
    (request: AlertRequest) => {
      const now = Date.now()
      setActiveAlert({
        ...request,
        startedAt: now,
        expiresAt: now + stateDuration(request.stateId),
      })
    },
    [stateDuration],
  )

  const enqueueAlert = useCallback(
    (stateId: string, soundName: string) => {
      if (stateId === 'default') return
      const incomingPriority = statePriority(stateId)
      const current = activeRef.current
      if (!current) {
        activate({ stateId, soundNames: [soundName] })
        return
      }
      const currentPriority = statePriority(current.stateId)
      if (incomingPriority > currentPriority) {
        setPendingAlerts((items) =>
          items.filter((item) => statePriority(item.stateId) >= incomingPriority),
        )
        activate({ stateId, soundNames: [soundName] })
        return
      }
      if (incomingPriority === currentPriority && current.stateId === stateId) {
        if (!current.soundNames.includes(soundName))
          setActiveAlert({ ...current, soundNames: [...current.soundNames, soundName] })
        return
      }
      setPendingAlerts((items) => {
        const existing = items.find((item) => item.stateId === stateId)
        if (existing)
          return items.map((item) =>
            item.stateId === stateId && !item.soundNames.includes(soundName)
              ? { ...item, soundNames: [...item.soundNames, soundName] }
              : item,
          )
        return [...items, { stateId, soundNames: [soundName] }]
      })
    },
    [activate, statePriority],
  )

  useEffect(() => {
    if (!detections.length && !activeAlert) return
    const interval = window.setInterval(() => {
      const now = Date.now()
      setClock(now)
      setDetections((items) => items.filter((item) => item.expiresAt > now))
    }, 250)
    return () => window.clearInterval(interval)
  }, [activeAlert, detections.length])

  useEffect(() => {
    if (!activeAlert) return
    const timeout = window.setTimeout(
      () => {
        const queue = [...pendingRef.current].sort(
          (a, b) => statePriority(b.stateId) - statePriority(a.stateId),
        )
        const next = queue[0]
        setPendingAlerts(queue.slice(1))
        if (next) activate(next)
        else setActiveAlert(null)
      },
      Math.max(0, activeAlert.expiresAt - Date.now()),
    )
    return () => window.clearTimeout(timeout)
  }, [activeAlert, activate, statePriority])

  const updateRule = (id: string, update: Partial<SoundRule>) => {
    setRules((items) => items.map((rule) => (rule.id === id ? { ...rule, ...update } : rule)))
  }
  const changeEffect = (ruleId: string, effectId: string, visual: boolean) => {
    const visualIds = new Set(states.map((state) => state.id))
    setRules((items) =>
      items.map((rule) => {
        if (rule.id !== ruleId) return rule
        if (visual) {
          return {
            ...rule,
            effects: [effectId, ...rule.effects.filter((effect) => !visualIds.has(effect))],
          }
        }
        return {
          ...rule,
          effects: rule.effects.includes(effectId)
            ? rule.effects.filter((effect) => effect !== effectId)
            : [...rule.effects, effectId],
        }
      }),
    )
  }
  const removeRule = (id: string) => {
    setRules((items) => items.filter((rule) => rule.id !== id))
    setSelectedId(rules.find((rule) => rule.id !== id)?.id ?? '')
  }
  const resetSettings = () => {
    setRules(DEFAULT_RULES)
    setCustomStates([])
    setStateOrder(SYSTEM_STATE_ORDER)
    setSimple(DEFAULT_SETTINGS.simple)
    setPriorityCollapsed(DEFAULT_SETTINGS.priorityCollapsed)
    setSelectedId(DEFAULT_RULES[0].id)
  }

  const stopMicrophone = useCallback(() => {
    micSessionRef.current += 1 // ignore results of requests still in flight
    const audio = audioRef.current
    if (audio) {
      cancelAnimationFrame(audio.frame)
      audio.stream.getTracks().forEach((track) => track.stop())
      void audio.context.close()
      audioRef.current = null
    }
    waveformSamplesRef.current = null
    setAudioLevel(0)
    setListening('paused')
  }, [])
  const startMicrophone = async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone API unavailable')
      // Request the broad permission first; strict device constraints can reject before a prompt.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const track = stream.getAudioTracks()[0]
      if (track) {
        void track
          .applyConstraints({
            autoGainControl: false,
            echoCancellation: false,
            noiseSuppression: false,
          })
          .catch(() => {
            // Optional raw-audio tuning is not supported by every browser/device.
          })
      }
      const context = new AudioContext()
      const analyser = context.createAnalyser()
      analyser.fftSize = 256
      analyser.smoothingTimeConstant = 0.82
      const source = context.createMediaStreamSource(stream)
      source.connect(analyser)
      const data = new Uint8Array(analyser.fftSize)
      const audio = { context, stream, frame: 0 }
      audioRef.current = audio
      const session = ++micSessionRef.current
      // Live detection: stream 2 s chunks to the backend. If capture fails, the waveform still works.
      void startChunkCapture(context, source, CHUNK_SECONDS, (samples) => {
        void sendChunk(samples, context.sampleRate, session)
      }).catch(() => setBackendStatus('error'))
      // Mic on: start the sound model now (on Databricks it sleeps while nobody listens). The chunks
      // that follow every 2 s keep it awake; once the mic is off it goes back to sleep by itself.
      void wakeModel(getAccessToken)
        .then((woken) => {
          if (woken.status === 'waking' && session === micSessionRef.current) {
            modelWakingRef.current = true
            setBackendStatus('loading')
          }
        })
        .catch(() => {
          /* detection chunks report their own errors */
        })
      void checkBackend()
      const sample = () => {
        analyser.getByteTimeDomainData(data)
        waveformSamplesRef.current = data.slice()
        if (!simpleRef.current) {
          const rootMeanSquare = Math.sqrt(
            data.reduce((sum, value) => {
              const normalized = (value - 128) / 128
              return sum + normalized * normalized
            }, 0) / data.length,
          )
          setAudioLevel(Math.min(rootMeanSquare * 4, 1))
        }
        audio.frame = requestAnimationFrame(sample)
      }
      sample()
      setListening('listening')
    } catch {
      setListening('unavailable')
    }
  }
  useEffect(() => stopMicrophone, [stopMicrophone])

  const triggerRule = useCallback(
    (id: string, live?: { confidence: number }) => {
      const rule = rules.find((item) => item.id === id)
      if (!rule || !rule.enabled) return
      setDetections((items) => {
        const next = items.filter((item) => item.id !== id)
        const urgent = rule.effects.some(
          (effect) => states.find((state) => state.id === effect)?.priority === 2,
        )
        return [
          ...next,
          {
            id,
            label: rule.label,
            critical: urgent,
            expiresAt: Date.now() + 6500,
            live: Boolean(live),
            confidence: live?.confidence,
          },
        ].slice(-6)
      })
      const visualEffect = rule.effects.find((effect) =>
        states.some((state) => state.id === effect),
      )
      if (visualEffect) enqueueAlert(visualEffect, rule.label)
      if (rule.effects.includes('vibration') && navigator.vibrate) navigator.vibrate([120, 70, 120])
      if (!simpleRef.current) {
        setAudioLevel(0.72)
        window.setTimeout(() => setAudioLevel(0.04), 900)
      }
    },
    [enqueueAlert, rules, statePriority, states],
  )
  const sustainRule = useCallback(
    (id: string, live: { confidence: number }) => {
      // Same sound still going: extend what is already shown instead of re-alerting (no new vibration).
      const rule = rules.find((item) => item.id === id)
      if (!rule || !rule.enabled) return
      const now = Date.now()
      setDetections((items) =>
        items.some((item) => item.id === id)
          ? items.map((item) =>
              item.id === id
                ? { ...item, expiresAt: now + 6500, confidence: live.confidence }
                : item,
            )
          : items,
      )
      const visualEffect = rule.effects.find((effect) =>
        states.some((state) => state.id === effect),
      )
      if (!visualEffect || visualEffect === 'default') return
      const current = activeRef.current
      if (current && current.stateId === visualEffect) {
        setActiveAlert({
          ...current,
          soundNames: current.soundNames.includes(rule.label)
            ? current.soundNames
            : [...current.soundNames, rule.label],
          expiresAt: Math.max(current.expiresAt, now + stateDuration(visualEffect)),
        })
      } else {
        // The alert already ended (or a lower one is showing) while the sound is still going: bring it back.
        enqueueAlert(visualEffect, rule.label)
      }
    },
    [enqueueAlert, rules, stateDuration, states],
  )

  useEffect(() => {
    // The audio callback lives for the whole mic session; the ref gives it the latest rules/states.
    sustainRuleRef.current = sustainRule
    triggerRuleRef.current = triggerRule
  }, [sustainRule, triggerRule])

  const addPhrase = () => {
    const label = phraseDraft.trim()
    if (!label) return
    const id = `phrase-${Date.now()}`
    setRules((items) => [
      ...items,
      { id, label, kind: 'phrase', enabled: true, effects: ['warning'] },
    ])
    setSelectedId(id)
    setPhraseDraft('')
    setShowPhraseForm(false)
  }
  const addCustomState = () => {
    if (!stateDraft.name.trim()) return
    if (editingStateId) {
      setCustomStates((items) =>
        items.map((state) =>
          state.id === editingStateId
            ? {
                ...state,
                name: stateDraft.name.trim(),
                color: stateDraft.color,
                duration: stateDraft.duration * 1000,
              }
            : state,
        ),
      )
    } else {
      const state: VisualState = {
        id: `custom-${Date.now()}`,
        name: stateDraft.name.trim(),
        color: stateDraft.color,
        priority: 1,
        duration: stateDraft.duration * 1000,
      }
      setCustomStates((items) => [...items, state])
      setStateOrder((items) => {
        const defaultIndex = items.indexOf('default')
        const insertion = defaultIndex === -1 ? items.length : defaultIndex
        return [...items.slice(0, insertion), state.id, ...items.slice(insertion)]
      })
    }
    setEditingStateId(null)
    setStateDraft({ name: '', color: '#4f7fd8', duration: 4 })
    setDialog(null)
  }
  const deleteCustomState = (id: string) => {
    setCustomStates((items) => items.filter((state) => state.id !== id))
    setStateOrder((items) => items.filter((stateId) => stateId !== id))
    setRules((items) =>
      items.map((rule) =>
        rule.effects.includes(id)
          ? {
              ...rule,
              effects: [
                'default',
                ...rule.effects.filter(
                  (effect) => effect !== id && !states.some((state) => state.id === effect),
                ),
              ],
            }
          : rule,
      ),
    )
    setPendingAlerts((items) => items.filter((alert) => alert.stateId !== id))
    if (activeRef.current?.stateId === id) setActiveAlert(null)
  }
  const reorderStates = (draggedId: string, targetId: string) => {
    if (draggedId === 'urgent' || draggedId === 'default') return
    setStateOrder((items) => {
      const next = items.filter((id) => id !== draggedId)
      const targetIndex =
        targetId === 'urgent'
          ? 1
          : targetId === 'default'
            ? Math.max(1, next.length - 1)
            : next.indexOf(targetId)
      if (targetIndex === -1) return items
      next.splice(targetIndex, 0, draggedId)
      return ['urgent', ...next.filter((id) => id !== 'urgent' && id !== 'default'), 'default']
    })
  }
  const openCreateState = () => {
    setEditingStateId(null)
    setStateDraft({ name: '', color: '#4f7fd8', duration: 4 })
    setDialog('custom-state')
  }
  const openEditState = (state: VisualState) => {
    if (state.system) return
    setEditingStateId(state.id)
    setStateDraft({
      name: state.name,
      color: state.color,
      duration: state.duration / 1000,
    })
    setDialog('custom-state')
  }

  const completeHelp = () => {
    if (introHelp) {
      setIntroComplete(true)
    }
    setIntroHelp(false)
    setDialog(null)
  }
  const accountSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    const email = String(form.get('email'))
    const name = String(form.get('name') || email.split('@')[0])
    const next = { name, email }
    localStorage.setItem(ACCOUNT_KEY, JSON.stringify(next))
    setLocalAccount(next)
    setDialog(null)
    if (accountMode === 'create' && !introComplete) {
      setIntroHelp(true)
      setDialog('help')
    }
  }

  const isCustom = !previewState.system
  const style = {
    '--accent': previewState.color,
    '--app-bg': isCustom
      ? `color-mix(in srgb, ${previewState.color} 13%, #09090b)`
      : isUrgent
        ? '#1b0d0f'
        : isWarning
          ? '#1b1708'
          : '#09090b',
    '--surface': isCustom
      ? `color-mix(in srgb, ${previewState.color} 17%, #121215)`
      : isUrgent
        ? '#251113'
        : isWarning
          ? '#24200f'
          : '#121215',
    '--surface-2': isCustom
      ? `color-mix(in srgb, ${previewState.color} 20%, #18181b)`
      : isUrgent
        ? '#2c1517'
        : isWarning
          ? '#2b2510'
          : '#18181b',
    '--border': isCustom
      ? `color-mix(in srgb, ${previewState.color} 45%, #27272a)`
      : isUrgent
        ? '#613036'
        : isWarning
          ? '#65572b'
          : '#27272a',
    '--border-strong': isCustom
      ? `color-mix(in srgb, ${previewState.color} 58%, #3f3f46)`
      : isUrgent
        ? '#84434b'
        : isWarning
          ? '#806f36'
          : '#3f3f46',
    '--text': isUrgent ? '#fff1f2' : isWarning ? '#fffbeb' : '#f4f4f5',
    '--text-2': isUrgent ? '#e7b8bd' : isWarning ? '#d9c98e' : '#a1a1aa',
    '--text-3': isUrgent ? '#a96b72' : isWarning ? '#9f8c4d' : '#71717a',
  } as CSSProperties

  const editorProps: EditorProps = {
    rules,
    states,
    selectedId,
    setSelectedId,
    updateRule,
    changeEffect,
    removeRule,
    onAddPhrase: () => setShowPhraseForm(true),
    onAddState: openCreateState,
    onDeleteState: deleteCustomState,
    onTest: triggerRule,
  }

  return (
    <div
      className={`alert-app ${simple ? 'simple-mode' : ''} ${simple && view === 'home' ? 'simple-home' : ''}`}
      style={style}
      data-state={currentState.id}
    >
      <header className="topbar">
        <button className="brand" type="button" onClick={() => setView('home')}>
          <AlertLogo />
          <span>ALERT</span>
        </button>
        <nav className="nav-controls" aria-label="Application controls">
          <button
            className="nav-button help-control"
            type="button"
            aria-label="Help"
            onClick={() => {
              setIntroHelp(false)
              setDialog('help')
            }}
          >
            <Icon name="help" />
            {simple && <span>Help</span>}
          </button>
          <div className="simply-control">
            <span>Simple UI</span>
            <Toggle
              checked={simple}
              label="Toggle simple settings view"
              disabled={!settingsReady}
              onChange={() => {
                setSimple((value) => !value)
              }}
            />
          </div>
          <button
            className={`nav-button ${view === 'configure' ? 'active' : ''}`}
            type="button"
            aria-label="Menu"
            disabled={!settingsReady}
            onClick={() => setView('configure')}
          >
            <Icon name="menu" />
            {simple && <span className="menu-label">Menu</span>}
          </button>
          <button
            className="nav-button login-control"
            type="button"
            onClick={() => setDialog('account')}
          >
            {account ? <Icon name="user" /> : <span>Log in</span>}
          </button>
        </nav>
      </header>

      {view === 'home' && simple ? (
        <main className="simple-home-main">
          <section className="simple-home-content" aria-labelledby="simple-status">
            <div className="simple-listening">
              <span
                className={`status-dot ${listening !== 'listening' ? 'status-inactive' : ''}`}
              />
              <p id="simple-status">
                {listening === 'listening'
                  ? 'Listening…'
                  : listening === 'unavailable'
                    ? 'Microphone unavailable'
                    : 'Detection paused'}
              </p>
            </div>
            {backendStatus !== 'ready' && (
              <p className="integration-note">
                {API_STATUS_HINT[backendStatus].charAt(0).toUpperCase() +
                  API_STATUS_HINT[backendStatus].slice(1)}
                .
              </p>
            )}
            {listening !== 'listening' && (
              <button className="simple-start" type="button" onClick={() => void startMicrophone()}>
                Start listening
              </button>
            )}
            {activeAlert && (
              <p className="simple-alert-label" role="alert">
                <span>{currentState.name}</span>
                {activeAlert.soundNames.join(', ')} detected
              </p>
            )}
            <div className="simple-sounds" aria-live="polite">
              {detections.length ? (
                <ul>
                  {detections.map((detection) => (
                    <li key={detection.id} className={detection.critical ? 'critical' : ''}>
                      {detection.label}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>Detected sounds will appear here.</p>
              )}
            </div>
          </section>
        </main>
      ) : view === 'home' ? (
        <main className="workspace">
          <section className="visualizer" aria-labelledby="status-label">
            <div className="content-guide status-row">
              <span
                className={`status-dot ${listening !== 'listening' ? 'status-inactive' : ''}`}
              />
              <span id="status-label">
                {listening === 'listening'
                  ? 'LISTENING'
                  : listening === 'unavailable'
                    ? 'MICROPHONE UNAVAILABLE'
                    : 'PAUSED'}
              </span>
              <span className="demo-status" title={`Backend: ${API_STATUS_HINT[backendStatus]}`}>
                {API_STATUS_LABEL[backendStatus]}
              </span>
            </div>
            <div className="waveform-wrap">
              <Waveform
                level={audioLevel}
                active={listening === 'listening' || detections.length > 0}
                color={previewState.color}
                samplesRef={waveformSamplesRef}
              />
              {activeAlert && (
                <div className="alert-notice" role="alert">
                  <span>
                    {currentState.system
                      ? currentState.priority === 2
                        ? 'Urgent'
                        : 'Warning'
                      : currentState.name}
                  </span>
                  <strong>{activeAlert.soundNames.join(', ')} detected</strong>
                  <small>{Math.max(0, Math.ceil((activeAlert.expiresAt - clock) / 1000))}s</small>
                </div>
              )}
            </div>
            {listening !== 'listening' && !simple && (
              <button
                className="start-listening"
                type="button"
                onClick={() => void startMicrophone()}
              >
                Start listening<span>Requires microphone access</span>
              </button>
            )}
            <div className="demo-trigger-strip">
              <span>Simulate</span>
              {rules
                .filter((rule) => rule.kind === 'sound')
                .slice(0, 6)
                .map((rule) => (
                  <button key={rule.id} type="button" onClick={() => triggerRule(rule.id)}>
                    {rule.label}
                  </button>
                ))}
            </div>
          </section>
          <section className="subtitle-dock" aria-labelledby="subtitle-title">
            <div className="content-guide subtitle-content">
              <div className="subtitle-heading">
                <h1 id="subtitle-title">Sound subtitles</h1>
                {!simple && (
                  <span>
                    {detections.length ? `${detections.length} active` : 'Waiting for sound'}
                  </span>
                )}
              </div>
              <div className="subtitle-scroll" aria-live="polite">
                {detections.length ? (
                  <ul className="sound-list">
                    {detections.map((detection) => (
                      <li
                        key={detection.id}
                        className={`${detection.critical ? 'critical' : ''} ${detection.expiresAt - clock < 1400 ? 'expiring' : ''}`}
                      >
                        <span>{detection.label}</span>
                        <small>
                          {detection.live
                            ? `LIVE ${Math.round((detection.confidence ?? 0) * 100)}%`
                            : 'DEMO'}
                        </small>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="empty-state">Detected sounds will appear here as live subtitles.</p>
                )}
              </div>
            </div>
          </section>
        </main>
      ) : (
        <main className="configuration-screen">
          <header className="configuration-header">
            <div>
              <button type="button" onClick={() => setView('home')}>
                <Icon name="back" /> Home
              </button>
              <h1>Sound & effect configuration</h1>
              <p>Connect recognized inputs to one or more responses.</p>
            </div>
            <div className="save-actions">
              <span
                className={saveStatus === 'error' ? 'save-error' : ''}
                role="status"
                aria-live="polite"
              >
                {!settingsReady
                  ? 'Loading settings…'
                  : saveStatus === 'error'
                    ? 'Unable to save on this device'
                    : 'Saved on this device'}
              </span>
              <button type="button" onClick={resetSettings}>
                Reset defaults
              </button>
            </div>
          </header>
          <div className={`configuration-body ${priorityCollapsed ? 'priority-is-collapsed' : ''}`}>
            <StatePriorityPanel
              states={states}
              collapsed={priorityCollapsed}
              onToggle={() => setPriorityCollapsed((value) => !value)}
              onReorder={reorderStates}
              onEdit={openEditState}
              onDelete={deleteCustomState}
            />
            <div className="editor-area">
              {simple ? <SimpleEditor {...editorProps} /> : <NodeEditor {...editorProps} />}
            </div>
          </div>
          <footer className="integration-footer">
            <span>Detection status</span>
            <p>
              Tests on this screen are simulated. While listening, sounds marked “live detection”
              are recognised by the Alert backend ({API_STATUS_HINT[backendStatus]}). Phrases still
              need a speech-processing integration.
            </p>
          </footer>
        </main>
      )}

      {dialog === 'help' && (
        <Dialog
          title={introHelp ? 'Welcome to Alert' : 'How Alert works'}
          wide
          onClose={completeHelp}
        >
          <HelpContent />
          <div className="dialog-footer">
            <button className="primary-action" type="button" onClick={completeHelp}>
              {introHelp ? 'I understand' : 'Done'}
            </button>
          </div>
        </Dialog>
      )}

      {dialog === 'account' && (
        <Dialog
          title={account ? 'Profile' : accountMode === 'create' ? 'Create account' : 'Log in'}
          onClose={() => setDialog(null)}
        >
          {account ? (
            <div className="account-content">
              <span className="profile-mark">{account.name.charAt(0).toUpperCase()}</span>
              <h3>{account.name}</h3>
              <p>{account.email}</p>
              <div className="account-settings">
                <span>Introductory help</span>
                <button
                  type="button"
                  onClick={() => {
                    setDialog('help')
                    setIntroHelp(false)
                  }}
                >
                  View again
                </button>
              </div>
              <button
                type="button"
                className="secondary-action"
                onClick={() => {
                  setDialog(null)
                  if (auth) {
                    auth.logout()
                    return
                  }
                  localStorage.removeItem(ACCOUNT_KEY)
                  setLocalAccount(null)
                }}
              >
                Log out
              </button>
            </div>
          ) : auth ? (
            // Real sign-in: Auth0's hosted page handles passwords, Google and sign-up securely.
            <div className="account-form">
              <div className="provider-options" aria-label="Sign-in providers">
                <button type="button" disabled={auth.loading} onClick={() => auth.login('google')}>
                  <GoogleIcon />
                  <span>
                    <b>Continue with Google</b>
                    <small>Secure sign-in with Auth0</small>
                  </span>
                </button>
              </div>
              <div className="provider-divider">
                <span>or use email</span>
              </div>
              <button
                className="primary-action"
                type="button"
                disabled={auth.loading}
                onClick={() => auth.login(accountMode === 'create' ? 'create' : 'login')}
              >
                {auth.loading
                  ? 'Checking sign-in…'
                  : accountMode === 'create'
                    ? 'Create account with email'
                    : 'Continue with email'}
              </button>
              <button
                className="text-action"
                type="button"
                onClick={() => setAccountMode((mode) => (mode === 'login' ? 'create' : 'login'))}
              >
                {accountMode === 'login' ? 'Create an account' : 'Already have an account? Log in'}
              </button>
              <p className="form-disclaimer">
                You&apos;ll finish signing in on Auth0&apos;s secure page, then return here.
              </p>
            </div>
          ) : (
            <form className="account-form" onSubmit={accountSubmit}>
              <div className="provider-options" aria-label="External sign-in providers">
                <button type="button" disabled>
                  <GoogleIcon />
                  <span>
                    <b>Continue with Google</b>
                    <small>OAuth connection placeholder</small>
                  </span>
                </button>
                <button type="button" disabled>
                  <i className="auth0-mark">a0</i>
                  <span>
                    <b>Enterprise SSO</b>
                    <small>Auth0 support planned</small>
                  </span>
                </button>
              </div>
              <div className="provider-divider">
                <span>or use email</span>
              </div>
              {accountMode === 'create' && (
                <label>
                  Name
                  <input name="name" required placeholder="Your name" />
                </label>
              )}
              <label>
                Email
                <input type="email" name="email" required placeholder="name@example.com" />
              </label>
              <label>
                Password
                <input
                  type="password"
                  name="password"
                  required
                  minLength={6}
                  placeholder="At least 6 characters"
                />
              </label>
              <button className="primary-action" type="submit">
                {accountMode === 'create' ? 'Create account' : 'Continue'}
              </button>
              <button
                className="text-action"
                type="button"
                onClick={() => setAccountMode((mode) => (mode === 'login' ? 'create' : 'login'))}
              >
                {accountMode === 'login' ? 'Create an account' : 'Already have an account? Log in'}
              </button>
              <p className="form-disclaimer">
                Local preview account (no real sign-in). Add the Auth0 settings to .env.local to
                enable secure login.
              </p>
            </form>
          )}
        </Dialog>
      )}

      {dialog === 'custom-state' && (
        <Dialog
          title={editingStateId ? 'Edit visual state' : 'Create visual state'}
          onClose={() => {
            setEditingStateId(null)
            setDialog(null)
          }}
        >
          <form
            className="custom-state-form"
            onSubmit={(event) => {
              event.preventDefault()
              addCustomState()
            }}
          >
            <label>
              State name
              <input
                value={stateDraft.name}
                maxLength={60}
                onChange={(event) => setStateDraft({ ...stateDraft, name: event.target.value })}
                placeholder="Information"
                required
              />
            </label>
            <label>
              Interface colour
              <div className="color-field">
                <input
                  type="color"
                  value={stateDraft.color}
                  onChange={(event) => setStateDraft({ ...stateDraft, color: event.target.value })}
                />
                <span>{stateDraft.color.toUpperCase()}</span>
              </div>
            </label>
            <label>
              Display duration
              <div className="duration-field">
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={stateDraft.duration}
                  onChange={(event) =>
                    setStateDraft({ ...stateDraft, duration: Number(event.target.value) })
                  }
                />
                <span>{stateDraft.duration}s</span>
              </div>
            </label>
            <div className="state-preview" style={{ borderColor: stateDraft.color }}>
              <i style={{ background: stateDraft.color }} />
              <div>
                <b>{stateDraft.name || 'Custom state'}</b>
                <span>Global interface preview</span>
              </div>
            </div>
            <p>Set this state&apos;s precedence by dragging it in the priority list.</p>
            <button className="primary-action" type="submit">
              {editingStateId ? 'Update state' : 'Create state'}
            </button>
          </form>
        </Dialog>
      )}

      {showPhraseForm && (
        <div className="inline-overlay" onMouseDown={() => setShowPhraseForm(false)}>
          <form
            className="phrase-popover"
            onSubmit={(event) => {
              event.preventDefault()
              addPhrase()
            }}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div>
              <h3>Add recognized phrase</h3>
              <button type="button" onClick={() => setShowPhraseForm(false)}>
                <Icon name="close" />
              </button>
            </div>
            <p>
              Defines an incoming spoken word or phrase. Recognition requires a speech-processing
              integration.
            </p>
            <label>
              Word or phrase
              <input
                autoFocus
                value={phraseDraft}
                maxLength={80}
                onChange={(event) => setPhraseDraft(event.target.value)}
                placeholder="Watch out"
              />
            </label>
            <button className="primary-action" type="submit">
              Add phrase
            </button>
          </form>
        </div>
      )}
    </div>
  )
}

/** Connects Auth0's login state to the app and to API calls. Rendered only when Auth0 is configured. */
function Auth0Bridge() {
  const { isLoading, isAuthenticated, user, loginWithRedirect, logout, getAccessTokenSilently } =
    useAuth0()
  // Layout effect: runs before the app's own effects, so they already see the right token getter.
  useLayoutEffect(() => {
    setAccessTokenGetter(async () =>
      isAuthenticated ? ((await getAccessTokenSilently()) ?? null) : null,
    )
  }, [isAuthenticated, getAccessTokenSilently])
  const auth = useMemo<RealAuth>(
    () => ({
      loading: isLoading,
      user:
        isAuthenticated && user?.sub
          ? {
              id: user.sub,
              name: user.name || user.nickname || user.email || 'Account',
              email: user.email || '',
            }
          : null,
      login: (mode) =>
        void loginWithRedirect({
          authorizationParams:
            mode === 'create'
              ? { screen_hint: 'signup' }
              : mode === 'google'
                ? { connection: 'google-oauth2' }
                : {},
        }),
      // Trailing slash matches the URLs Auth0's setup screen registers (it compares exactly).
      logout: () => void logout({ logoutParams: { returnTo: `${window.location.origin}/` } }),
    }),
    [isAuthenticated, isLoading, loginWithRedirect, logout, user],
  )
  return <AlertApp auth={auth} />
}

export default function AlertAppRoot() {
  // Auth0 needs `window`, so it starts after the first client render.
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  if (!AUTH0_ENABLED) return <AlertApp auth={null} />
  if (!mounted) return null
  return (
    <Auth0Provider
      domain={AUTH0_CONFIG.domain}
      clientId={AUTH0_CONFIG.clientId}
      authorizationParams={{
        redirect_uri: `${window.location.origin}/`,
        audience: AUTH0_CONFIG.audience, // must equal the backend's AUTH0_AUDIENCE
        scope: 'openid profile email offline_access',
      }}
      // Keeps users signed in across reloads (browsers block Auth0's cookie-based silent login).
      cacheLocation="localstorage"
      useRefreshTokens
      useRefreshTokensFallback
    >
      <Auth0Bridge />
    </Auth0Provider>
  )
}
