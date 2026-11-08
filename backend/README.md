# Alert API

The Python service behind [Alert](../README.md). It verifies Auth0 access tokens,
classifies short audio uploads, and records account metadata in Databricks SQL.
The browser renders detections as visual alerts.

```
Browser ── Auth0 token + short audio chunk ──▶ FastAPI
                                                ├─ Auth0 JWT verification
                                                ├─ Audio decode and preprocessing
                                                ├─ YAMNet (local) or Databricks Model Serving
                                                └─ Databricks SQL (account metadata only)
```

Audio and detection results are processed in memory and are not persisted.
Only account metadata—Auth0 user ID, email, username, and login timestamps—is
stored in the Databricks users table. Alert preferences remain in the browser.

## Contents

1. [Quick start](#1-quick-start)
2. [Auth0 setup](#2-auth0-setup)
3. [Databricks users database](#3-databricks-users-database)
4. [Sound classification](#4-sound-classification-yamnet)
5. [Databricks inference](#5-databricks-inference)
6. [API reference](#6-api-reference)
7. [Testing](#7-testing)
8. [Deployment](#8-deployment)
9. [Security notes](#9-security-notes)
10. [Known limitations](#10-known-limitations)

Frontend setup is covered in the [project README](../README.md). For Auth0,
Databricks, API, and backend operations, use this guide.

---

## 1. Quick start

Requires Python 3.10–3.12 for local YAMNet inference (TensorFlow does not
support Python 3.13). Python 3.12 is configured for Vercel; local development
has been tested on Windows with Python 3.11.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env               # then fill in Auth0 + Databricks values
uvicorn app.main:app --reload
```

- Swagger UI: http://localhost:8000/docs. Click **Authorize** and paste an Auth0 access token.
- Health: http://localhost:8000/health. With local YAMNet, `model.status` goes from `loading` to
  `ready` in about 5 to 30 s. The first run downloads YAMNet (~15 MB).
- With Auth0 or the Databricks users database unconfigured, the server still starts. The affected endpoints return
  `503 service_not_configured`. In `ENVIRONMENT=production` the server refuses to start instead.

Optional: install **ffmpeg** to accept WebM/MP4 from `MediaRecorder`. WAV, FLAC, OGG and MP3 work
without it.

## 2. Auth0 setup

Create or use an Auth0 tenant with an API and a Single Page Application.

1. **Create the API.** Go to Applications → APIs → _Create API_.
   - Name: `Alert API`
   - Identifier: `https://api.deaf-accessibility.app` (any URI). This is `AUTH0_AUDIENCE`.
   - Signing algorithm: **RS256**
2. **Create the SPA app** for the web frontend. Go to Applications → _Create Application_ → _Single Page Web Applications_.
   - Allowed Callback URLs, Logout URLs, and Web Origins: `http://localhost:3000/` (plus your deployed URL).
   - Set the **Domain** and **Client ID** as `NEXT_PUBLIC_AUTH0_DOMAIN` and `NEXT_PUBLIC_AUTH0_CLIENT_ID` in the frontend `.env.local`. Set the domain as `AUTH0_DOMAIN` in `backend/.env`.
   - Set `NEXT_PUBLIC_AUTH0_AUDIENCE` in the frontend and `AUTH0_AUDIENCE` in the backend to the same API identifier.
   - The SPA has **no client secret**. Never put a secret or Management API credential in frontend code.
3. **Login methods.** Authentication → Database → `Username-Password-Authentication` (email/password,
   with sign-ups enabled). Authentication → Social → **Google** → enable it for the SPA app.
   For production, add your own Google OAuth client ID. The Auth0 dev keys are for testing only.
4. **Add email/name to access tokens.** This is recommended; otherwise the backend calls `/userinfo`
   once per user. Go to Actions → Library → _Create Action_ → _Login / Post Login_:
   ```js
   exports.onExecutePostLogin = async (event, api) => {
     const ns = 'https://deaf-accessibility.app' // = AUTH0_CLAIM_NAMESPACE
     api.accessToken.setCustomClaim(`${ns}/email`, event.user.email)
     api.accessToken.setCustomClaim(`${ns}/name`, event.user.name || event.user.nickname)
   }
   ```
   Deploy it, then add it to the **Post Login** trigger flow.

The backend verifies tokens locally with the cached JWKS
(`https://<domain>/.well-known/jwks.json`). It checks the signature, the `RS256` algorithm, `iss`,
`aud`, `exp`/`iat` with 30 s leeway, and `sub`. Passwords and sessions are handled entirely by Auth0.
Session management means the SPA SDK refreshes tokens and logout clears the Auth0 session.

## 3. Databricks users database

The database records **users only**: one row per Auth0 account, written when the frontend calls
`GET /api/users/me` after login. Alert settings stay in browser storage per profile on each device,
and detections are not stored.

| Column              | Meaning                                                        |
| ------------------- | -------------------------------------------------------------- |
| `id`                | Auth0 user id (the access token's `sub`)                       |
| `email`, `username` | from the token claims (section 2, step 4) or Auth0 `/userinfo` |
| `created_at`        | first login                                                    |
| `last_login_at`     | most recent login                                              |

Create or use a Databricks workspace with a SQL warehouse.

1. **SQL warehouse.** SQL Warehouses → pick one (or _Create_ a small serverless one) → **Connection
   details**. Copy the **Server hostname** into `DATABRICKS_HOST` as `https://<hostname>`, and the id
   at the end of **HTTP path** (`/sql/1.0/warehouses/<id>`) into `DATABRICKS_WAREHOUSE_ID`.
2. **Token.** Your avatar → Settings → Developer → Access tokens → _Generate new token_. Put it in
   `DATABRICKS_TOKEN`. It stays in `backend/.env` only, never in frontend code.
3. **Table.** Open SQL Editor, choose the warehouse, and run [sql/databricks_users.sql](sql/databricks_users.sql).
   It creates the `workspace.alert` schema and the `users` table. The backend also creates the table on
   first use, so only the schema has to exist. For a different location set `DATABRICKS_USERS_TABLE`
   (`catalog.schema.table`).
4. Restart the backend. `/health` shows `"database_configured": true`.

**How access control works.** The browser never talks to Databricks. FastAPI verifies the Auth0 token and
writes the row keyed by that token's `sub`; the user id is never taken from the request. All values
are sent as SQL statement parameters, and the table name is validated as an identifier at startup.

A stopped serverless warehouse takes a few seconds to wake up, and a classic one can take minutes.
Requests wait up to `DATABASE_TIMEOUT_SECONDS` (default 30) and then return `503 database_unavailable`;
login itself still works, and the row is written on the next login.

No audio is ever stored.

## 4. Sound classification (YAMNet)

- **Input.** YAMNet takes mono float32 audio at 16 kHz in [-1, 1]. The audio processor decodes,
  down-mixes, resamples with soxr, and scales down only when the audio clips. It never boosts quiet
  audio, because amplifying noise causes false alarms. Chunks below an RMS floor count as silence
  and skip inference.
- **Output.** YAMNet gives 521 AudioSet scores per 0.96 s frame (0.48 s hop). Scores are combined
  per clip with the **max** over frames (configurable). A category's score is the highest score
  among its YAMNet classes.
- **Mapping** ([app/services/sound_categories.py](app/services/sound_categories.py)). Every label was
  checked against the real `yamnet_class_map.csv` (521 classes, 0 missing). The loader re-checks at
  startup and logs any missing labels.

| Category        | YAMNet classes                                                                                                        | Priority |
| --------------- | --------------------------------------------------------------------------------------------------------------------- | -------- |
| fire_alarm      | Fire alarm                                                                                                            | critical |
| smoke_detector  | Smoke detector, smoke alarm                                                                                           | critical |
| emergency_siren | Siren, Civil defense siren, Police car (siren), Ambulance (siren), Fire engine, fire truck (siren), Emergency vehicle | critical |
| glass_breaking  | Shatter (there is **no** "glass breaking" class)                                                                      | high     |
| baby_crying     | Baby cry, infant cry                                                                                                  | high     |
| car_alarm       | Car alarm                                                                                                             | high     |
| alarm           | Alarm, Buzzer                                                                                                         | high     |
| doorbell        | Doorbell, Ding-dong                                                                                                   | normal   |
| door_knock      | Knock                                                                                                                 | normal   |
| phone_ringing   | Telephone bell ringing, Ringtone, Telephone                                                                           | normal   |
| alarm_clock     | Alarm clock                                                                                                           | normal   |
| dog_barking     | Bark                                                                                                                  | normal   |

- **Decision.** A category passes if its score is at least the threshold (`CONFIDENCE_THRESHOLD=0.6`
  by default, with per-category overrides). If several pass, the most urgent priority wins.
  Otherwise the response is `detected: false, sound_type: "unknown"`.
- **Cooldown and event grouping.** Repeated detections of the same sound for the same user, each
  within `DETECTION_COOLDOWN_SECONDS` of the last, form one event. Only the first chunk has
  `is_new_event: true`.
- **Failures.** If the model fails, the API returns an error (`503 model_unavailable` or
  `502 inference_failed`). It never returns a made-up detection.

**Observed on real YAMNet (synthetic signals, not real recordings):**

- A synthetic 3.1 kHz T3 smoke-alarm beep scored `alarm` (Buzzer 0.82) with `smoke_detector` at
  0.67. Fire alarm, smoke detector and generic alarm overlap, so treat all three as urgent in the UI.
- A pure sine "siren" sweep scored as _Synthesizer_, not _Siren_. Synthetic tests do not show real
  accuracy.
- Latency for a warm 2 s chunk was **~11–16 ms** (preprocess + inference, CPU).

**Calibrate before trusting the default threshold of 0.6.** Collect real clips (your own recordings, or Freesound / ESC-50
/ AudioSet clips with compatible licences) in `eval_data/<category>/*.wav`, plus
`eval_data/background/` for TV, speech, kitchen noise and similar. Then run:

```bash
python -m scripts.evaluate_model eval_data            # add --mlflow to log to Databricks MLflow
```

Pick thresholds that favour **recall** for critical sounds, since a missed fire alarm is worse
than a false flash. Set them with `CATEGORY_THRESHOLDS`.

## 5. Databricks inference

Local YAMNet is the default for local development. Databricks Model Serving provides hosted
inference and is required on Vercel because the Vercel backend dependency set excludes TensorFlow.
No endpoint is included or created automatically.

1. Experiments: run `scripts/evaluate_model.py --mlflow` with `MLFLOW_TRACKING_URI=databricks`
   (plus `DATABRICKS_HOST` and `DATABRICKS_TOKEN`) to track threshold experiments.
2. Package the model: in a Databricks notebook, run
   [databricks/yamnet_pyfunc.py](databricks/yamnet_pyfunc.py). It logs YAMNet plus the _same_
   `sound_categories.py` mapping as an MLflow pyfunc and registers it.
3. Serve it: go to **Serving → Create serving endpoint**, pick the registered model, choose CPU
   Small, and turn on scale-to-zero.
4. Configure the backend:
   ```
   INFERENCE_PROVIDER=databricks
   DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
   DATABRICKS_TOKEN=<PAT or service principal token>
   DATABRICKS_ENDPOINT_NAME=<endpoint name>
   DATABRICKS_FALLBACK_TO_LOCAL=true     # use local YAMNet if the endpoint errors or times out
   ```
5. Compare: `inference.provider` and `inference.inference_ms` in each response show which path ran
   and how long it took. Remote inference adds a network round trip and may have cold starts when
   the endpoint scales to zero.

The wire format (base64 int16 PCM, about 85 KB per 2 s) is documented at the top of
[app/services/databricks_service.py](app/services/databricks_service.py).

## 6. API reference

In production, API routes require authorization except for `/api/sound-types`. Token-less detection
is allowed only in local development when Auth0 is not configured.

Every `/api/*` route except `/api/sound-types` needs `Authorization: Bearer <Auth0 access token>`.
Errors always use the shape `{"error": {"code", "message", "details?"}}`. Full interactive docs are
at `/docs`.

| Method | Path                | Purpose                                                                              | Notes                                    |
| ------ | ------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------- |
| GET    | `/health`           | Model, auth and DB status                                                            | public                                   |
| GET    | `/api/sound-types`  | Detectable categories and thresholds                                                 | public                                   |
| GET    | `/api/users/me`     | Record the logged-in user (row created on first call, `last_login_at` updated after) | called by the frontend after login       |
| POST   | `/api/detect-sound` | Classify one chunk                                                                   | multipart field `audio`; 0.5–10 s; ≤2 MB |

Status codes:

- `401` missing or invalid token
- `413` file too large
- `415` not audio
- `422` validation error or invalid audio
- `429` rate limited (`Retry-After`)
- `502` inference failed
- `503` model loading or a service is unavailable

### Examples

```bash
TOKEN="<paste access token>"   # Frontend: await getAccessTokenSilently(); or Auth0 → APIs → Test tab
API=http://localhost:8000

curl $API/api/users/me -H "Authorization: Bearer $TOKEN"

curl -X POST $API/api/detect-sound -H "Authorization: Bearer $TOKEN" -F "audio=@doorbell.wav;type=audio/wav"
```

## 7. Testing

```bash
pytest                              # no network or credentials needed
RUN_MODEL_TESTS=1 pytest -s tests/test_yamnet_integration.py   # real YAMNet + latency (PowerShell: $env:RUN_MODEL_TESTS=1)
```

The tests cover:

- JWT validation: expiry, audience, issuer, unknown key, forged signature, HS256 algorithm confusion
- per-user data isolation, and rejection of a `user_id` in the request body
- token-less detection only in local development without Auth0, never touching the database
- audio decoding: stereo/44.1 kHz → mono/16 kHz, corrupt, empty, too long/short, silence
- threshold, priority and cooldown logic, and invalid model output
- user recording keyed by the token, never by the request
- the rate limiter and CORS
- the Databricks SQL and Model Serving HTTP contracts (mocked), table-name validation, timeouts, and the fallback provider

Test doubles live only in `tests/`.

**End-to-end with real services.** Once your `.env` is configured:

1. log in through the web app
2. confirm a row appears: `SELECT * FROM workspace.alert.users ORDER BY last_login_at DESC`
3. reload the page and confirm `last_login_at` moves forward on the same row
4. sign in as a second user and confirm a second row appears

## 8. Deployment

For a single-container deployment, build and run the Docker image below. For the configured
two-service Vercel deployment, see [Vercel](#vercel).

- **Host:** any container host that gives at least **2 GB RAM** works, e.g. Render, Railway,
  Fly.io or Google Cloud Run. TensorFlow and YAMNet use about 700 MB to 1 GB.
- **Build and run:** From `backend/`, run `docker build -t alert-api .` and then
  `docker run --env-file .env -p 8000:8000 alert-api`. The YAMNet weights are baked into the
  image, so starts do not need to download the model.
- **Environment variables:** set the same variables as `.env`, plus `ENVIRONMENT=production` and
  `CORS_ORIGINS=https://your-frontend.app`. Use the host's secret manager, never a committed file.
- **One worker:** run a single worker or instance. The rate limiter and event cooldown are in
  memory. Moving them to Redis is the upgrade path if you ever need several instances.
- **Auth0:** add the deployed frontend URL to the Auth0 SPA callback, logout and web-origin lists.
- **Frontend:** set `NEXT_PUBLIC_API_URL` to the backend URL when it is on a different origin.
- **HTTPS:** required, since browsers only allow microphone access on HTTPS or `localhost`.

### Vercel

The Vercel backend service installs `requirements.txt`, which deliberately excludes TensorFlow;
TensorFlow is too large for a standard Vercel Python function bundle. The Vercel deployment
therefore defaults to Databricks inference and disables local-model fallback. Create a working
Databricks Model Serving endpoint and configure `ENVIRONMENT=production`, `AUTH0_DOMAIN`,
`AUTH0_AUDIENCE`, `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_WAREHOUSE_ID`,
`INFERENCE_PROVIDER=databricks`, and `DATABRICKS_ENDPOINT_NAME` on the backend service.
Missing required production settings cause startup validation to fail rather than running with
insecure or nonfunctional defaults. Keep `DATABRICKS_FALLBACK_TO_LOCAL=false`.

Local development installs `requirements-dev.txt`, which also installs
`requirements-local.txt` for local YAMNet inference. Do not add TensorFlow back to
`requirements.txt`, or Vercel deployments may exceed the Python function bundle limit.

## 9. Security notes

- The server holds every secret: the Databricks token, and any Auth0
  Management credentials, which this backend does not need at all.
- The logs never contain tokens, audio bytes, or secret values (settings use `SecretStr`).
- Uploads are capped by both `Content-Length` and the bytes actually read. They are decoded in
  memory (or as a temp file for ffmpeg, deleted right away) and never stored.
- Rate limiting is per user: 120 detections and 60 other requests per minute by default.
- CORS allows only listed origins with no credentials or cookies, and the server refuses to start
  with a wildcard in production.
- Every response sets `nosniff`, `X-Frame-Options: DENY` and `Referrer-Policy: no-referrer`, and
  `/api` responses also set `Cache-Control: no-store`.

## 10. Known limitations

- **Not a safety device.** Browsers suspend background tabs, and mobile operating systems cut the
  microphone when the screen locks. Never present this app as a replacement for certified
  strobe or vibrating alarms.
- YAMNet was trained on YouTube audio. Real-world accuracy for your devices is unknown until you
  calibrate it (section 4). Fire alarm, smoke detector and generic alarm are often confused.
- HTTP chunk upload adds a small per-request overhead. If you need lower latency later, a
  WebSocket endpoint can reuse the same `DetectionService` unchanged.
