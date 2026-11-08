# Alert

**See important sounds.**

Alert listens for environmental sounds and turns detections—such as alarms,
sirens, doorbells, and a baby crying—into visual alerts. Sign in with Auth0 to
use authenticated detection and account features.

> **Safety notice:** Alert is an assistive tool, not a certified alarm or
> life-safety system. It may miss sounds or report false detections. Do not use
> it as a replacement for approved smoke alarms, strobe alarms, or other
> emergency equipment.

## How it works

- The browser captures short audio chunks only while sound monitoring is active.
- The FastAPI service processes each chunk and returns a sound category,
  confidence, and detection status.
- The frontend turns detections into the user's configured visual alerts.
- Auth0 provides sign-in. The backend verifies access tokens and records
  account metadata in a Databricks SQL table.

Audio recordings and sound detections are not stored by the backend. Signed-in
users' Alert preferences sync with their account through the backend database.
No Alert preferences or local-preview account data are written to browser
storage. Without an authenticated account, settings use defaults and reset when
the page is refreshed. If account sync is unavailable, the settings screen
reports the error and does not claim changes were saved.

## Project layout

| Path                  | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| `src/app/(frontend)/` | Next.js app and browser UI                         |
| `src/app/(payload)/`  | Payload CMS admin and API routes                   |
| `src/collections/`    | Payload collections                                |
| `src/migrations/`     | Versioned PostgreSQL schema migrations             |
| `backend/`            | FastAPI API, audio processing, auth, and inference |
| `vercel.json`         | Vercel app/backend services and public API routing |

The Payload CMS is separate from the app's account store. Payload uses
PostgreSQL and Vercel Blob for CMS media; app user records use Databricks SQL.
The existing Figma-hosted CMS content and media have not been migrated.

## Run locally

### Prerequisites

- Node.js 24 and pnpm 10
- Python 3.12
- Auth0 credentials for login and authenticated API requests
- Databricks SQL warehouse credentials to record users
- Optional for the CMS: PostgreSQL and a Vercel Blob token

### 1. Install frontend dependencies

From the repository root:

```powershell
corepack enable
pnpm install
Copy-Item .env.example .env.local
```

Set the public Auth0 values in `.env.local`:

```dotenv
NEXT_PUBLIC_AUTH0_DOMAIN=your-tenant.us.auth0.com
NEXT_PUBLIC_AUTH0_CLIENT_ID=your-auth0-application-client-id
NEXT_PUBLIC_AUTH0_AUDIENCE=https://your-auth0-api-identifier
```

The audience must exactly match the identifier of an API configured in the
Auth0 tenant. Add `http://localhost:3000/` to the SPA's Allowed Callback URLs,
Allowed Logout URLs, and Allowed Web Origins.

### 2. Configure and run the backend

In a second terminal:

```powershell
Set-Location backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Fill in the Auth0 and Databricks settings in `backend/.env`. The same Auth0 API
audience must be used by the frontend and backend. `requirements-dev.txt`
includes the optional TensorFlow dependencies for local YAMNet inference.

Start the API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

For macOS/Linux, activate the virtual environment with
`source .venv/bin/activate`; use `cp` instead of `Copy-Item`.

### 3. Run the frontend

From the repository root in another terminal:

```powershell
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The API health check and
interactive documentation are available at [http://localhost:8000/health](http://localhost:8000/health)
and [http://localhost:8000/docs](http://localhost:8000/docs).

If Auth0 is not configured in the frontend, the app uses its local preview
account. Authenticated API features require valid Auth0 configuration.

For backend setup, Auth0 and Databricks configuration, API details, and
troubleshooting, see the [backend guide](backend/README.md).

## Deploy to Vercel

The repository configures two services:

- **`app`** — Next.js frontend and Payload CMS.
- **`backend`** — FastAPI sound detection and user account API.

The backend is public only through the API paths and health check listed in
[vercel.json](vercel.json); all other requests go to the app. No service binding
is used.

### App service environment

Configure these variables for the Vercel `app` service:

| Variable                      | Purpose                                   |
| ----------------------------- | ----------------------------------------- |
| `NEXT_PUBLIC_AUTH0_DOMAIN`    | Public Auth0 tenant domain                |
| `NEXT_PUBLIC_AUTH0_CLIENT_ID` | Public Auth0 SPA client ID                |
| `NEXT_PUBLIC_AUTH0_AUDIENCE`  | Auth0 API identifier used by the backend  |
| `DATABASE_URL`                | PostgreSQL connection URL for Payload     |
| `PAYLOAD_SECRET`              | Private Payload signing/encryption secret |
| `BLOB_READ_WRITE_TOKEN`       | Vercel Blob storage token                 |
| `NEXT_PUBLIC_SITE_URL`        | Optional canonical site URL               |

Use the exact deployed URL in the Auth0 SPA callback, logout, and web-origin
allowlists. Keep secrets such as `PAYLOAD_SECRET` private; only the Auth0
`NEXT_PUBLIC_*` identifiers are intended for the browser.

### Backend service environment

Configure the backend variables in `backend/.env.example` in the Vercel
`backend` service settings. Production requires Auth0, Databricks SQL, and
Databricks Model Serving:

- `ENVIRONMENT=production`
- `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`
- `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_WAREHOUSE_ID`
- `INFERENCE_PROVIDER=databricks`
- `DATABRICKS_ENDPOINT_NAME`
- `DATABRICKS_FALLBACK_TO_LOCAL=false`

The Vercel backend dependency set intentionally excludes TensorFlow to fit the
Python function bundle limit. Consequently, inference on Vercel requires a
configured Databricks Model Serving endpoint. See the [backend deployment
guide](backend/README.md#vercel) for all production settings.

### Payload database migrations

The initial PostgreSQL migration is committed in `src/migrations/`. Configure
`DATABASE_URL` and `PAYLOAD_SECRET`, then apply migrations to the target database
before deploying:

```powershell
pnpm migrate
```

For future schema changes, generate and commit a migration with
`pnpm payload migrate:create <name>`, then run `pnpm migrate` against the
appropriate database before deploying. Use a separate database for Vercel
Preview deployments to avoid applying preview migrations to production.

The application build (`pnpm build`) does not connect to the database. Payload
CMS runtime features still require the app service's database, secret, and Blob
environment variables.

## Verification

Run backend tests from the repository root:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests
```

Build and type-check the frontend:

```powershell
pnpm build
pnpm exec next typegen
pnpm exec tsc --noEmit
```

## Privacy and limitations

- Microphone access is requested by the browser and is used for live detection.
- The backend does not persist submitted audio or detection history.
- Databricks stores account metadata used to record logins.
- Sound classification can be inaccurate and should not be relied on for
  emergency response.
- Local inference and Vercel inference use different providers: local YAMNet
  versus Databricks Model Serving.

## License

No license is currently specified. All rights reserved unless the repository
owner publishes a license.
