# Alert

Alert is an accessibility-focused sound awareness app for deaf and hard-of-hearing
people. It listens for selected environmental sounds and turns detections into
customizable visual alerts. The project was built for **Hack Camp 2026**.

[View our Devpost submission](https://devpost.com/software/alert-75ij9a)

## Features

- Detects sounds such as alarms, doorbells, sirens, and dog barking.
- Maps sounds to configurable visual states and effects through an interactive
  node-tree interface.
- Offers a simplified interface for people who prefer fewer controls.
- Supports sign-in and account-based settings sync.

## How it works

1. When monitoring is enabled, the browser captures overlapping two-second
   audio segments, advancing every second.
2. The frontend encodes each segment and sends it to the FastAPI service. Audio
   is processed in memory and is not retained.
3. The backend filters silence, prepares the audio, and classifies it with
   Google's YAMNet model. Inference can run locally or through Databricks Model
   Serving.
4. The backend applies confidence thresholds, including per-sound overrides,
   and returns the detection and confidence to the frontend.
5. The frontend matches the detected sound to the user's configured rules and
   displays the corresponding visual alert.

Auth0 provides sign-in and protects account endpoints. When configured,
Databricks SQL stores account metadata and each user's Alert settings; it does
not store audio or detection history.

## Technology

- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **API:** Python, FastAPI
- **Sound classification:** YAMNet, with optional Databricks Model Serving
- **Accounts and settings:** Auth0 and Databricks SQL
- **CMS:** Payload
- **Deployment:** Vercel

## Run locally

### Prerequisites

- Node.js 24 and pnpm 10
- Python 3.12

### 1. Start the backend

In a terminal:

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

For macOS/Linux, create the environment with `python3.12 -m venv .venv`, activate
it with `source .venv/bin/activate`, and copy the configuration with
`cp .env.example .env`.

The backend uses local YAMNet inference by default. Local development can run
detection without Auth0; authenticated accounts and settings sync require
matching Auth0 configuration and Databricks credentials. See
[`backend/.env.example`](backend/.env.example) for available settings.

### 2. Start the frontend

In a second terminal at the repository root:

```powershell
pnpm install
Copy-Item .env.example .env.local
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `.env.local`, then start the
app:

```powershell
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). Payload CMS features also
require the database and storage settings listed in `.env.example`.

## Tests

Run the backend test suite from the repository root:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests
```

## Safety and privacy

Alert is a proof of concept, not a certified alarm or life-safety system.
Detection may be inaccurate; do not rely on Alert as a replacement for approved
emergency equipment. Audio is processed in memory and not stored by the
backend. Account metadata and settings are stored only when the related
services are configured.
