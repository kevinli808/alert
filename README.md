# Alert

## Local development

Install dependencies with `pnpm install`, then run the frontend with `pnpm dev`.
The FastAPI backend lives in `backend/`; its setup and local inference
dependencies are documented in [backend/README.md](backend/README.md).
Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `.env.local` for local
frontend development.

## Vercel services

The root `vercel.json` deploys the Next.js app and FastAPI backend as separate
services. The app forwards its API requests through the shared deployment
domain; the backend serves the sound-detection API paths and `/health`.

The Payload CMS uses standalone infrastructure, not Figma-managed services.
Provision PostgreSQL (use a pooled connection URL) and a Vercel Blob store.
Configure these variables on the **app** service in Vercel:

- `DATABASE_URL` — PostgreSQL connection string for Payload.
- `PAYLOAD_SECRET` — long, random Payload encryption/auth secret.
- `BLOB_READ_WRITE_TOKEN` — token supplied by Vercel Blob.
- `NEXT_PUBLIC_SITE_URL` — optional canonical public site URL.

The initial Payload schema migration is included in `src/migrations/`. The Vercel
build runs pending migrations before building the app, so `DATABASE_URL` and
`PAYLOAD_SECRET` must be available to the build environment. Keep future
migration files in source control; do not run `migrate:create` as part of each
deployment. Use a separate database for Vercel Preview deployments so preview
migrations cannot alter production data.

Configure the frontend's public Auth0 identifiers (`NEXT_PUBLIC_AUTH0_DOMAIN`,
`NEXT_PUBLIC_AUTH0_CLIENT_ID`, and `NEXT_PUBLIC_AUTH0_AUDIENCE`) on the app
service. Set the audience to the exact Auth0 API identifier (without a trailing
slash); the frontend also removes accidental trailing slashes.

The **backend** service intentionally excludes TensorFlow to keep its Vercel
function bundle within the 500 MB Python function limit
([Vercel documentation](https://vercel.com/docs/functions/limitations#bundle-size-limits)).
It uses remote Databricks inference.
Create and configure the Databricks model-serving endpoint, then set
`ENVIRONMENT=production`, `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `SUPABASE_URL`,
`SUPABASE_SERVICE_KEY`, `INFERENCE_PROVIDER=databricks`, `DATABRICKS_HOST`,
`DATABRICKS_TOKEN`, and `DATABRICKS_ENDPOINT_NAME` on the backend service.
Keep `DATABRICKS_FALLBACK_TO_LOCAL=false`; local TensorFlow inference cannot run
in the Vercel function bundle.
Never commit secret values. Local YAMNet inference remains available through
`backend/requirements-local.txt`.

The existing Figma-hosted Payload database and media are not migrated by this
configuration change. Export and import that content separately before
switching users to the new CMS.

Create the first CMS administrator through Payload's first-user setup after the
deployment has been configured. Add the deployed URL to the Auth0 SPA callback,
logout, and web-origin allowlists.
