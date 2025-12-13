# MELD Backend (Rebuilt)

This is a clean rebuild of the MELD backend focused on **stability first**:

- OpenAI is the primary engine (2-pass: composer ➜ answer)
- Optional **xAI** "scout" used in Multi / Ultimate run modes
- Single `runMode` selector supported via `/health.runModes`
- Redis-backed sessions (optional; falls back safely if missing)
- Token auth for `/chat` and `/repo` using `X-MELD-TOKEN` (preferred) or `Authorization: Bearer ...`

## Endpoints

- `GET /health` (no auth)
- `POST /chat` (auth)
- `POST /repo` (auth)

## Render setup

**Build:** `npm install`  
**Start:** `npm start`

### Environment variables

Required:
- `OPENAI_API_KEY`
- `MELD_BACKEND_TOKEN`

Recommended:
- `REDIS_URL`

Optional:
- `XAI_API_KEY` to enable xAI scout + Multi/Ultimate runModes

## Auth

Send one of:
- `X-MELD-TOKEN: <MELD_BACKEND_TOKEN>` (recommended)
- `Authorization: Bearer <MELD_BACKEND_TOKEN>`

