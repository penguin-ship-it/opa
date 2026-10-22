---
name: run-opa
description: Run, start, launch, seed, smoke-test, or screenshot the OPA / PayGuard app (FastAPI backend :8001 + React/Vite frontend :5174). Use to bring the stack up from a clean checkout, log in as a demo user, hit the API with curl-style checks, drive the worklist and case-detail pages in a headless browser, or confirm a change works in the running app (not just pytest).
---

# Run OPA / PayGuard

FastAPI backend (`server/`, :8001) + React/Vite SPA (`client/`, :5174) over SQLite.
**Agent path = `smoke.py`**: an API pass (health → login → cases → case detail) and a
headless-browser pass (log in through the form → worklist → click first case →
screenshots). All paths below are relative to the repo root (`opa/`).

## Prerequisites (macOS, verified)

- Shared venv — **not** inside the repo: `/Users/issam/claude/.venv`
  (has uvicorn, alembic, pytest, playwright). CLAUDE.md's `/Users/issamzeinoun/...`
  path does not exist on this machine. Nothing is on PATH; call `.venv/bin/*` directly.
- Node/npm (Homebrew) and `/Applications/Google Chrome.app` (the driver's browser).

Every block runs from the repo root, in one shell (the exports carry over).

## Setup (cold start)

Optional — use a throwaway DB instead of `server/opa.db`. Run this first; the
alembic, seed and uvicorn commands below all honour these variables:

```bash
mkdir -p "$TMPDIR/opa-cold" && export DATABASE_URL="sqlite+aiosqlite:///$TMPDIR/opa-cold/opa.db" DB_PATH="$TMPDIR/opa-cold/opa.db" ML_MODELS_DIR="$TMPDIR/opa-cold/ml_models"
```

```bash
(cd client && npm install)
(cd server && /Users/issam/claude/.venv/bin/alembic upgrade head)
(cd server && /Users/issam/claude/.venv/bin/python -m seed.seed_all)
```

Startup migrations are disabled in `main.py`, so **`alembic upgrade head` is mandatory** —
the app boots on an empty DB without it. Seed takes ~9 s (incl. both ML models) and
ends with `Cases 112`.

## Launch

```bash
(cd server && nohup /Users/issam/claude/.venv/bin/uvicorn app.main:app --port 8001 > "$TMPDIR/opa-backend.log" 2>&1 &)
(cd client && nohup npm run dev > "$TMPDIR/opa-frontend.log" 2>&1 &)
for i in $(seq 1 40); do curl -sf -o /dev/null localhost:8001/health && curl -sf -o /dev/null localhost:5174/ && break; sleep 1; done; curl -s localhost:8001/health
```

Stop: `pkill -f "uvicorn app.main:app --port 8001"; pkill -f vite`.

In the Claude desktop app, prefer `.claude/launch.json` (`backend`, `frontend`) via
`preview_start` — same commands, and the pane shows the page.

## Run (agent path) — the driver

```bash
/Users/issam/claude/.venv/bin/python .claude/skills/run-opa/smoke.py all
```

- `api` — health, `POST /api/auth/login`, `GET /api/cases`, `GET /api/cases/{id}`, status counts.
- `ui` — logs in via the form, opens `--path` (default `/worklist`), skips the tour,
  screenshots, then clicks the first `OPA-…` row and screenshots the case page.
- Flags: `--user sarah.kim` (supervisor), `--path /admin`, `--api URL`, `--web URL`, `--out PNG`.
- Screenshots: `.claude/skills/run-opa/screenshots/{worklist,case-detail}.png` — **open them**.
- Exits non-zero on the first failed check; prints `PASS` at the end.

Ad-hoc API calls: log in, then send the bearer token.

```bash
curl -s -X POST localhost:8001/api/auth/login -H 'content-type: application/json' -d '{"username":"ana.chen","password":"ana.chen"}'
```

## Run (human path)

`make dev` calls bare `uvicorn` (see Makefile), so activate the venv first (not exercised in this skill's verification). Open
http://localhost:5174, sign in as any seeded user with **password = username**
(`ana.chen`, `james.park` analysts; `sarah.kim`, `david.osei` supervisors).

## Test

```bash
(cd server && /Users/issam/claude/.venv/bin/pytest tests/ -q)
```

## Gotchas

- **Demo auth is password == username** (`auth_service.authenticate_user`). Wrong password → 401.
- **No token ≠ 401.** `GET /api/cases` with no auth returns 200 via the `system`-user
  fallback; but an explicit `X-User-Id: system` header returns 401. Don't use "no token"
  to test auth.
- **`/api/cases/{case_id}` takes the integer `id`**, despite the param name. Passing the
  UUID field `case_id` → 422.
- **`/api/cases?limit=` is ignored** — use `page` / `page_size` (default 25).
- **Frontend is pinned:** `strictPort: 5174`, and the Vite `/api` proxy is hardcoded to
  `localhost:8001`. A backend on any other port is invisible to the UI.
- **Fresh browser = "Analyst tour" modal** over every page; it blocks clicks until
  "Skip". The driver dismisses it.
- **Two `401 GET /api/auth/me` on `/login` are normal** — the session probe before sign-in.
- **Playwright build mismatch:** the venv's playwright wants a Chromium build that isn't
  in `~/Library/Caches/ms-playwright`; the driver launches system Chrome (`channel="chrome"`)
  to avoid a download.
- **Seeding rewrites a tracked file**, `server/seed/outputs/fee_schedule_rate_card.pdf`;
  `git checkout` it if you didn't mean to change it.
- **Stale servers squat the ports.** Check `lsof -iTCP:8001 -iTCP:5174 -sTCP:LISTEN` and
  each PID's cwd — an old copy (e.g. one in `~/.Trash`) can be serving instead of this checkout.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `BrowserType.launch: Executable doesn't exist … chromium_headless_shell-NNNN` | Install Chrome (the driver prefers it). Untested alternative: `playwright install chromium` with the venv's python |
| `GET /api/cases/<uuid>` → 422 | Use the integer `id` from the list response |
| UI loads but every API call fails | Backend isn't on :8001 (proxy target is fixed) |
| Empty DB / `no such table` | `alembic upgrade head` (startup migrations are off) |
