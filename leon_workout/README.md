# LEON — Tactical Training Console

A Resident Evil 4 / Leon S. Kennedy themed dashboard for your Hevy workouts.
Filters your account to "v2"-labelled sessions (substring match on title or
description), projects progressive overload, tracks supplements, and runs
a rest-timer / stopwatch — all wrapped in an attaché-case CRT UI.

## Requirements

- Python 3.10+
- A **Hevy Pro** account with an API key
  (grab it at https://hevy.com/settings?developer)

## Setup

```bash
cd leon_workout
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — paste your HEVY_API_KEY, optionally change LEON_LABEL
python app.py
```

Open http://127.0.0.1:5000

## .env

| key            | default | meaning                                                |
|----------------|---------|--------------------------------------------------------|
| `HEVY_API_KEY` | —       | required; from your Hevy developer settings           |
| `LEON_LABEL`   | `v2`    | substring matched against workout title / description |
| `LEON_PORT`    | `5000`  | local port                                             |

## Views

- **INVENTORY** — every v2 workout as an attaché-case slot (top set per
  exercise, total volume, duration).
- **FILES** — progressive-overload table: linear fit on Epley e1RM over the
  last N sessions, suggests next top-set weight rounded to 1.25 kg. Plus
  global stats (volume, sets/reps, training days, PRs), weekly volume bar
  chart, and push/pull/legs/core balance bars.
- **CASE** — supplement log for creatine, protein, sodium, potassium, water.
  Daily targets, hit-streaks, 7-day history. Persists to
  `data/supplements.json`.
- **TIMER** — countdown for rest periods (with audio cue) and a stopwatch
  with lap splits for whole-workout timing.
- **MAP** — quick reference / config notes.

## Hosting

The app is a normal WSGI application — gunicorn is in `requirements.txt`
and a `Dockerfile`, `Procfile`, and `fly.toml` are included. Pick one:

### Option 1: Fly.io (recommended — free tier, ~5 min)

```bash
brew install flyctl   # or: curl -L https://fly.io/install.sh | sh
cd leon_workout
fly auth login
fly launch --no-deploy --copy-config       # accept defaults; rename app if 'leon-tactical' is taken
fly volumes create leon_data --size 1      # persists data/supplements.json
fly secrets set HEVY_API_KEY=your-rotated-key LEON_PASSWORD=pickAStrongOne
fly deploy
```

You'll get an `https://<app>.fly.dev` URL. Auto-stops when idle, ~$0/mo
for one user.

### Option 2: Render / Railway / Heroku-likes

Push the repo, point the service at `leon_workout/`, set the start command
to use the `Procfile` (most platforms detect it). Add env vars:
`HEVY_API_KEY`, `LEON_PASSWORD`. Mount a persistent disk at `/app/data`
if you want supplement history to survive redeploys.

### Option 3: Docker on any VPS

```bash
cd leon_workout
docker build -t leon .
docker run -d --name leon -p 8080:8080 \
  -e HEVY_API_KEY=your-rotated-key \
  -e LEON_PASSWORD=pickAStrongOne \
  -v leon_data:/app/data \
  --restart unless-stopped leon
```

Then put nginx / Caddy / Cloudflare Tunnel in front for TLS.

### Option 4: Just expose your laptop (for the gym)

Run locally, then expose with one of these — no server, no deploy:

```bash
# Cloudflare Tunnel (free, handles TLS + auth):
cloudflared tunnel --url http://localhost:5000

# ngrok (free tier):
ngrok http 5000
```

For either, set `LEON_PASSWORD` first or anyone with the URL can read your data.

### Option 5: Tailscale (zero public exposure)

If you only need it on your own devices, run locally and join your laptop
+ phone to a Tailscale tailnet. The app is then reachable at
`http://<machine>:5000` from your phone in the gym, with no public surface.

### Auth

Set `LEON_PASSWORD` (and optionally `LEON_USERNAME`, default `leon`) to
gate the whole app behind HTTP Basic Auth. **Required for any public
deployment** — otherwise anyone with the URL can read your workouts and
write to your supplement log. Leave blank for local-only use.

## Notes

- The Hevy API has no native "label" concept, so v2 filtering is a substring
  match on `title` / `description`. Rename your routine "Push Day v2",
  "Legs v2", etc. — anything containing `v2`.
- Workout list is cached in-memory for 60 s to keep your API quota happy.
- The progressive-overload model is a simple linear regression on e1RM
  across the last 5 (configurable) sessions per exercise. Trend > 0 means
  you're climbing; trend < 0 means deload or sleep more.

> "Saved your progress. Don't waste it."
