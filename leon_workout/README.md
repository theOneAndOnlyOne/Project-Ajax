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

## Notes

- The Hevy API has no native "label" concept, so v2 filtering is a substring
  match on `title` / `description`. Rename your routine "Push Day v2",
  "Legs v2", etc. — anything containing `v2`.
- Workout list is cached in-memory for 60 s to keep your API quota happy.
- The progressive-overload model is a simple linear regression on e1RM
  across the last 5 (configurable) sessions per exercise. Trend > 0 means
  you're climbing; trend < 0 means deload or sleep more.

> "Saved your progress. Don't waste it."
