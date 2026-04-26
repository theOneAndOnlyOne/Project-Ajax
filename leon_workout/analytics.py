"""Workout analytics: progressive overload projection, volume, PRs, balance."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean
from typing import Iterable


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _working_sets(exercise: dict) -> list[dict]:
    out = []
    for s in exercise.get("sets", []) or []:
        if s.get("set_type") == "warmup":
            continue
        if s.get("weight_kg") is None or s.get("reps") is None:
            continue
        out.append(s)
    return out


def _epley_1rm(weight: float, reps: int) -> float:
    """Estimated 1RM via Epley formula. Caps at 1 rep = weight."""
    if reps <= 0 or weight <= 0:
        return 0.0
    if reps == 1:
        return weight
    return weight * (1 + reps / 30.0)


def per_exercise_history(workouts: Iterable[dict]) -> dict:
    """Group sets by exercise template id with timestamps for trend analysis."""
    history: dict[str, dict] = {}
    for w in workouts:
        ts = _parse_ts(w.get("start_time") or w.get("created_at"))
        for ex in w.get("exercises", []) or []:
            eid = ex.get("exercise_template_id") or ex.get("title")
            if not eid:
                continue
            entry = history.setdefault(
                eid,
                {"title": ex.get("title", "Unknown"), "sessions": []},
            )
            sets = _working_sets(ex)
            if not sets:
                continue
            top = max(sets, key=lambda s: _epley_1rm(s["weight_kg"], s["reps"]))
            volume = sum(s["weight_kg"] * s["reps"] for s in sets)
            entry["sessions"].append(
                {
                    "ts": ts.isoformat() if ts else None,
                    "top_weight": top["weight_kg"],
                    "top_reps": top["reps"],
                    "e1rm": round(_epley_1rm(top["weight_kg"], top["reps"]), 1),
                    "volume": round(volume, 1),
                    "sets": len(sets),
                }
            )
    for entry in history.values():
        entry["sessions"].sort(key=lambda s: s["ts"] or "")
    return history


def progressive_overload(history: dict, lookback: int = 5) -> list[dict]:
    """Project the next session's expected top set from a simple linear fit."""
    out = []
    for eid, entry in history.items():
        sessions = entry["sessions"][-lookback:]
        if len(sessions) < 2:
            continue
        # Linear fit: index -> e1rm
        xs = list(range(len(sessions)))
        ys = [s["e1rm"] for s in sessions]
        n = len(xs)
        x_mean = mean(xs)
        y_mean = mean(ys)
        denom = sum((x - x_mean) ** 2 for x in xs) or 1e-9
        slope = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n)) / denom
        intercept = y_mean - slope * x_mean
        projected_e1rm = slope * n + intercept

        last = sessions[-1]
        # Suggest a concrete next prescription: same reps, +slope kg (rounded to 1.25 kg plate increment)
        suggested_weight = last["top_weight"] + slope
        plate_step = 1.25
        suggested_weight = round(suggested_weight / plate_step) * plate_step
        suggested_weight = max(suggested_weight, last["top_weight"])

        out.append(
            {
                "exercise_id": eid,
                "title": entry["title"],
                "last_top": f"{last['top_weight']}kg x {last['top_reps']}",
                "last_e1rm": last["e1rm"],
                "projected_e1rm": round(projected_e1rm, 1),
                "trend_kg_per_session": round(slope, 2),
                "suggested_next": f"{suggested_weight}kg x {last['top_reps']}",
                "sessions_used": len(sessions),
            }
        )
    out.sort(key=lambda r: -r["trend_kg_per_session"])
    return out


def overall_stats(workouts: list[dict]) -> dict:
    """Aggregate volume, frequency, PRs, and recency."""
    total_volume = 0.0
    total_sets = 0
    total_reps = 0
    pr_by_exercise: dict[str, dict] = {}
    days = set()
    durations = []

    for w in workouts:
        start = _parse_ts(w.get("start_time"))
        end = _parse_ts(w.get("end_time"))
        if start and end:
            durations.append((end - start).total_seconds() / 60.0)
        if start:
            days.add(start.date().isoformat())
        for ex in w.get("exercises", []) or []:
            title = ex.get("title", "Unknown")
            for s in _working_sets(ex):
                vol = s["weight_kg"] * s["reps"]
                total_volume += vol
                total_sets += 1
                total_reps += s["reps"]
                e1rm = _epley_1rm(s["weight_kg"], s["reps"])
                pr = pr_by_exercise.get(title)
                if not pr or e1rm > pr["e1rm"]:
                    pr_by_exercise[title] = {
                        "e1rm": round(e1rm, 1),
                        "weight": s["weight_kg"],
                        "reps": s["reps"],
                        "ts": start.isoformat() if start else None,
                    }

    # Frequency: workouts in the last 7 / 30 days
    now = datetime.utcnow()
    last7 = sum(
        1
        for w in workouts
        if (t := _parse_ts(w.get("start_time"))) and (now - t.replace(tzinfo=None)) < timedelta(days=7)
    )
    last30 = sum(
        1
        for w in workouts
        if (t := _parse_ts(w.get("start_time"))) and (now - t.replace(tzinfo=None)) < timedelta(days=30)
    )

    return {
        "workout_count": len(workouts),
        "total_volume_kg": round(total_volume, 1),
        "total_sets": total_sets,
        "total_reps": total_reps,
        "avg_duration_min": round(mean(durations), 1) if durations else 0,
        "training_days": len(days),
        "last_7_days": last7,
        "last_30_days": last30,
        "prs": [
            {"exercise": k, **v}
            for k, v in sorted(pr_by_exercise.items(), key=lambda kv: -kv[1]["e1rm"])
        ][:15],
    }


def volume_timeline(workouts: list[dict], buckets: int = 12) -> list[dict]:
    """Per-week volume for the last `buckets` weeks."""
    by_week: dict[str, float] = defaultdict(float)
    for w in workouts:
        ts = _parse_ts(w.get("start_time"))
        if not ts:
            continue
        # ISO week key
        year, week, _ = ts.isocalendar()
        key = f"{year}-W{week:02d}"
        for ex in w.get("exercises", []) or []:
            for s in _working_sets(ex):
                by_week[key] += s["weight_kg"] * s["reps"]
    series = sorted(by_week.items())[-buckets:]
    return [{"week": k, "volume": round(v, 1)} for k, v in series]


def muscle_balance(workouts: list[dict]) -> list[dict]:
    """Cheap heuristic: bucket exercise titles by keywords. No external db."""
    buckets = {
        "PUSH": ["bench", "press", "dip", "push", "tricep", "shoulder", "overhead"],
        "PULL": ["row", "pull", "chin", "curl", "lat", "face pull", "rear delt"],
        "LEGS": ["squat", "deadlift", "lunge", "leg", "calf", "hip", "glute", "rdl"],
        "CORE": ["ab", "crunch", "plank", "sit-up", "rollout", "hanging"],
    }
    totals = {k: 0.0 for k in buckets}
    for w in workouts:
        for ex in w.get("exercises", []) or []:
            title = (ex.get("title") or "").lower()
            for bucket, keys in buckets.items():
                if any(k in title for k in keys):
                    for s in _working_sets(ex):
                        totals[bucket] += s["weight_kg"] * s["reps"]
                    break
    return [{"group": k, "volume": round(v, 1)} for k, v in totals.items()]
