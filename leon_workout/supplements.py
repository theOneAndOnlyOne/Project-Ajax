"""Supplement tracking: creatine, protein, electrolytes. JSON-file persistence."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock

DATA_FILE = Path(__file__).parent / "data" / "supplements.json"
_LOCK = Lock()

# Sane defaults — Leon trains hard.
DEFAULT_TARGETS = {
    "creatine_g": 5.0,        # daily monohydrate
    "protein_g": 160.0,       # rough cut for an 80kg lifter (~2 g/kg)
    "electrolytes_mg_sodium": 2000.0,
    "electrolytes_mg_potassium": 1000.0,
    "water_ml": 3000.0,
}


def _load() -> dict:
    if not DATA_FILE.exists():
        return {"targets": DEFAULT_TARGETS, "log": {}}
    try:
        return json.loads(DATA_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"targets": DEFAULT_TARGETS, "log": {}}


def _save(state: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, DATA_FILE)


def _today() -> str:
    return date.today().isoformat()


def get_state(day: str | None = None) -> dict:
    with _LOCK:
        state = _load()
    targets = {**DEFAULT_TARGETS, **state.get("targets", {})}
    day = day or _today()
    today = state.get("log", {}).get(day, {k: 0 for k in targets})
    streaks = _compute_streaks(state.get("log", {}), targets)
    return {
        "date": day,
        "targets": targets,
        "today": today,
        "streaks": streaks,
        "history": _last_n_days(state.get("log", {}), targets, n=7),
    }


def add_intake(field: str, amount: float) -> dict:
    if field not in DEFAULT_TARGETS:
        raise ValueError(f"Unknown supplement field: {field}")
    with _LOCK:
        state = _load()
        log = state.setdefault("log", {})
        day_entry = log.setdefault(_today(), {k: 0 for k in DEFAULT_TARGETS})
        day_entry[field] = round(day_entry.get(field, 0) + amount, 2)
        _save(state)
    return get_state()


def reset_today(field: str | None = None) -> dict:
    with _LOCK:
        state = _load()
        log = state.setdefault("log", {})
        today = log.setdefault(_today(), {k: 0 for k in DEFAULT_TARGETS})
        if field:
            today[field] = 0
        else:
            for k in today:
                today[k] = 0
        _save(state)
    return get_state()


def set_targets(new_targets: dict) -> dict:
    with _LOCK:
        state = _load()
        targets = state.setdefault("targets", DEFAULT_TARGETS.copy())
        for k, v in new_targets.items():
            if k in DEFAULT_TARGETS and v is not None:
                targets[k] = float(v)
        _save(state)
    return get_state()


def _compute_streaks(log: dict, targets: dict) -> dict:
    """Consecutive days hitting the creatine target ending today."""
    streak = 0
    d = date.today()
    while True:
        entry = log.get(d.isoformat())
        if not entry:
            break
        if entry.get("creatine_g", 0) < targets.get("creatine_g", 5.0):
            break
        streak += 1
        d -= timedelta(days=1)
    return {"creatine": streak}


def _last_n_days(log: dict, targets: dict, n: int = 7) -> list[dict]:
    out = []
    today = date.today()
    for i in range(n - 1, -1, -1):
        d = today - timedelta(days=i)
        key = d.isoformat()
        entry = log.get(key, {})
        out.append(
            {
                "date": key,
                "creatine_g": entry.get("creatine_g", 0),
                "protein_g": entry.get("protein_g", 0),
                "water_ml": entry.get("water_ml", 0),
                "creatine_hit": entry.get("creatine_g", 0) >= targets.get("creatine_g", 5.0),
                "protein_hit": entry.get("protein_g", 0) >= targets.get("protein_g", 160.0),
            }
        )
    return out
