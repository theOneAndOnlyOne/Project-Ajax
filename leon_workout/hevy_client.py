"""Thin wrapper around the Hevy public API.

The free Hevy API exposes paginated workout history at /v1/workouts.
Auth is a single `api-key` header (Hevy Pro account required).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable

import requests

HEVY_BASE = "https://api.hevyapp.com/v1"


class HevyError(RuntimeError):
    pass


@dataclass
class HevyClient:
    api_key: str
    timeout: float = 15.0
    _cache: dict = None  # type: ignore[assignment]
    _cache_ts: float = 0.0
    cache_ttl: float = 60.0

    def __post_init__(self) -> None:
        self._cache = {}

    def _headers(self) -> dict:
        return {"api-key": self.api_key, "accept": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{HEVY_BASE}{path}"
        r = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        if r.status_code == 401:
            raise HevyError("Invalid api-key. Check HEVY_API_KEY in .env.")
        if r.status_code == 429:
            raise HevyError("Rate limited by Hevy. Slow down, rookie.")
        if not r.ok:
            raise HevyError(f"Hevy API {r.status_code}: {r.text[:200]}")
        return r.json()

    def list_workouts(self, page: int = 1, page_size: int = 10) -> dict:
        return self._get("/workouts", {"page": page, "pageSize": page_size})

    def all_workouts(self, max_pages: int = 20, page_size: int = 10) -> list[dict]:
        """Pull every workout up to max_pages, with a tiny in-memory cache."""
        now = time.time()
        if self._cache.get("workouts") and now - self._cache_ts < self.cache_ttl:
            return self._cache["workouts"]

        out: list[dict] = []
        for page in range(1, max_pages + 1):
            data = self.list_workouts(page=page, page_size=page_size)
            workouts = data.get("workouts", [])
            if not workouts:
                break
            out.extend(workouts)
            page_count = data.get("page_count") or data.get("pageCount")
            if page_count and page >= page_count:
                break
        self._cache["workouts"] = out
        self._cache_ts = now
        return out

    def filter_by_label(self, workouts: Iterable[dict], label: str) -> list[dict]:
        """Match workouts whose title or description contains the label substring."""
        needle = label.lower().strip()
        if not needle:
            return list(workouts)
        keep = []
        for w in workouts:
            title = (w.get("title") or "").lower()
            desc = (w.get("description") or "").lower()
            if needle in title or needle in desc:
                keep.append(w)
        return keep


def from_env() -> HevyClient:
    key = os.environ.get("HEVY_API_KEY", "").strip()
    if not key:
        raise HevyError("HEVY_API_KEY not set. Copy .env.example to .env and fill it in.")
    return HevyClient(api_key=key)
