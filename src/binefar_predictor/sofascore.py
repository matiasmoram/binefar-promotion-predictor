"""Thin, polite Sofascore API client with on-disk caching.

Only the endpoints that actually return data for a tier-5 Spanish league are
wrapped: standings, season lists, and event (match) lists. Match statistics,
line-ups and per-player stats return 404 at this level and are deliberately
not exposed.

Caching: every raw JSON response is written to ``data/raw`` keyed by a slug of
the endpoint. Re-runs read from disk unless ``force_refresh=True``. This makes
the whole pipeline reproducible and keeps us off Sofascore's rate limiter.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from curl_cffi import requests as _creq

from . import config

_LOG_PREFIX = "[sofascore]"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


class SofascoreClient:
    """Fetches and caches Sofascore JSON."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        delay: float = config.REQUEST_DELAY_SECONDS,
        verbose: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else config.RAW_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.verbose = verbose
        self._last_request_ts = 0.0

    # -- internals ---------------------------------------------------------- #
    def _log(self, *args: Any) -> None:
        if self.verbose:
            print(_LOG_PREFIX, *args)

    def _cache_path(self, endpoint: str) -> Path:
        return self.cache_dir / f"{_slug(endpoint)}.json"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_ts = time.monotonic()

    def get(self, endpoint: str, force_refresh: bool = False) -> dict | None:
        """GET ``{API}/{endpoint}`` with caching and retry.

        Returns the parsed JSON dict, or ``None`` if the resource does not
        exist (HTTP 404 — expected for empty endpoints at this tier).
        """
        cache_path = self._cache_path(endpoint)
        if cache_path.exists() and not force_refresh:
            with cache_path.open(encoding="utf-8") as fh:
                return json.load(fh)

        url = f"{config.SOFASCORE_API}/{endpoint.lstrip('/')}"
        last_err: Exception | None = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = _creq.get(
                    url,
                    headers=config.HTTP_HEADERS,
                    impersonate=config.IMPERSONATE,
                    timeout=config.REQUEST_TIMEOUT,
                )
            except Exception as exc:  # network / TLS errors -> retry
                last_err = exc
                self._log(f"attempt {attempt} error: {type(exc).__name__}: {exc}")
                time.sleep(self.delay * attempt)
                continue

            if resp.status_code == 404:
                self._log(f"404 (no data): {endpoint}")
                return None
            if resp.status_code == 200:
                data = resp.json()
                with cache_path.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False)
                self._log(f"OK {endpoint} ({len(resp.content)} bytes)")
                return data
            # 403/429/5xx -> back off and retry
            self._log(f"attempt {attempt} HTTP {resp.status_code}: {endpoint}")
            last_err = RuntimeError(f"HTTP {resp.status_code}")
            time.sleep(self.delay * attempt * 2)

        raise RuntimeError(f"Failed to fetch {endpoint}: {last_err}")

    # -- typed endpoint helpers -------------------------------------------- #
    def team(self, team_id: int = config.SOFASCORE_TEAM_ID, **kw) -> dict | None:
        return self.get(f"team/{team_id}", **kw)

    def list_seasons(
        self, tournament_id: int = config.UNIQUE_TOURNAMENT_ID, **kw
    ) -> dict[str, int]:
        """Return ``{year_label: season_id}`` for a unique tournament."""
        data = self.get(f"unique-tournament/{tournament_id}/seasons", **kw)
        if not data:
            return {}
        return {s["year"]: s["id"] for s in data.get("seasons", [])}

    def standings(
        self,
        season_id: int,
        tournament_id: int = config.UNIQUE_TOURNAMENT_ID,
        **kw,
    ) -> dict | None:
        return self.get(
            f"unique-tournament/{tournament_id}/season/{season_id}/standings/total",
            **kw,
        )

    def incidents(self, event_id: int, **kw) -> dict | None:
        """Match incidents — includes goals with scorer names (works at tier 5).

        Line-ups and statistics are 404 at this level, but goal incidents carry
        the scoring player, minute and side, which is enough for a goalscorer
        model.
        """
        return self.get(f"event/{event_id}/incidents", **kw)

    def season_events(
        self,
        season_id: int,
        tournament_id: int = config.UNIQUE_TOURNAMENT_ID,
        upcoming: bool = False,
        max_pages: int = 20,
        **kw,
    ) -> list[dict]:
        """All events (matches) for a league-season, following pagination.

        ``upcoming=False`` returns finished/past events; ``True`` returns the
        scheduled fixtures.
        """
        direction = "next" if upcoming else "last"
        events: list[dict] = []
        for page in range(max_pages):
            endpoint = (
                f"unique-tournament/{tournament_id}/season/{season_id}"
                f"/events/{direction}/{page}"
            )
            data = self.get(endpoint, **kw)
            if not data or not data.get("events"):
                break
            events.extend(data["events"])
            # Sofascore signals the last page with hasNextPage=False.
            if not data.get("hasNextPage", False):
                break
        return events
