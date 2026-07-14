"""Best-effort squad scraper (Transfermarkt).

At tier 5 Transfermarkt exposes the roster (names, positions, ages,
nationalities) but leaves **market values empty** ("-") for essentially the
whole squad — so a value-weighted strength prior is not usable here. We still
pull the roster because the club's squad is part of an exhaustive picture, and
we surface *whatever* values exist so the value->strength hook is ready the day
this club (or a higher-tier one) has priced players.

If Transfermarkt is unreachable, functions degrade to an empty roster rather
than raising — the predictor never depends on this data.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as _creq

from . import config

_TM_BASE = "https://www.transfermarkt.com"
_HEADERS = {"Accept-Language": "en-US,en;q=0.9"}

_VALUE_RE = re.compile(r"€\s*([\d.,]+)\s*([mk]?)", re.IGNORECASE)


@dataclass
class Player:
    name: str
    position: str | None
    age: int | None
    nationality: str | None
    market_value_eur: float | None


def _parse_value(text: str) -> float | None:
    m = _VALUE_RE.search(text.replace("\xa0", " "))
    if not m:
        return None
    num = float(m.group(1).replace(".", "").replace(",", "."))
    unit = m.group(2).lower()
    if unit == "m":
        return num * 1_000_000
    if unit == "k":
        return num * 1_000
    return num


def fetch_squad(
    club_id: int = config.TRANSFERMARKT_CLUB_ID,
    season_year: int = 2025,
    slug: str = "cd-binefar",
) -> list[Player]:
    """Return the squad for a given season (``season_year`` = start year).

    Returns an empty list on any network/parse failure.
    """
    url = f"{_TM_BASE}/{slug}/kader/verein/{club_id}/saison_id/{season_year}/plus/1"
    try:
        resp = _creq.get(url, headers=_HEADERS, impersonate=config.IMPERSONATE, timeout=config.REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return []

    table = soup.select_one("table.items")
    if not table:
        return []

    players: list[Player] = []
    for tr in table.select("tbody > tr"):
        name_el = tr.select_one("td.hauptlink a")
        if not name_el:
            continue
        tds = tr.find_all("td", recursive=False)
        pos_el = tr.select_one("table.inline-table tr:nth-of-type(2) td")
        # age: a "(23)" inside a birthdate cell, or a standalone age column
        age = None
        age_match = re.search(r"\((\d{2})\)", tr.get_text(" "))
        if age_match:
            age = int(age_match.group(1))
        nat_img = tr.select_one("img.flaggenrahmen")
        nationality = nat_img.get("title") if nat_img else None
        market_value = _parse_value(tds[-1].get_text(strip=True)) if tds else None
        players.append(
            Player(
                name=name_el.get_text(strip=True),
                position=pos_el.get_text(strip=True) if pos_el else None,
                age=age,
                nationality=nationality,
                market_value_eur=market_value,
            )
        )
    return players


def squad_dataframe(players: list[Player]) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) for p in players])


def squad_value_summary(players: list[Player]) -> dict:
    """Aggregate squad value stats (usually mostly None at this tier)."""
    values = [p.market_value_eur for p in players if p.market_value_eur is not None]
    return {
        "squad_size": len(players),
        "n_valued": len(values),
        "total_value_eur": float(sum(values)) if values else None,
        "log_total_value": float(math.log(sum(values))) if values else None,
    }
