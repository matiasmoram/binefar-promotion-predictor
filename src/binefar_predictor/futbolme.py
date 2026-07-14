"""Second data source: Futbolme top-scorer (goleadores) table.

Sofascore incidents give us goals *per match* (from which we build our own
scorer tallies); Futbolme publishes the official-style **season goleadores
table** for Tercera Federación Grupo 17. We scrape it as an independent
cross-check on the goalscorer model — if our incident-derived tallies disagree
wildly with Futbolme's table, something is wrong.

Futbolme serves plain server-rendered HTML (no JS wall), so a browser-TLS fetch
via curl_cffi is enough. BeSoccer, which is richer, sits behind a JavaScript
"Client Challenge" and is intentionally not used here (it would need a headless
browser and a heavy runtime dependency).
"""
from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as _creq

from . import config

# Futbolme numeric tournament id for Tercera Federación Grupo 17 (stable; the
# slug in the path is cosmetic).
FUTBOLME_TOURNAMENT_ID = 3071
_BASE = "https://www.futbolme.com/resultados-directo/torneo"
_HEADERS = {"Accept-Language": "es-ES,es;q=0.9"}


def fetch_top_scorers(
    tournament_id: int = FUTBOLME_TOURNAMENT_ID,
    slug: str = "tercera-federacion-grupo-17",
) -> pd.DataFrame:
    """Return the goleadores table: rank, player, team, goals.

    Empty DataFrame on any failure (never raises — it's a cross-check, not a
    dependency).
    """
    url = f"{_BASE}/{slug}/{tournament_id}/goleadores"
    try:
        r = _creq.get(url, headers=_HEADERS, impersonate=config.IMPERSONATE, timeout=config.REQUEST_TIMEOUT)
        if r.status_code != 200:
            return pd.DataFrame()
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        return pd.DataFrame()

    rows = []
    for tr in soup.find_all("tr"):
        a = tr.find("a", href=re.compile(r"/jugador/"))
        if not a:
            continue
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        # expected shape: [rank, player, team, goals]
        if len(cells) >= 4 and cells[-1].isdigit():
            rows.append(
                {
                    "rank": int(cells[0]) if cells[0].isdigit() else None,
                    "player": a.get_text(strip=True),
                    "team": cells[2],
                    "goals": int(cells[-1]),
                }
            )
    return pd.DataFrame(rows)


def team_top_scorers(team_substring: str = "Bin", **kw) -> pd.DataFrame:
    df = fetch_top_scorers(**kw)
    if df.empty:
        return df
    return df[df["team"].str.contains(team_substring, case=False, na=False)].reset_index(drop=True)
