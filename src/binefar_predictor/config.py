"""Static configuration: identifiers, endpoints, league rules.

Everything that ties the pipeline to *this specific club and league* lives here so
the rest of the code stays generic. All IDs were verified live against the
Sofascore API on 2026-07-14 (see README for provenance).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# The club and its competition (Sofascore identifiers)
# --------------------------------------------------------------------------- #
CLUB_NAME = "CD Binéfar"
SOFASCORE_TEAM_ID = 263819               # CD Binéfar
TRANSFERMARKT_CLUB_ID = 21551            # cd-binefar

# Tercera Federación, Group 17 (Aragón) — tier 5 of the Spanish pyramid.
UNIQUE_TOURNAMENT_ID = 11366

# Sofascore season IDs for uniqueTournament 11366, verified 2026-07-14.
# These rotate every August; ``SofascoreClient.list_seasons`` resolves them
# dynamically. This map is a pinned fallback for offline / reproducible runs.
SEASON_IDS = {
    "25/26": 81196,
    "24/25": 66118,
    "23/24": 54355,
    "22/23": 45387,
    "21/22": 38179,
    "20/21": 34561,
    "19/20": 25075,
    "2018": 18384,
    "17/18": 14120,
}
# Season we are trying to forecast promotion *for*.
TARGET_SEASON = "26/27"

# --------------------------------------------------------------------------- #
# Promotion rules for Tercera Federación Group 17 -> Segunda Federación
# --------------------------------------------------------------------------- #
# 18 teams, 34-match double round-robin.
#   - 1st place: automatic/direct promotion.
#   - 2nd-5th: territorial promotion play-off (semi-finals + final). The winner
#     advances to a national inter-group phase for a further Segunda Fed. spot.
# We model "promoted" = won direct promotion OR came through the play-off route.
LEAGUE_SIZE = 18
MATCHES_PER_TEAM = (LEAGUE_SIZE - 1) * 2  # 34
DIRECT_PROMOTION_SLOTS = 1
PLAYOFF_SLOTS = 4                          # positions 2..5
PLAYOFF_POSITIONS = tuple(range(2, 2 + PLAYOFF_SLOTS))  # (2, 3, 4, 5)

# The territorial play-off (positions 2-5) is simulated explicitly (semis +
# final, single match, home tie to the higher seed). Its winner does NOT go up
# automatically: they enter a national inter-group phase for a limited number of
# extra Segunda Federación slots. This constant is the probability the
# territorial winner converts that national phase into actual promotion.
# Tunable; see README "Play-off model".
NATIONAL_PHASE_CONVERSION = 0.40

# Teams in the target-season group with no rating history (newly promoted from
# Regional Preferente / relegated from Segunda Fed) are modelled as league
# average shifted by this net-strength penalty (promoted sides tend to be below
# average). 0.0 = neutral. See README "Cold-start of unknown teams".
NEWCOMER_NET_STRENGTH_PENALTY = 0.25

# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
SOFASCORE_API = "https://api.sofascore.com/api/v1"
# The API is Cloudflare-fronted and TLS-fingerprints clients, so a browser
# User-Agent is not enough — we impersonate a real browser's TLS stack via
# curl_cffi. This header set is still sent alongside.
HTTP_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}
IMPERSONATE = "chrome"          # curl_cffi browser profile
REQUEST_DELAY_SECONDS = 1.0     # be gentle: ~1 req/sec
REQUEST_TIMEOUT = 25
MAX_RETRIES = 4

# Sofascore event status code for a finished match.
STATUS_FINISHED = 100
