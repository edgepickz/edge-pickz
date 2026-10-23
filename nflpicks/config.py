"""Settings and static reference data."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("NFLPICKS_CACHE", ROOT / "data_cache"))
OUT_DIR = Path(os.environ.get("NFLPICKS_OUT", ROOT / "site"))
HISTORY_FILE = Path(os.environ.get("NFLPICKS_HISTORY", ROOT / "history" / "locked_picks.json"))

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()

# Picks lock this many hours before kickoff (the "official" pick).
LOCK_HOURS = 0.5  # final picks lock 30 minutes before kickoff
# Show games kicking off within this many days.
LOOKAHEAD_DAYS = 8

# Model knobs
SD_MARGIN = 13.4         # std dev of final margin around expectation
SD_TOTAL = 13.0          # std dev of total points
MODEL_TRUST = 0.35       # how far we move off the market toward our model (0 = pure market)
RATING_HALF_LIFE_WEEKS = 10
HFA = 1.6                # home field advantage in points (fit value is blended with this)
MIN_EV_GAME = 0.02       # min expected value (per $1) to call a game pick a "bet"
MIN_EV_PROP = 0.04
PROP_SIMS = 20000

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
NAME_TO_ABBR = {v: k for k, v in TEAM_NAMES.items()}
NAME_TO_ABBR["Washington Football Team"] = "WAS"

# ESPN uses a few different abbreviations.
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}

# Home stadium coordinates (lat, lon) for weather lookups.
TEAM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4009), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9737, -93.2581),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9078, -76.8645),
}
# Neutral-site venues, matched by a word in the stadium name.
VENUE_COORDS = {
    "Stade de France": (48.9244, 2.3601), "Tottenham": (51.6043, -0.0664),
    "Wembley": (51.5560, -0.2796), "Bernabeu": (40.4531, -3.6883),
    "Munich": (48.2188, 11.6247), "Allianz": (48.2188, 11.6247),
    "Banorte": (19.3029, -99.1505), "Azteca": (19.3029, -99.1505),
    "Melbourne": (-37.8200, 144.9834), "Maracana": (-22.9122, -43.2302),
    "Corinthians": (-23.5453, -46.4742), "Croke": (53.3607, -6.2512),
}
INDOOR_ROOFS = {"dome", "closed"}
