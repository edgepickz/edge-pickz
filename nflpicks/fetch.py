"""Data sources. Every source is optional except the nflverse files:
if a feed is down, the run continues and the page says what was missing."""
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config

UA = {"User-Agent": "nflpicks/1.0 (personal research tool)"}
_status = {}


def source_status():
    return dict(_status)


def _get(url, params=None, timeout=30):
    r = requests.get(url, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


def _cached_csv(name, url, max_age_hours=2):
    """Download a CSV into the cache unless a fresh copy is already there."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / name
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600
    if not fresh:
        try:
            path.write_bytes(_get(url, timeout=120).content)
        except Exception as e:  # fall back to a stale copy if there is one
            if not path.exists():
                raise
            _status[name] = f"stale cache used ({e.__class__.__name__})"
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------- nflverse
def load_nflverse(season):
    prev = season - 1
    data = {
        "games": _cached_csv("games.csv", f"{config.NFLVERSE}/schedules/games.csv"),
        "players": _cached_csv(f"stats_player_week_{season}.csv",
                               f"{config.NFLVERSE}/stats_player/stats_player_week_{season}.csv"),
        "players_prev": _cached_csv(f"stats_player_week_{prev}.csv",
                                    f"{config.NFLVERSE}/stats_player/stats_player_week_{prev}.csv",
                                    max_age_hours=24 * 30),
        "teams": _cached_csv(f"stats_team_week_{season}.csv",
                             f"{config.NFLVERSE}/stats_team/stats_team_week_{season}.csv"),
        "teams_prev": _cached_csv(f"stats_team_week_{prev}.csv",
                                  f"{config.NFLVERSE}/stats_team/stats_team_week_{prev}.csv",
                                  max_age_hours=24 * 30),
    }
    try:
        data["injuries"] = _cached_csv(f"injuries_{season}.csv",
                                       f"{config.NFLVERSE}/injuries/injuries_{season}.csv")
        _status.setdefault("nflverse injuries", "ok")
    except Exception as e:
        data["injuries"] = pd.DataFrame()
        _status["nflverse injuries"] = f"unavailable ({e.__class__.__name__})"
    _status.setdefault("nflverse stats/schedule/lines", "ok")
    return data


# ---------------------------------------------------------------- ESPN
def espn_injuries():
    """Latest injury designations from ESPN, keyed by team abbr -> list."""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
    try:
        js = _get(url).json()
    except Exception as e:
        _status["ESPN injuries"] = f"unavailable ({e.__class__.__name__})"
        return {}
    out = {}
    for team in js.get("injuries", []):
        abbr = None
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {})
            t = (ath.get("team") or {}).get("abbreviation")
            if t:
                abbr = config.ESPN_ABBR.get(t, t)
            out.setdefault(abbr or team.get("displayName", "?"), []).append({
                "name": ath.get("displayName"),
                "position": (ath.get("position") or {}).get("abbreviation"),
                "status": inj.get("status"),
                "detail": inj.get("shortComment") or "",
            })
        if abbr is None and team.get("displayName") in config.NAME_TO_ABBR:
            name = team["displayName"]
            out[config.NAME_TO_ABBR[name]] = out.pop(name, [])
    _status["ESPN injuries"] = "ok"
    return out


def espn_news(limit=80):
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"
    try:
        js = _get(url, params={"limit": limit}).json()
    except Exception as e:
        _status["ESPN news"] = f"unavailable ({e.__class__.__name__})"
        return []
    items = []
    for a in js.get("articles", []):
        teams = set()
        for c in a.get("categories", []):
            if c.get("type") == "team":
                t = (c.get("team") or {}).get("abbreviation") or c.get("description")
                if t:
                    t = config.ESPN_ABBR.get(t, t)
                    teams.add(config.NAME_TO_ABBR.get(t, t))
        items.append({
            "headline": a.get("headline"),
            "description": a.get("description"),
            "published": a.get("published"),
            "link": ((a.get("links") or {}).get("web") or {}).get("href"),
            "teams": sorted(teams),
        })
    _status["ESPN news"] = "ok"
    return items


# ---------------------------------------------------------------- weather
def weather_at(lat, lon, kickoff_utc):
    """Open-Meteo forecast for the kickoff hour (free, no key)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability,precipitation",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
        "timezone": "UTC", "forecast_days": 16,
    }
    try:
        h = _get(url, params=params).json()["hourly"]
    except Exception as e:
        _status["Open-Meteo weather"] = f"unavailable ({e.__class__.__name__})"
        return None
    target = kickoff_utc.strftime("%Y-%m-%dT%H:00")
    if target not in h["time"]:
        return None
    i = h["time"].index(target)
    # Average over the ~3 game hours.
    sl = slice(i, min(i + 3, len(h["time"])))
    avg = lambda k: sum(x or 0 for x in h[k][sl]) / len(h[k][sl])
    _status["Open-Meteo weather"] = "ok"
    return {
        "temp_f": round(avg("temperature_2m")),
        "wind_mph": round(avg("wind_speed_10m")),
        "gust_mph": round(max(x or 0 for x in h["wind_gusts_10m"][sl])),
        "precip_prob": round(max(x or 0 for x in h["precipitation_probability"][sl])),
        "precip_in": round(sum(x or 0 for x in h["precipitation"][sl]), 2),
    }


# ---------------------------------------------------------------- odds
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
PROP_MARKETS = ["player_pass_yds", "player_pass_tds", "player_rush_yds",
                "player_reception_yds", "player_receptions", "player_anytime_td"]


def odds_game_lines():
    """Current best spread/total/moneyline across US books (needs ODDS_API_KEY).
    Costs 3 credits per call on The Odds API."""
    if not config.ODDS_API_KEY:
        _status["Live odds (The Odds API)"] = "no API key set, using nflverse consensus lines"
        return {}
    try:
        r = _get(f"{ODDS_BASE}/odds", params={
            "apiKey": config.ODDS_API_KEY, "regions": "us",
            "markets": "h2h,spreads,totals", "oddsFormat": "american"})
    except Exception as e:
        _status["Live odds (The Odds API)"] = f"unavailable ({e.__class__.__name__})"
        return {}
    _status["Live odds (The Odds API)"] = f"ok, {r.headers.get('x-requests-remaining', '?')} credits left"
    out = {}
    for ev in r.json():
        home = config.NAME_TO_ABBR.get(ev["home_team"])
        away = config.NAME_TO_ABBR.get(ev["away_team"])
        if not home or not away:
            continue
        g = {"event_id": ev["id"], "books": {}}
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    side = config.NAME_TO_ABBR.get(o["name"], o["name"])
                    key = (m["key"], side)
                    cand = {"point": o.get("point"), "price": o["price"], "book": bk["title"]}
                    best = g["books"].get(key)
                    if best is None or _better(m["key"], side, cand, best, home):
                        g["books"][key] = cand
        out[(away, home)] = g
    return out


def _better(market, side, cand, best, home):
    """Pick the line that is best for the bettor."""
    if market in ("spreads", "totals") and cand["point"] != best["point"]:
        if market == "spreads":
            return cand["point"] > best["point"]
        return cand["point"] < best["point"] if side == "Over" else cand["point"] > best["point"]
    return cand["price"] > best["price"]


def odds_event_props(event_id):
    """Player prop lines for one game. Costs len(PROP_MARKETS) credits."""
    if not config.ODDS_API_KEY:
        return []
    try:
        r = _get(f"{ODDS_BASE}/events/{event_id}/odds", params={
            "apiKey": config.ODDS_API_KEY, "regions": "us",
            "markets": ",".join(PROP_MARKETS), "oddsFormat": "american"})
    except Exception as e:
        _status["Prop odds (The Odds API)"] = f"unavailable ({e.__class__.__name__})"
        return []
    _status["Prop odds (The Odds API)"] = f"ok, {r.headers.get('x-requests-remaining', '?')} credits left"
    rows = []
    for bk in r.json().get("bookmakers", []):
        for m in bk.get("markets", []):
            for o in m.get("outcomes", []):
                rows.append({"market": m["key"], "player": o.get("description"),
                             "side": o["name"], "point": o.get("point"),
                             "price": o["price"], "book": bk["title"]})
    return rows


def utcnow():
    return datetime.now(timezone.utc)


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
