"""College football: game picks (spread, total, moneyline) for games with an AP Top 25 team.

Everything comes from free sources: ESPN (schedule, rankings, lines, scores, news),
ESPN's odds archive (past lines, for trends) and Open-Meteo (weather).
No player props for college."""
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config, fetch, model, trends

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
CACHE = config.CACHE_DIR / "cfb"
HISTORY_FILE = config.HISTORY_FILE.parent / "cfb_locked.json"

SD_MARGIN = 16.0      # college games are more lopsided and less predictable than NFL
SD_TOTAL = 15.5
HFA = 2.5
HALF_LIFE_WEEKS = 6
TOP = 25
TRUST = 0.25        # lean on the market a bit more than in the NFL: college ratings are noisier
MAX_DOG_ML = 400
_status = {}


def key(team):
    """Unique team key that can't collide with NFL abbreviations."""
    return "c" + str(team["id"])


def _get_json(url, params=None, cache_name=None, max_age_hours=None):
    path = CACHE / cache_name if cache_name else None
    if path and path.exists() and (max_age_hours is None or time.time() - path.stat().st_mtime < max_age_hours * 3600):
        return json.loads(path.read_text())
    js = fetch._get(url, params=params).json()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(js))
    return js


def _num(x):
    try:
        return float(str(x).replace("o", "").replace("u", "").replace("+", ""))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- data
def load_weeks(now):
    """This season's regular-season scoreboards. Finished weeks are cached for good."""
    cur = _get_json(f"{BASE}/scoreboard", {"groups": 80, "limit": 300})
    season = cur["season"]["year"]
    week = cur["week"]["number"]
    seasontype = cur["season"]["type"]
    boards = {}
    for w in range(1, week + 1 if seasontype == 2 else 17):
        live = w >= week - 1
        try:
            boards[w] = _get_json(f"{BASE}/scoreboard", {"groups": 80, "limit": 300, "week": w, "seasontype": 2},
                                  cache_name=f"sb_{season}_{w}.json", max_age_hours=0.1 if live else None)
        except Exception:
            continue
    # Next week too, so picks can be posted 24+ hours ahead.
    try:
        boards[week + 1] = _get_json(f"{BASE}/scoreboard", {"groups": 80, "limit": 300, "week": week + 1, "seasontype": seasontype})
    except Exception:
        pass
    if seasontype != 2:
        boards[100 + week] = cur
    return season, week, boards


def flatten(boards):
    rows, teams = [], {}
    for w, b in boards.items():
        for ev in b.get("events", []):
            c = ev["competitions"][0]
            comp = {x["homeAway"]: x for x in c["competitors"]}
            if "home" not in comp or "away" not in comp:
                continue
            h, a = comp["home"], comp["away"]
            for x in (h, a):
                t = x["team"]
                k = key(t)
                info = teams.setdefault(k, {})
                info.update({"abbr": t.get("abbreviation", "?"), "name": t.get("displayName", "?"),
                             "short": t.get("shortDisplayName") or t.get("location") or t.get("abbreviation"),
                             "color": "#" + (t.get("color") or "555555"), "id": str(t["id"])})
                rank = (x.get("curatedRank") or {}).get("current", 99)
                if w == max(boards):  # latest board has the current poll
                    info["rank"] = rank if rank and rank <= TOP else None
                elif "rank" not in info:
                    info["rank"] = rank if rank and rank <= TOP else None
            st = c["status"]["type"]
            odds = (c.get("odds") or [{}])[0]
            ml = odds.get("moneyline") or {}
            ps = odds.get("pointSpread") or {}
            tot = odds.get("total") or {}
            rows.append({
                "event_id": ev["id"], "week": w, "date": ev["date"],
                "home": key(h["team"]), "away": key(a["team"]),
                "home_score": _num(h.get("score")) if st["completed"] else None,
                "away_score": _num(a.get("score")) if st["completed"] else None,
                "state": st["state"], "completed": st["completed"],
                "neutral": bool(c.get("neutralSite")),
                "venue": (c.get("venue") or {}).get("fullName"),
                "city": ((c.get("venue") or {}).get("address") or {}).get("city"),
                "state_abbr": ((c.get("venue") or {}).get("address") or {}).get("state"),
                "indoor": bool((c.get("venue") or {}).get("indoor")),
                "home_line": odds.get("spread"), "total_line": odds.get("overUnder"),
                "home_ml": _num((ml.get("home") or {}).get("close", {}).get("odds")),
                "away_ml": _num((ml.get("away") or {}).get("close", {}).get("odds")),
                "home_spread_odds": _num((ps.get("home") or {}).get("close", {}).get("odds")),
                "away_spread_odds": _num((ps.get("away") or {}).get("close", {}).get("odds")),
                "over_odds": _num((tot.get("over") or {}).get("close", {}).get("odds")),
                "under_odds": _num((tot.get("under") or {}).get("close", {}).get("odds")),
                "book": (odds.get("provider") or {}).get("name"),
            })
    df = pd.DataFrame(rows).drop_duplicates("event_id", keep="last")
    df["kickoff"] = pd.to_datetime(df.date, utc=True)
    return df, teams


def past_lines(df, team_keys):
    """Closing spread/total for finished games (ESPN's odds archive), cached forever."""
    done = df[df.completed & (df.home.isin(team_keys) | df.away.isin(team_keys))]
    for r in done.itertuples():
        if pd.notna(r.home_line) and pd.notna(r.total_line):
            continue
        try:
            js = _get_json(f"{CORE}/events/{r.event_id}/competitions/{r.event_id}/odds", cache_name=f"odds/{r.event_id}.json")
        except Exception:
            continue
        it = (js.get("items") or [{}])[0]
        if it.get("spread") is not None:
            df.loc[df.event_id == r.event_id, "home_line"] = it.get("spread")
        if it.get("overUnder") is not None:
            df.loc[df.event_id == r.event_id, "total_line"] = it.get("overUnder")
    return df


# ---------------------------------------------------------------- model
def ratings(df, now):
    """Weighted ridge fit on points, like the NFL model. Teams with few games
    in our data (mostly FCS opponents) share one pooled rating."""
    done = df[df.completed].copy()
    # ESPN's FBS scoreboard only lists games with an FBS team in them, so FBS teams show up
    # every week while FCS opponents show up once or twice.
    counts = pd.concat([done.home, done.away]).value_counts()
    weeks_played = done.week[done.week < 100].nunique()
    fbs = set(counts[counts >= max(2, min(4, weeks_played // 2))].index)
    pool = lambda t: t if t in fbs else "FCS"
    done["h"] = done.home.map(pool)
    done["a"] = done.away.map(pool)
    teams = sorted(set(done.h) | set(done.a))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    weeks_ago = (pd.Timestamp(now) - done.kickoff).dt.days.clip(lower=0) / 7
    w = (0.5 ** (weeks_ago / HALF_LIFE_WEEKS)).to_numpy()
    X, y, wt = [], [], []
    for r, wi in zip(done.itertuples(), w):
        hf = 0 if r.neutral else HFA / 2
        for team, opp, pts, sign in ((r.h, r.a, r.home_score, 1), (r.a, r.h, r.away_score, -1)):
            x = np.zeros(2 + 2 * n)
            x[0] = 1
            x[2 + idx[team]] = 1
            x[2 + n + idx[opp]] = -1
            X.append(x); y.append(pts - sign * hf); wt.append(wi)
    X, y, sw = np.array(X), np.array(y), np.sqrt(np.array(wt))
    reg = np.sqrt(3.0) * np.eye(X.shape[1]); reg[0, 0] = reg[1, 1] = 0
    beta = np.linalg.lstsq(np.vstack([X * sw[:, None], reg]), np.concatenate([y * sw, np.zeros(X.shape[1])]), rcond=None)[0]
    off = {t: beta[2 + idx[t]] for t in teams}
    dfn = {t: beta[2 + n + idx[t]] for t in teams}
    hfa = HFA
    rate = lambda t: (off.get(pool(t), off.get("FCS", -10)), dfn.get(pool(t), dfn.get("FCS", -10)))
    rate.debug = {"n_games": int(len(done)), "n_fbs": len(fbs), "n_teams": n, "mu": float(beta[0]), "hfa": float(beta[1]),
                  "top": sorted(((t, round(off[t] + dfn[t], 1)) for t in teams), key=lambda x: -x[1])[:40]}
    return rate, beta[0], hfa


def project(home, away, neutral, rate, mu, hfa):
    oh, dh = rate(home)
    oa, da = rate(away)
    h = 0 if neutral else hfa
    return mu + oh - da + h / 2, mu + oa - dh - h / 2


# ---------------------------------------------------------------- weather
_geo = {}


def coords(city, state):
    k = f"{city},{state}"
    if k in _geo:
        return _geo[k]
    try:
        js = _get_json("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 5, "country": "US"},
                       cache_name=f"geo/{city}_{state}.json".replace(" ", "_"))
        res = js.get("results") or []
        pick = next((r for r in res if state and state.upper() in (r.get("admin1_code", ""), _STATE.get(r.get("admin1", ""), ""))), res[0] if res else None)
        _geo[k] = (pick["latitude"], pick["longitude"]) if pick else None
    except Exception:
        _geo[k] = None
    return _geo[k]


_STATE = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
          "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
          "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
          "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
          "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
          "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
          "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
          "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
          "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
          "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"}


# ---------------------------------------------------------------- news
def news_by_team():
    try:
        js = _get_json(f"{BASE}/news", {"limit": 100})
    except Exception as e:
        _status["ESPN college news"] = f"unavailable ({e.__class__.__name__})"
        return {}
    out = {}
    for a in js.get("articles", []):
        ids = {str(c.get("teamId")) for c in a.get("categories", []) if c.get("type") == "team" and c.get("teamId")}
        if not ids or len(ids) > 4:  # skip big roundups that mention 20 teams
            continue
        item = {"headline": a.get("headline"), "link": ((a.get("links") or {}).get("web") or {}).get("href")}
        for i in ids:
            out.setdefault("c" + i, []).append(item)
    _status["ESPN college news"] = "ok"
    return out


# ---------------------------------------------------------------- grading / history
def grade(history, df):
    res = df.set_index("event_id")
    for eid, h in history.items():
        if eid not in res.index or not res.loc[eid, "completed"]:
            continue
        r = res.loc[eid]
        hs, as_ = float(r.home_score), float(r.away_score)
        margin, total = hs - as_, hs + as_
        h["final"] = f"{h['away_abbr']} {int(as_)} - {h['home_abbr']} {int(hs)}"
        for k, p in h["picks"].items():
            if k == "spread":
                v = (margin if p["team"] == h["home"] else -margin) + p["line"]
            elif k == "total":
                line = float(p["pick"].split()[1])
                v = (total - line) if p["pick"].startswith("Over") else (line - total)
            else:
                v = 0 if margin == 0 else (1 if (p["pick"].startswith(h["home"] + " ") == (margin > 0)) else -1)
            p["result"] = "W" if v > 0 else "L" if v < 0 else "P"
    return history


def as_trend_frame(df):
    """Reshape into the columns trends.team_trends expects (nflverse-style)."""
    d = df[df.completed].copy()
    return pd.DataFrame({
        "season": 0, "home_team": d.home, "away_team": d.away, "gameday": d.kickoff,
        "result": d.home_score - d.away_score, "total": d.home_score + d.away_score,
        "spread_line": -d.home_line.astype(float), "total_line": d.total_line.astype(float),
    })


# ---------------------------------------------------------------- main
def build(now):
    _status.clear()
    try:
        season, week, boards = load_weeks(now)
        df, teams = flatten(boards)
        _status["ESPN college schedule, rankings and lines"] = "ok"
    except Exception as e:
        _status["ESPN college schedule, rankings and lines"] = f"unavailable ({e.__class__.__name__})"
        return {"games": [], "teams": {}, "sources": dict(_status), "record": {}, "recent_results": [], "tracker_recent": []}

    ranked = {k for k, t in teams.items() if t.get("rank")}
    upcoming = df[(~df.completed) & (df.kickoff > now - timedelta(hours=5)) & (df.kickoff < now + timedelta(days=config.LOOKAHEAD_DAYS))
                  & (df.home.isin(ranked) | df.away.isin(ranked))].sort_values("kickoff")
    df = past_lines(df, set(upcoming.home) | set(upcoming.away))
    rate, mu, hfa = ratings(df, now)
    tf = as_trend_frame(df)
    news = news_by_team()

    history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else {}
    games = []
    for r in upcoming.itertuples():
        ko = r.kickoff.to_pydatetime()
        mh, ma = project(r.home, r.away, r.neutral, rate, mu, hfa)
        wx, wnotes = None, []
        if not r.indoor and r.city:
            c = coords(r.city, r.state_abbr)
            if c:
                wx = fetch.weather_at(*c, ko.replace(minute=0, second=0, microsecond=0))
            wadj, wnotes = model.weather_total_adj(wx, "outdoors")
            mh += wadj / 2; ma += wadj / 2
        g = {"home_team": r.home, "away_team": r.away,
             "spread_line": -r.home_line if pd.notna(r.home_line) else float("nan"),
             "total_line": r.total_line if pd.notna(r.total_line) else float("nan"),
             "home_moneyline": r.home_ml, "away_moneyline": r.away_ml,
             "home_spread_odds": r.home_spread_odds, "away_spread_odds": r.away_spread_odds,
             "over_odds": r.over_odds, "under_odds": r.under_odds}
        picks = model.game_picks(g, mh, ma, {}, sd_margin=SD_MARGIN, sd_total=SD_TOTAL, trust=TRUST)
        ml = picks.get("moneyline")
        if ml and float(ml["price"]) > MAX_DOG_ML:  # long-shot moneylines: too swingy to call a bet
            ml["bet"], ml["stars"] = False, 0
        tr = sorted(trends.team_trends(tf, 0, r.away, r.home, picks) + trends.team_trends(tf, 0, r.home, r.away, picks),
                    key=trends.score, reverse=True)
        lock_at = ko - timedelta(hours=config.LOCK_HOURS)
        hn, an = teams[r.home], teams[r.away]
        if r.event_id not in history and lock_at <= now < ko and picks:
            history[r.event_id] = {"locked_at": now.isoformat(), "kickoff": ko.isoformat(), "week": int(r.week),
                                   "home": r.home, "away": r.away, "home_abbr": hn["abbr"], "away_abbr": an["abbr"],
                                   "matchup": f"{an['abbr']} @ {hn['abbr']}", "espn_id": r.event_id,
                                   "picks": {k: dict(v) for k, v in picks.items()}}
        games.append({
            "league": "cfb", "game_id": "cfb-" + r.event_id, "espn_id": r.event_id, "week": int(r.week),
            "away": r.away, "home": r.home, "away_name": an["name"], "home_name": hn["name"],
            "away_rank": an.get("rank"), "home_rank": hn.get("rank"),
            "kickoff": ko.isoformat(), "lock_at": lock_at.isoformat(),
            "stadium": r.venue or "", "roof": "dome" if r.indoor else "outdoors", "neutral": bool(r.neutral),
            "qbs": {}, "model_score": {r.away: round(ma, 1), r.home: round(mh, 1)},
            "picks": picks, "props": [], "weather": wx, "weather_notes": wnotes,
            "injury_adj": {}, "injuries": [], "news": news.get(r.home, [])[:2] + news.get(r.away, [])[:2],
            "trends": tr, "book": r.book, "locked": history.get(r.event_id),
        })
    history = grade(history, df)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=1, default=str))
    for gm in games:
        gm["locked"] = history.get(gm["espn_id"])

    shown = {gm["espn_id"] for gm in games}
    recent = []
    for eid, h in history.items():
        ko = datetime.fromisoformat(h["kickoff"])
        if eid in shown or not (now - timedelta(days=5) <= ko <= now):
            continue
        recent.append({"league": "cfb", "game_id": "cfb-" + eid, "espn_id": eid, "away": h["away"], "home": h["home"],
                       "away_name": teams.get(h["away"], {}).get("name", h["away_abbr"]),
                       "home_name": teams.get(h["home"], {}).get("name", h["home_abbr"]),
                       "kickoff": h["kickoff"], "week": h["week"], "final": h.get("final"), "picks": h["picks"], "props": []})
    used = {gm["home"] for gm in games} | {gm["away"] for gm in games} | {h["home"] for h in history.values()} | {h["away"] for h in history.values()}
    _status["Odds (DraftKings via ESPN)"] = "ok" if any(gm["picks"] for gm in games) else "no lines posted yet"
    return {
        "season": season, "week": week, "games": games, "debug": getattr(rate, "debug", None),
        "teams": {k: v for k, v in teams.items() if k in used},
        "sources": dict(_status),
        "record": _record(history),
        "recent_results": sorted([dict(v, game_id=k) for k, v in history.items() if v.get("final")],
                                 key=lambda v: v["kickoff"], reverse=True)[:16],
        "tracker_recent": sorted(recent, key=lambda g: g["kickoff"], reverse=True),
    }


def _record(history):
    rec = {}
    for h in history.values():
        for k, p in h["picks"].items():
            if not p.get("bet") or p.get("result") not in ("W", "L", "P"):
                continue
            r = rec.setdefault(k, {"W": 0, "L": 0, "P": 0, "units": 0.0})
            r[p["result"]] += 1
            if p["result"] == "W":
                r["units"] += model.american_to_decimal(float(p["price"])) - 1
            elif p["result"] == "L":
                r["units"] -= 1
    for r in rec.values():
        r["units"] = round(r["units"], 2)
    return rec
