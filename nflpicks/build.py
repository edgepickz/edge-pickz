"""Main entry point: python -m nflpicks.build

Pulls fresh data, re-runs the model, locks picks for games that are
24 hours or less from kickoff, grades finished games, and writes the site."""
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, fetch, model, props, render, trends

ET = ZoneInfo("America/New_York")


def kickoff_utc(r):
    dt = datetime.strptime(f"{r.gameday} {r.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    return dt.astimezone(timezone.utc)


def current_season(now):
    return now.year if now.month >= 3 else now.year - 1


def game_coords(r):
    if r.location == "Neutral":
        for k, v in config.VENUE_COORDS.items():
            if k.lower() in str(r.stadium).lower():
                return v
    return config.TEAM_COORDS.get(r.home_team)


def load_history():
    if config.HISTORY_FILE.exists():
        return json.loads(config.HISTORY_FILE.read_text())
    return {}


def grade(history, games, players):
    res = games.set_index("game_id")
    pstats = players.set_index(["game_id", "player_display_name"]) if len(players) else None
    for gid, h in history.items():
        if gid not in res.index or pd.isna(res.loc[gid, "result"]):
            continue
        r = res.loc[gid]
        margin, total = float(r.result), float(r.total)
        h["final"] = f"{r.away_team} {int(r.away_score)} - {r.home_team} {int(r.home_score)}"
        for k, p in h["picks"].items():
            if k == "spread":
                m = margin if p["team"] == r.home_team else -margin
                v = m + p["line"]
            elif k == "total":
                line = float(p["pick"].split()[1])
                v = (total - line) if p["pick"].startswith("Over") else (line - total)
            else:
                won_home = margin > 0
                v = 1 if (p["pick"].startswith(r.home_team + " ") == won_home and margin != 0) else -1
                if margin == 0:
                    v = 0
            p["result"] = "W" if v > 0 else "L" if v < 0 else "P"
        for p in h.get("props", []):
            b = p.get("book")
            if not b or pstats is None:
                continue
            try:
                s = pstats.loc[(gid, p["player"])]
            except KeyError:
                p["result"] = "Void"; continue
            col = {"pass_yds": "passing_yards", "pass_tds": "passing_tds", "rush_yds": "rushing_yards",
                   "rec_yds": "receiving_yards", "receptions": "receptions"}.get(p["key"])
            if p["key"] == "anytime_td":
                v = (s.rushing_tds or 0) + (s.receiving_tds or 0) - 0.5
            else:
                v = (s[col] or 0) - b["line"]
                v = v if b["side"] == "Over" else -v
            p["result"] = "W" if v > 0 else "L" if v < 0 else "P"
    return history


def record(history):
    rec = {}
    for h in history.values():
        items = [(k, p) for k, p in h["picks"].items() if p.get("bet")]
        items += [("props", p["book"] | {"result": p.get("result")}) for p in h.get("props", [])
                  if p.get("book", {}).get("bet")]
        for k, p in items:
            res = p.get("result")
            if res not in ("W", "L", "P"):
                continue
            r = rec.setdefault(k, {"W": 0, "L": 0, "P": 0, "units": 0.0})
            r[res] += 1
            if res == "W":
                r["units"] += model.american_to_decimal(float(p["price"])) - 1
            elif res == "L":
                r["units"] -= 1
    for r in rec.values():
        r["units"] = round(r["units"], 2)
    return rec


def main():
    now = datetime.fromisoformat(os.environ["NFLPICKS_NOW"]) if os.environ.get("NFLPICKS_NOW") else fetch.utcnow()
    season = current_season(now)
    data = fetch.load_nflverse(season)
    games = data["games"]
    games = games[games.game_type.notna() & games.gametime.notna()].copy()
    graded_games = games
    # Hide results of games that hadn't kicked off yet at `now` (matters when NFLPICKS_NOW replays a past date).
    games = games.copy()
    ko = pd.Series([kickoff_utc(r) for r in games.itertuples()], index=games.index)
    future = ko > now
    games.loc[future, ["result", "total", "home_score", "away_score"]] = float("nan")
    players = data["players"][~data["players"].game_id.isin(games.loc[future, "game_id"])]

    pr = model.points_ratings(games, now, [season - 1, season])
    er = model.epa_ratings([data["teams_prev"], data["teams"]], games[games.result.notna()], now)
    ppg = {}
    cur = games[(games.season == season) & games.result.notna()]
    for t in config.TEAM_NAMES:
        pts = pd.concat([cur[cur.home_team == t].home_score, cur[cur.away_team == t].away_score])
        ppg[t] = float(pts.mean()) if len(pts) else 22.0

    upcoming = games[(games.season == season) & games.result.isna()].copy()
    upcoming["kickoff"] = [kickoff_utc(r) for r in upcoming.itertuples()]
    upcoming = upcoming[(upcoming.kickoff > now - timedelta(hours=4)) &
                        (upcoming.kickoff < now + timedelta(days=config.LOOKAHEAD_DAYS))]

    espn_inj = fetch.espn_injuries()
    news = fetch.espn_news()
    history = load_history()

    lock_due = [r.game_id for r in upcoming.itertuples()
                if r.game_id not in history and r.kickoff - timedelta(hours=config.LOCK_HOURS) <= now < r.kickoff]
    live = fetch.odds_game_lines() if lock_due else {}
    if not lock_due:
        fetch._status.setdefault("Live odds (The Odds API)",
                                 "not needed this run (only pulled when a game reaches its 24h lock)"
                                 if config.ODDS_API_KEY else "no API key set, using nflverse consensus lines")

    pdf = props._prep(players, data["players_prev"])
    dfac = props.defense_factors(pdf[pdf.season == season])

    out_games = []
    for r in upcoming.sort_values("kickoff").itertuples():
        g = r._asdict()
        neutral = r.location == "Neutral"
        inj = model.injury_table(data["injuries"], espn_inj, season, r.week)
        mh, ma = model.project_points(r.home_team, r.away_team, neutral, pr, er)
        adj_h, notes_h = model.injury_adjustment(r.home_team, inj, players)
        adj_a, notes_a = model.injury_adjustment(r.away_team, inj, players)
        mh += adj_h; ma += adj_a
        wx = None
        if r.roof not in config.INDOOR_ROOFS and isinstance(r.roof, str):
            c = game_coords(r)
            wx = fetch.weather_at(*c, r.kickoff) if c else None
        wadj, wnotes = model.weather_total_adj(wx, r.roof)
        mh += wadj / 2; ma += wadj / 2
        wx_pass = 1 - 0.012 * max(0, (wx or {}).get("wind_mph", 0) - 12)

        lv = live.get((r.away_team, r.home_team), {})
        picks = model.game_picks(g, mh, ma, lv.get("books", {}))
        spread, total = g["spread_line"], g["total_line"]
        implied = {}
        if pd.notna(spread) and pd.notna(total):
            implied = {r.home_team: total / 2 + spread / 2, r.away_team: total / 2 - spread / 2}
        book_rows = []
        if r.game_id in lock_due and lv.get("event_id"):
            book_rows = fetch.odds_event_props(lv["event_id"])
        prop_rows = props.game_props(pdf, season, g, inj, implied, wx_pass, dfac, ppg, book_rows)

        lock_at = r.kickoff - timedelta(hours=config.LOCK_HOURS)
        entry = {
            "game_id": r.game_id, "week": int(r.week),
            "away": r.away_team, "home": r.home_team,
            "away_name": config.TEAM_NAMES[r.away_team], "home_name": config.TEAM_NAMES[r.home_team],
            "kickoff": r.kickoff.isoformat(), "lock_at": lock_at.isoformat(),
            "stadium": r.stadium, "roof": r.roof if isinstance(r.roof, str) else "retractable",
            "neutral": neutral,
            "qbs": {r.away_team: r.away_qb_name, r.home_team: r.home_qb_name},
            "model_score": {r.away_team: round(ma, 1), r.home_team: round(mh, 1)},
            "picks": picks, "props": prop_rows, "weather": wx, "weather_notes": wnotes,
            "injury_notes": {r.away_team: notes_a, r.home_team: notes_h},
            "injury_adj": {r.away_team: round(adj_a, 1), r.home_team: round(adj_h, 1)},
            "injuries": [v for (t, _), v in inj.items() if t in (r.home_team, r.away_team)
                         and v["status"] in model.OUT | model.QUESTIONABLE],
            "news": [n for n in news if set(n["teams"]) & {r.home_team, r.away_team}][:5],
            "trends": sorted(trends.team_trends(games, season, r.away_team, r.home_team, picks)
                             + trends.team_trends(games, season, r.home_team, r.away_team, picks)
                             + trends.player_trends(pdf, season, prop_rows), key=trends.score, reverse=True),
        }
        if r.game_id in lock_due:
            history[r.game_id] = {
                "locked_at": now.isoformat(), "matchup": f"{r.away_team} @ {r.home_team}",
                "kickoff": r.kickoff.isoformat(), "week": int(r.week),
                "picks": {k: dict(v) for k, v in picks.items()},
                "props": [p for p in prop_rows if p.get("book")],
            }
        out_games.append(entry)

    history = grade(history, graded_games, data["players"])
    config.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.HISTORY_FILE.write_text(json.dumps(history, indent=1, default=str))

    for e in out_games:
        e["locked"] = history.get(e["game_id"])

    payload = {
        "generated_at": now.isoformat(), "season": season,
        "sources": fetch.source_status(), "games": out_games,
        "record": record(history),
        "recent_results": sorted(
            [dict(v, game_id=k) for k, v in history.items() if v.get("final")],
            key=lambda v: v["kickoff"], reverse=True)[:16],
        "settings": {"lock_hours": config.LOCK_HOURS, "model_trust": config.MODEL_TRUST,
                     "min_ev_game": config.MIN_EV_GAME, "min_ev_prop": config.MIN_EV_PROP,
                     "live_odds": bool(config.ODDS_API_KEY)},
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "picks.json").write_text(json.dumps(payload, indent=1, default=str))
    (config.OUT_DIR / "index.html").write_text(render.page(payload))
    (config.OUT_DIR / "artifact.html").write_text(render.fragment(payload))
    print(f"Wrote {len(out_games)} games, {len(lock_due)} newly locked. Sources: {payload['sources']}")


if __name__ == "__main__":
    main()
