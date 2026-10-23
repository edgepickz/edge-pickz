"""Hot streaks: team results against the spread / total, and player props against a line.

A trend only shows up when it's a real streak: at least 3 games and 75%+ hit rate
(3/3, 4/4, 4/5, 5/5, ...). Trends are context, not predictions on their own, so each one
also says whether our model agrees with it."""
import pandas as pd

from .props import STATS

LOOKBACK = 5
MIN_GAMES = 3
MIN_RATE = 0.75


def _streak(hits):
    """hits is oldest->newest list of True/False/None (None = push). Returns summary or None."""
    h = [x for x in hits if x is not None][-LOOKBACK:]
    if len(h) < MIN_GAMES:
        return None
    best = None
    # Check the last 3, 4 and 5 games and keep the most impressive window.
    for n in range(MIN_GAMES, len(h) + 1):
        w = h[-n:]
        k = sum(w)
        rate = k / n
        if rate >= MIN_RATE and (best is None or (k, rate) >= (best["hit"], best["rate"])):
            best = {"hit": k, "of": n, "rate": rate, "dots": w}
    return best


def _team_games(games, team, season):
    g = games[(games.season == season) & games.result.notna() & ((games.home_team == team) | (games.away_team == team))]
    return g.sort_values("gameday")


def team_trends(games, season, team, opp, picks):
    """Spread, total and straight-up streaks for one team."""
    g = _team_games(games, team, season)
    if g.empty:
        return []
    home = (g.home_team == team).to_numpy()
    margin = pd.Series(g.result.to_numpy() * (home * 2 - 1))
    line = pd.Series(g.spread_line.to_numpy() * (home * 2 - 1))  # how many points this team was favored by
    ats = [None if pd.isna(l) or m == l else bool(m > l) for m, l in zip(margin, line)]
    ou = [None if pd.isna(tl) or t == tl else bool(t > tl) for t, tl in zip(g.total, g.total_line)]
    su = [None if m == 0 else bool(m > 0) for m in margin]
    out = []

    spread_pick = (picks.get("spread") or {}).get("team")
    total_pick = (picks.get("total") or {}).get("pick", "")
    ml_pick = (picks.get("moneyline") or {}).get("pick", "")

    def add(kind, side_label, hits, agrees, bet_text):
        s = _streak(hits)
        if s:
            out.append({"type": "team", "team": team, "opp": opp, "kind": kind, "label": side_label,
                        "bet": bet_text, "agrees": agrees, **s})

    add("spread", "covered the spread", ats, spread_pick == team, "spread")
    add("spread", "failed to cover", [None if x is None else not x for x in ats], spread_pick == opp, "spread")
    add("total", "gone over the total", ou, total_pick.startswith("Over"), "over")
    add("total", "gone under the total", [None if x is None else not x for x in ou], total_pick.startswith("Under"), "under")
    add("moneyline", "won", su, ml_pick.startswith(team + " "), "moneyline")
    add("moneyline", "lost", [None if x is None else not x for x in su], ml_pick.startswith(opp + " "), "moneyline")
    return out


def player_trends(pdf, season, prop_rows):
    """For each prop we show, how often the player beat that same line in his recent games."""
    out = []
    cur = pdf[pdf.season == season].sort_values("week")
    for r in prop_rows:
        hist = cur[(cur.player_display_name == r["player"]) & (cur.team == r["team"])]
        if len(hist) < MIN_GAMES:
            continue
        if r["key"] == "anytime_td":
            vals = (hist.rushing_tds + hist.receiving_tds).tolist()
            s = _streak([v > 0 for v in vals])
            if s:
                out.append({"type": "player", "player": r["player"], "team": r["team"], "opp": r["opp"], "pos": r["pos"],
                            "stat": r["stat"], "key": r["key"], "side": "Yes", "line": None,
                            "values": vals[-s["of"]:], "agrees": r.get("prob_yes", 0) >= 0.4,
                            "status": r.get("status"), **s})
            continue
        ref = r.get("book") or r.get("matchup")
        if not ref or ref.get("line") is None:
            continue
        line = ref["line"]
        col = STATS[r["key"]][0]
        vals = hist[col].tolist()
        lean = ref.get("side")
        for side in ("Over", "Under"):
            hits = [None if v == line else (v > line if side == "Over" else v < line) for v in vals]
            s = _streak(hits)
            if s:
                out.append({"type": "player", "player": r["player"], "team": r["team"], "opp": r["opp"], "pos": r["pos"],
                            "stat": r["stat"], "key": r["key"], "side": side, "line": line,
                            "values": [round(v) for v in vals[-s["of"]:]], "agrees": lean == side,
                            "status": r.get("status"), **s})
    return out


def score(t):
    """Sort key: perfect streaks first, longer streaks next, model agreement as a tiebreak."""
    return (t["rate"], t["of"], t["agrees"])
