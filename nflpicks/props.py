"""Player prop projections and, when book lines exist, edges."""
import numpy as np
import pandas as pd

from . import config
from .model import OUT, QUESTIONABLE, ev, fmt_odds, norm_name

rng = np.random.default_rng(7)

# stat name -> (nflverse column, book market key, label)
STATS = {
    "pass_yds": ("passing_yards", "player_pass_yds", "Pass Yds"),
    "pass_tds": ("passing_tds", "player_pass_tds", "Pass TDs"),
    "rush_yds": ("rushing_yards", "player_rush_yds", "Rush Yds"),
    "rec_yds": ("receiving_yards", "player_reception_yds", "Rec Yds"),
    "receptions": ("receptions", "player_receptions", "Receptions"),
    "anytime_td": (None, "player_anytime_td", "Anytime TD"),
}
POS_STATS = {"QB": ["pass_yds", "pass_tds", "rush_yds", "anytime_td"],
             "RB": ["rush_yds", "rec_yds", "receptions", "anytime_td"],
             "WR": ["rec_yds", "receptions", "anytime_td"],
             "TE": ["rec_yds", "receptions", "anytime_td"]}
# Gamma shape (higher = less spread) for yardage props.
SHAPE = {"pass_yds": 12.0, "rush_yds": 3.2, "rec_yds": 2.2}


def _prep(players, players_prev):
    cols = ["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week",
            "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards", "rushing_tds",
            "targets", "receptions", "receiving_yards", "receiving_tds"]
    df = pd.concat([players[cols], players_prev[cols]], ignore_index=True)
    df = df[df.position.isin(POS_STATS)]
    df[cols[7:]] = df[cols[7:]].fillna(0)
    df["tds"] = df.rushing_tds + df.receiving_tds
    return df


def defense_factors(cur):
    """How much each defense allows vs. league average, by position and stat (regressed)."""
    out = {}
    for pos, stats in (("QB", ["passing_yards", "passing_tds"]), ("RB", ["rushing_yards", "receiving_yards", "receptions"]),
                       ("WR", ["receiving_yards", "receptions"]), ("TE", ["receiving_yards", "receptions"])):
        d = cur[cur.position == pos].groupby(["opponent_team", "week"])[stats].sum().groupby("opponent_team").mean()
        lg = d.mean()
        for t in d.index:
            for s in stats:
                out[(t, pos, s)] = 1 + 0.5 * (d.loc[t, s] / lg[s] - 1) if lg[s] else 1
    return out


def _wmean(vals, cur_mask):
    """Recency-weighted mean: this season's games decay by game, last season is a light prior."""
    w = np.zeros(len(vals))
    n_cur = int(cur_mask.sum())
    if n_cur:
        w[cur_mask] = 0.85 ** np.arange(n_cur)[::-1]
    prev = ~cur_mask
    if prev.any():
        w[prev] = 2.5 / prev.sum()  # whole prior season ~ 2.5 games of weight
    return float((vals * w).sum() / w.sum()) if w.sum() else 0.0


TD_PRIOR = {"QB": 0.12, "RB": 0.35, "WR": 0.28, "TE": 0.22}


def candidates(df, team, season, injuries):
    """Players likely to play meaningful snaps for a team."""
    cur = df[(df.team == team) & (df.season == season)]
    if cur.empty:
        return []
    last_weeks = sorted(cur.week.unique())[-3:]
    recent = cur[cur.week.isin(last_weeks)]
    use = recent.groupby(["player_id", "player_display_name", "position"]).agg(
        att=("attempts", "sum"), car=("carries", "sum"), tgt=("targets", "sum")).reset_index()
    picks = []
    qb = use[use.position == "QB"].sort_values("att", ascending=False).head(1)
    rb = use[use.position == "RB"].assign(u=lambda d: d.car + d.tgt).sort_values("u", ascending=False).head(2)
    rec = use[use.position.isin(["WR", "TE"])].sort_values("tgt", ascending=False).head(4)
    for d in (qb, rb, rec):
        picks.extend(d.to_dict("records"))
    out = []
    for p in picks:
        inj = injuries.get((team, norm_name(p["player_display_name"])))
        status = inj["status"] if inj else None
        if status in OUT:
            continue
        if p["position"] == "RB" and (p["car"] + p["tgt"]) < 12:
            continue
        if p["position"] in ("WR", "TE") and p["tgt"] < 7:
            continue
        p["status"] = status
        out.append(p)
    return out


def project_player(df, p, season, opp, dfac, team_factor, rush_factor, wx_pass):
    h = df[df.player_id == p["player_id"]].sort_values(["season", "week"])
    cur_mask = (h.season == season).to_numpy()
    proj = {}
    pos = p["position"]
    for key in POS_STATS[pos]:
        col = STATS[key][0]
        if key == "anytime_td":
            lam = _wmean(h.tds.to_numpy(float), cur_mask)
            lam = (0.75 * lam + 0.25 * TD_PRIOR[pos]) * team_factor
            proj[key] = {"lambda": lam}
            continue
        m = _wmean(h[col].to_numpy(float), cur_mask)
        base = m  # the player's typical game, before matchup adjustments (books anchor near this)
        stat_for_def = {"pass_yds": "passing_yards", "pass_tds": "passing_tds", "rush_yds": "rushing_yards",
                        "rec_yds": "receiving_yards", "receptions": "receptions"}[key]
        dpos = pos if key != "rush_yds" or pos == "RB" else None
        f = dfac.get((opp, dpos, stat_for_def), 1.0) if dpos else 1.0
        m *= f
        if key in ("pass_yds", "rec_yds", "receptions", "pass_tds"):
            m *= team_factor ** 0.6 * wx_pass
        if key == "rush_yds":
            m *= rush_factor
        proj[key] = {"mean": m, "base": base}
    return proj


def simulate(key, spec):
    if key == "anytime_td":
        return None
    m = max(spec["mean"], 0.01)
    if key in SHAPE:
        k = SHAPE[key]
        return rng.gamma(k, m / k, config.PROP_SIMS)
    return rng.poisson(m, config.PROP_SIMS).astype(float)


def summarize(key, spec, sims):
    if key == "anytime_td":
        p = 1 - np.exp(-spec["lambda"])
        fair = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
        return {"projection": f"{p:.0%} to score", "prob_yes": round(float(p), 3), "fair_odds": fmt_odds(round(fair))}
    med = float(np.median(sims))
    # Book lines where we'd bet: over if the line is at or below `lo`, under if at or above `hi`
    # (each needs a 56%+ model probability).
    grid = np.arange(0.5, float(np.quantile(sims, 0.995)) + 1, 1.0)
    p_over = np.array([(sims > g).mean() for g in grid])
    ok_o, ok_u = grid[p_over >= 0.56], grid[(1 - p_over) >= 0.56]
    return {"projection": round(float(sims.mean()), 1), "median": round(med, 1),
            "over_if_line_at_or_below": float(ok_o.max()) if len(ok_o) else None,
            "under_if_line_at_or_above": float(ok_u.min()) if len(ok_u) else None}


def price_line(key, spec, sims, lines):
    """Given book lines for this player/market, return the best-EV side."""
    best = None
    for ln in lines:
        if key == "anytime_td":
            if ln["side"] not in ("Yes", "Over"):
                continue
            p = 1 - np.exp(-spec["lambda"])
            side = "Yes"
        else:
            if ln["point"] is None:
                continue
            p_over = float((sims > ln["point"]).mean())
            p = p_over if ln["side"] == "Over" else 1 - p_over - float((sims == ln["point"]).mean())
            side = ln["side"]
        e = ev(p, ln["price"])
        if best is None or e > best["ev"]:
            best = {"side": side, "line": ln["point"], "price": fmt_odds(ln["price"]), "book": ln["book"],
                    "prob": round(p, 3), "ev": round(e, 3)}
    if best:
        best["bet"] = best["ev"] >= config.MIN_EV_PROP
    return best


def game_props(df, season, game, injuries, implied, wx_pass, dfac, team_ppg, book_rows):
    """Props for both teams in one game."""
    book = {}
    for r in book_rows or []:
        book.setdefault((norm_name(r["player"]), r["market"]), []).append(r)
    out = []
    for team, opp in ((game["away_team"], game["home_team"]), (game["home_team"], game["away_team"])):
        tf = (implied[team] / team_ppg.get(team, 22.0)) if implied.get(team) else 1.0
        tf = min(max(tf, 0.75), 1.3)
        fav_by = implied[team] - implied[opp] if implied.get(team) and implied.get(opp) else 0
        rush_factor = 1 + 0.012 * fav_by
        for p in candidates(df, team, season, injuries):
            proj = project_player(df, p, season, opp, dfac, tf, rush_factor, wx_pass)
            for key, spec in proj.items():
                sims = simulate(key, spec)
                row = {"player": p["player_display_name"], "team": team, "opp": opp, "pos": p["position"],
                       "status": p.get("status"), "stat": STATS[key][2], "key": key}
                row.update(summarize(key, spec, sims))
                lines = book.get((norm_name(p["player_display_name"]), STATS[key][1]))
                if lines:
                    row["book"] = price_line(key, spec, sims, lines)
                    if p.get("status") in QUESTIONABLE and row["book"]:
                        row["book"]["bet"] = False
                if key != "anytime_td":
                    row["matchup"] = matchup_lean(key, spec, sims)
                out.append(row)
    rank_top(out)
    return out


MIN_BASE = {"pass_yds": 150, "rush_yds": 25, "rec_yds": 25, "receptions": 2.5, "pass_tds": 1.0}


def matchup_lean(key, spec, sims):
    """Where this week's matchup pushes a player away from his usual number.
    The usual number stands in for the book line until real lines are available.
    We pick the half-point line closest to a coin flip for his usual game, then measure how far
    this week's projection moves the odds. Comparing against the usual game (not 50%) keeps
    rounding on small counts like receptions from creating fake edges."""
    base_sims = simulate(key, {"mean": spec["base"]})
    grid = np.arange(0.5, float(np.quantile(base_sims, 0.99)) + 1, 1.0 if key in ("receptions", "pass_tds") else 0.5)
    base_over = np.array([(base_sims > g).mean() for g in grid])
    i = int(np.argmin(np.abs(base_over - 0.5)))
    line = float(grid[i]) if float(grid[i]) % 1 else float(grid[i]) + 0.5
    pb = float((base_sims > line).mean())
    pa = float((sims > line).mean())
    shift = pa - pb
    side = "Over" if shift >= 0 else "Under"
    prob = 0.5 + abs(shift)
    return {"side": side, "line": line, "prob": round(prob, 3),
            "usual": round(spec["base"], 1), "eligible": spec["base"] >= MIN_BASE.get(key, 0)}


def rank_top(rows, n=4):
    """Mark the game's best `n` props: by EV when book lines exist, otherwise by matchup confidence.
    One prop per player so the list isn't four bets on the same guy."""
    def score(r):
        if r.get("status") in QUESTIONABLE:
            return None
        if r.get("book"):
            return 10 + r["book"]["ev"] if r["book"]["ev"] > 0 else None
        m = r.get("matchup")
        if not m or not m["eligible"] or m["prob"] < 0.53:
            return None
        return m["prob"]
    scored = sorted([(score(r), i) for i, r in enumerate(rows) if score(r) is not None], reverse=True)
    used, rank = set(), 0
    for _, i in scored:
        if rows[i]["player"] in used:
            continue
        used.add(rows[i]["player"]); rank += 1
        rows[i]["top"] = rank
        if rank == n:
            break
