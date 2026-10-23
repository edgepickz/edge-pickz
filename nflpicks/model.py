"""Team ratings and game picks (spread, total, moneyline)."""
import math
import re

import numpy as np
import pandas as pd

from . import config

TEAMS = sorted(config.TEAM_NAMES)
IDX = {t: i for i, t in enumerate(TEAMS)}


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def american_to_decimal(odds):
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def implied_prob(odds):
    return 1 / american_to_decimal(odds)


def ev(prob, odds):
    """Expected profit per $1 staked."""
    return prob * (american_to_decimal(odds) - 1) - (1 - prob)


def fmt_odds(o):
    return f"+{int(o)}" if o > 0 else str(int(o))


def norm_name(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())


# ---------------------------------------------------------------- ratings
def _weights(dates, now):
    weeks_ago = (pd.Timestamp(now).tz_localize(None) - pd.to_datetime(pd.Series(dates).reset_index(drop=True))).dt.days.clip(lower=0) / 7
    return 0.5 ** (weeks_ago / config.RATING_HALF_LIFE_WEEKS)


def points_ratings(games, now, seasons):
    """Weighted ridge fit: pts = mu + off[team] - def[opp] +/- hfa/2."""
    g = games[games.season.isin(seasons) & games.result.notna()].copy()
    w = _weights(g.gameday, now).to_numpy()
    n = len(TEAMS)
    rows, y, wt = [], [], []
    for (_, r), wi in zip(g.iterrows(), w):
        h = 0 if r.location == "Neutral" else 1
        for team, opp, pts, sign in ((r.home_team, r.away_team, r.home_score, 1),
                                     (r.away_team, r.home_team, r.away_score, -1)):
            x = np.zeros(2 + 2 * n)
            x[0] = 1
            x[1] = 0.5 * sign * h
            x[2 + IDX[team]] = 1
            x[2 + n + IDX[opp]] = -1
            rows.append(x); y.append(pts); wt.append(wi)
    X, y, wt = np.array(rows), np.array(y), np.array(wt)
    lam = 2.0
    sw = np.sqrt(wt)
    Xw, yw = X * sw[:, None], y * sw
    reg = np.sqrt(lam) * np.eye(X.shape[1])
    reg[0, 0] = reg[1, 1] = 0  # don't shrink intercept / home field
    beta = np.linalg.lstsq(np.vstack([Xw, reg]), np.concatenate([yw, np.zeros(X.shape[1])]), rcond=None)[0]
    return {
        "mu": beta[0], "hfa": beta[1],
        "off": {t: beta[2 + IDX[t]] for t in TEAMS},
        "def": {t: beta[2 + n + IDX[t]] for t in TEAMS},
    }


def epa_ratings(team_frames, games, now):
    """Recency-weighted offensive EPA/play and defensive EPA/play allowed."""
    tf = pd.concat(team_frames, ignore_index=True)
    tf = tf[tf.season_type == "REG"] if "season_type" in tf else tf
    dates = games.set_index("game_id").gameday
    tf = tf[tf.game_id.isin(dates.index)].copy()
    tf["date"] = tf.game_id.map(dates)
    tf["w"] = _weights(tf.date, now).to_numpy()
    tf["plays"] = tf.attempts.fillna(0) + tf.carries.fillna(0) + tf.sacks_suffered.fillna(0)
    tf["epa"] = tf.passing_epa.fillna(0) + tf.rushing_epa.fillna(0)
    lg = (tf.epa * tf.w).sum() / (tf.plays * tf.w).sum()
    off = tf.groupby("team").apply(lambda d: (d.epa * d.w).sum() / (d.plays * d.w).sum(), include_groups=False)
    dfn = tf.groupby("opponent_team").apply(lambda d: (d.epa * d.w).sum() / (d.plays * d.w).sum(), include_groups=False)
    plays = tf.groupby("team").apply(lambda d: (d.plays * d.w).sum() / d.w.sum(), include_groups=False)
    return {"lg": lg, "off": off.to_dict(), "def": dfn.to_dict(), "plays": plays.to_dict()}


def project_points(home, away, neutral, pr, er):
    """Blend points-based and EPA-based projections."""
    hfa = 0 if neutral else 0.5 * pr["hfa"] + 0.5 * config.HFA
    ph = pr["mu"] + pr["off"][home] - pr["def"][away] + hfa / 2
    pa = pr["mu"] + pr["off"][away] - pr["def"][home] - hfa / 2
    shrink = 0.6
    lg = er["lg"]
    plays = 62
    eh = pr["mu"] + hfa / 2 + shrink * plays * ((er["off"].get(home, lg) - lg) + (er["def"].get(away, lg) - lg))
    ea = pr["mu"] - hfa / 2 + shrink * plays * ((er["off"].get(away, lg) - lg) + (er["def"].get(home, lg) - lg))
    return 0.5 * ph + 0.5 * eh, 0.5 * pa + 0.5 * ea


# ---------------------------------------------------------------- injuries
OUT = {"Out", "Doubtful", "Injured Reserve", "IR", "Suspension", "PUP"}
QUESTIONABLE = {"Questionable"}


def injury_table(nfl_inj, espn_inj, season, week):
    """Merge nflverse (official report) with ESPN (fresher). ESPN wins on conflicts."""
    rows = {}
    if nfl_inj is not None and len(nfl_inj):
        cur = nfl_inj[(nfl_inj.season == season) & (nfl_inj.week == week)]
        for r in cur.itertuples():
            if isinstance(r.report_status, str):
                rows[(r.team, norm_name(r.full_name))] = {
                    "team": r.team, "name": r.full_name, "position": r.position,
                    "status": r.report_status,
                    "detail": r.report_primary_injury if isinstance(r.report_primary_injury, str) else "",
                    "source": "NFL report"}
    for team, lst in (espn_inj or {}).items():
        for p in lst:
            if not p.get("name") or not p.get("status"):
                continue
            rows[(team, norm_name(p["name"]))] = {
                "team": team, "name": p["name"], "position": p.get("position"),
                "status": p["status"], "detail": p.get("detail", ""), "source": "ESPN"}
    return rows


def player_values(players, team):
    """Points-per-game value of each player on a team, from recent EPA."""
    d = players[players.team == team]
    if d.empty:
        return {}, None
    recent_weeks = sorted(d.week.unique())[-3:]
    qbs = d[(d.position == "QB") & d.week.isin(recent_weeks)]
    starter = None
    if not qbs.empty:
        starter = qbs.groupby("player_display_name").attempts.sum().idxmax()
    vals = {}
    for name, p in d.groupby("player_display_name"):
        pos = p.position.iloc[0]
        games = p.week.nunique()
        if pos == "QB":
            plays = (p.attempts.fillna(0) + p.carries.fillna(0) + p.sacks_suffered.fillna(0)).sum()
            epa_pp = (p.passing_epa.fillna(0) + p.rushing_epa.fillna(0)).sum() / max(plays, 1)
            v = min(max(35 * (epa_pp + 0.12), 1.5), 8.0) * 0.8 if name == starter else 0.5
        elif pos in ("RB", "WR", "TE"):
            epa = (p.rushing_epa.fillna(0) + p.receiving_epa.fillna(0)).sum() / max(games, 1)
            touches = (p.carries.fillna(0) + p.targets.fillna(0)).sum() / max(games, 1)
            v = min(max(0.4 * epa, 0) + 0.04 * touches, 1.5)
        else:
            v = 0.15
        vals[norm_name(name)] = (round(v, 2), pos)
    return vals, starter


def injury_adjustment(team, inj_rows, players):
    vals, starter = player_values(players, team)
    total, notes = 0.0, []
    for (t, key), r in inj_rows.items():
        if t != team:
            continue
        status = r["status"]
        if status in OUT:
            mult = 1.0
        elif status in QUESTIONABLE:
            mult = 0.3
        else:
            continue
        v, pos = vals.get(key, (0.1, r.get("position")))
        total += v * mult
        if v * mult >= 0.25 or pos == "QB":
            notes.append(f"{r['name']} ({r.get('position') or pos}) {status}")
    return -min(total, 9.0), notes


# ---------------------------------------------------------------- weather
def weather_total_adj(wx, roof):
    if roof in config.INDOOR_ROOFS or wx is None:
        return 0.0, []
    adj, notes = 0.0, []
    if wx["wind_mph"] >= 12:
        a = 0.35 * (wx["wind_mph"] - 12) + (0.5 if wx["gust_mph"] >= 30 else 0)
        adj -= a; notes.append(f"wind {wx['wind_mph']} mph (gusts {wx['gust_mph']})")
    if wx["precip_prob"] >= 50 and wx["precip_in"] >= 0.05:
        adj -= 1.5; notes.append(f"{wx['precip_prob']}% chance of rain/snow")
    if wx["temp_f"] <= 25:
        adj -= 1.0; notes.append(f"{wx['temp_f']}°F")
    return adj, notes


# ---------------------------------------------------------------- picks
def game_picks(g, model_home, model_away, live, sd_margin=None, sd_total=None):
    """Compare the model to the market for spread, total and moneyline."""
    sd_m = sd_margin or config.SD_MARGIN
    sd_t = sd_total or config.SD_TOTAL
    out = {}
    market_spread = g["spread_line"]          # >0 means home favored by that many
    market_total = g["total_line"]
    model_margin = model_home - model_away
    model_total = model_home + model_away

    # Market lines and prices (best available if live odds are on).
    sp_home_price = live.get(("spreads", g["home_team"]), {}).get("price") or g.get("home_spread_odds") or -110
    sp_away_price = live.get(("spreads", g["away_team"]), {}).get("price") or g.get("away_spread_odds") or -110
    if live.get(("spreads", g["home_team"])):
        market_spread = -live[("spreads", g["home_team"])]["point"]
    over_price = live.get(("totals", "Over"), {}).get("price") or g.get("over_odds") or -110
    under_price = live.get(("totals", "Under"), {}).get("price") or g.get("under_odds") or -110
    if live.get(("totals", "Over")):
        market_total = live[("totals", "Over")]["point"]
    ml_home = live.get(("h2h", g["home_team"]), {}).get("price") or g.get("home_moneyline")
    ml_away = live.get(("h2h", g["away_team"]), {}).get("price") or g.get("away_moneyline")

    def clean(x, d=-110):
        return d if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    sp_home_price, sp_away_price = clean(sp_home_price), clean(sp_away_price)
    over_price, under_price = clean(over_price), clean(under_price)

    # Spread
    if market_spread is not None and not math.isnan(market_spread):
        est_margin = market_spread + config.MODEL_TRUST * (model_margin - market_spread)
        p_home = 1 - phi((market_spread - est_margin) / sd_m)
        ev_home, ev_away = ev(p_home, sp_home_price), ev(1 - p_home, sp_away_price)
        home_side = ev_home >= ev_away
        team = g["home_team"] if home_side else g["away_team"]
        line = -market_spread if home_side else market_spread
        out["spread"] = {
            "market": f"{g['home_team']} {_signed(-market_spread)}",
            "model_line": f"{g['home_team']} {_signed(-round(model_margin * 2) / 2)}",
            "edge_pts": round(abs(model_margin - market_spread), 1),
            "pick": f"{team} {_signed(line)}", "team": team, "line": line,
            "price": fmt_odds(sp_home_price if home_side else sp_away_price),
            "prob": round(p_home if home_side else 1 - p_home, 3),
            "ev": round(max(ev_home, ev_away), 3),
        }
    # Total
    if market_total is not None and not math.isnan(market_total):
        est_total = market_total + config.MODEL_TRUST * (model_total - market_total)
        p_over = 1 - phi((market_total - est_total) / sd_t)
        ev_o, ev_u = ev(p_over, over_price), ev(1 - p_over, under_price)
        over = ev_o >= ev_u
        out["total"] = {
            "market": f"{market_total:g}", "model_line": f"{model_total:.1f}",
            "edge_pts": round(abs(model_total - market_total), 1),
            "pick": f"{'Over' if over else 'Under'} {market_total:g}",
            "price": fmt_odds(over_price if over else under_price),
            "prob": round(p_over if over else 1 - p_over, 3),
            "ev": round(max(ev_o, ev_u), 3),
        }
    # Moneyline
    if ml_home is not None and ml_away is not None and not (isinstance(ml_home, float) and math.isnan(ml_home)):
        base = market_spread if market_spread is not None and not math.isnan(market_spread) else model_margin
        est_margin = base + config.MODEL_TRUST * (model_margin - base)
        ih, ia = implied_prob(ml_home), implied_prob(ml_away)
        # Start from the market's no-vig win probability and move it by the model's edge,
        # so the bell-curve approximation doesn't create fake value on big underdogs.
        p_mkt = ih / (ih + ia)
        p_home = p_mkt + phi(est_margin / sd_m) - phi(base / sd_m)
        p_home = min(max(p_home, 0.01), 0.99)
        ev_h, ev_a = ev(p_home, ml_home), ev(1 - p_home, ml_away)
        home_side = ev_h >= ev_a
        out["moneyline"] = {
            "market": f"{g['away_team']} {fmt_odds(ml_away)} / {g['home_team']} {fmt_odds(ml_home)}",
            "market_prob": round((ih if home_side else ia) / (ih + ia), 3),
            "pick": f"{g['home_team'] if home_side else g['away_team']} ML",
            "price": fmt_odds(ml_home if home_side else ml_away),
            "prob": round(p_home if home_side else 1 - p_home, 3),
            "ev": round(max(ev_h, ev_a), 3),
        }
    for k, v in out.items():
        v["bet"] = v["ev"] >= config.MIN_EV_GAME
        v["stars"] = _stars(v["ev"])
    return out


def _stars(e):
    return 3 if e >= 0.08 else 2 if e >= 0.05 else 1 if e >= config.MIN_EV_GAME else 0


def _signed(x):
    x = round(x * 2) / 2
    if x == 0:
        return "PK"
    return f"+{x:g}" if x > 0 else f"{x:g}"
