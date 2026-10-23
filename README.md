# Edge Pickz

Pulls NFL stats, schedule, betting lines, injury reports, news and weather every few minutes,
re-runs a pricing model, and publishes a dashboard with spread, total, moneyline and
player prop picks. Each game's picks **lock 30 minutes before kickoff** and are graded
after the game, so the record can't be cherry-picked.

Picks are model estimates for information only, not guarantees.

## Data sources

| Source | What | Cost |
|---|---|---|
| [nflverse](https://github.com/nflverse/nflverse-data) | Schedule, consensus lines, player and team stats, official injury reports | Free |
| ESPN public API | Latest injury designations, team news headlines | Free |
| [Open-Meteo](https://open-meteo.com) | Kickoff-hour wind, temp, precipitation for outdoor stadiums | Free |
| [The Odds API](https://the-odds-api.com) (optional) | Best live line across US books, player prop lines | Free tier 500 credits/mo; set `ODDS_API_KEY` |

Without an odds key the board uses nflverse consensus lines for games and shows prop
*projections* with the line you'd need to bet (no prop EV). With a key, live lines and
props are pulled once per game at its final lock (30 minutes before kickoff) (about 9 credits per game, ~400/month).

## College football

A **College** tab covers every game with an AP Top 25 team: spread, total and moneyline
picks from team ratings fit to this season's FBS scores, compared with DraftKings lines
(via ESPN). Same 24-hour lock, grading and record. Trends use ESPN's archive of past lines.
No player props and no injury reports for college (teams don't publish them).

## How the model works

- **Team strength**: a recency-weighted rating fit to points scored/allowed (this season
  and last), blended 50/50 with offensive and defensive EPA per play.
- **Injuries**: starting QB out costs 1.5 to 6.5 points depending on his EPA; skill players
  up to 1.5 each; questionable players count 30%.
- **Weather**: wind over 12 mph, heavy precipitation and sub-25°F temps lower the total.
- **Edge**: the model only moves 35% of the way from the market toward its own number
  (markets are sharp), then converts to win probability and expected value at the price.
  A pick is flagged as a bet at +2% EV for games and +4% for props.
- **Props**: recency-weighted player averages, adjusted for the opponent's defense vs. that
  position, the implied team total, and wind; simulated 20,000 times.

Tune everything in `nflpicks/config.py`.

## Run it

```bash
pip install -r requirements.txt
python -m nflpicks.build          # writes site/index.html and site/picks.json
ODDS_API_KEY=... python -m nflpicks.build
```

`history/locked_picks.json` is the permanent record of locked picks. Keep it committed.

## Host it (free, auto-updating)

1. Push this folder to a GitHub repo.
2. Settings → Pages → Source: **GitHub Actions**.
3. (Optional) Settings → Secrets → Actions → `ODDS_API_KEY`.
4. The workflow in `.github/workflows/update.yml` runs every hour and redeploys the page.
