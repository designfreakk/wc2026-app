"""Fast runtime engine for the World Cup app.

Loads precomputed base expected-goals (from build_artifacts.py), fetches live
market odds, applies the user's market-trust and injury tweaks, and runs the
Monte Carlo tournament. No model training, no sklearn/statsmodels needed here.
"""
import io, json, re, urllib.request
from collections import Counter, defaultdict
import numpy as np, pandas as pd

HERE = __file__.rsplit("/", 1)[0]
UA = {"User-Agent": "Mozilla/5.0"}
def _get(u, t=15): return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t)

POLY_URL = "https://gamma-api.polymarket.com/events?slug=world-cup-winner"
POLY_RENAME = {"Ivory Coast": "Côte d'Ivoire", "Cape Verde": "Cabo Verde"}
# offline fallback (decimal odds, late-May 2026 snapshot)
MARKET_ODDS = {"Spain":5.5,"France":6.0,"England":7.5,"Brazil":9.0,"Argentina":9.5,"Portugal":12.0,
    "Germany":15.0,"Netherlands":21.0,"Norway":29.0,"Belgium":34.0,"Uruguay":34.0,"Croatia":41.0,
    "Colombia":41.0,"Morocco":51.0,"USA":61.0,"Japan":67.0,"Senegal":67.0,"Mexico":81.0,
    "Switzerland":81.0,"Ecuador":101.0,"Canada":101.0,"Austria":151.0,"South Korea":151.0,"Australia":251.0}
DEFAULT_ODDS = 301.0

_US=["Los Angeles","Santa Clara","Seattle","Atlanta","Miami","Boston","East Rutherford",
     "Philadelphia","Dallas","Houston","Kansas City"]
_MX=["Mexico City","Guadalajara","Monterrey"]; _CA=["Toronto","Vancouver"]
def venue_country(v):
    if any(c in v for c in _US): return "USA"
    if any(c in v for c in _MX): return "Mexico"
    if any(c in v for c in _CA): return "Canada"
    return None


def load_artifacts():
    with open(HERE + "/artifacts.json") as f:
        a = json.load(f)
    gf = pd.read_csv(HERE + "/data/group_fixtures.csv")
    ks = pd.read_csv(HERE + "/data/knockout_slots.csv")
    return a["teams"], a["base"], a["meta"], gf, ks


def fetch_market(teams):
    """Live Polymarket title probabilities; fall back to bookmaker snapshot."""
    def _odds():
        impl = {t: 1.0 / MARKET_ODDS.get(t, DEFAULT_ODDS) for t in teams}
        s = sum(impl.values()); return {t: v / s for t, v in impl.items()}
    try:
        with _get(POLY_URL) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        quoted = {}
        for m in data[0].get("markets", []):
            name = POLY_RENAME.get(m.get("groupItemTitle") or "", m.get("groupItemTitle") or "")
            try: p = float(json.loads(m.get("outcomePrices", "[]"))[0])
            except Exception: continue
            if name in teams and p > 0: quoted[name] = p
        if len(quoted) >= 20:
            floor = min(quoted.values()) / 3
            impl = {t: quoted.get(t, floor) for t in teams}; s = sum(impl.values())
            return {t: v / s for t, v in impl.items()}, f"live (Polymarket, {len(quoted)} teams)"
    except Exception as e:
        return _odds(), f"offline odds ({type(e).__name__})"
    return _odds(), "offline odds"


def calibrate(market, model_prob, gamma=0.5, eps=0.012, lo=0.85, hi=1.18):
    """Per-team goal multiplier, blending toward market where the model disagrees."""
    teams = list(market)
    raw = {t: ((market[t] + eps) / (model_prob.get(t, 0) + eps)) ** gamma for t in teams}
    g = np.exp(np.mean([np.log(v) for v in raw.values()]))
    return {t: float(np.clip(raw[t] / g, lo, hi)) for t, v in raw.items()}


def make_pair(base, mult, avail):
    """Return pair_lambdas(home, away, venue) applying market + injury tilts on the fly."""
    def pair(h, a, venue=""):
        key = f"{h}|{a}|{venue}"
        lh, la = base.get(key) or base[f"{h}|{a}|"]      # group key, else neutral
        r = (mult.get(h, 1.0) / mult.get(a, 1.0)) ** 0.5
        lh *= r; la /= r
        lh *= avail.get(h, 1.0); la *= avail.get(a, 1.0)
        return lh, la
    return pair


def scoreline_grid(pair, h, a, venue="", kmax=6):
    """P(home scores i, away scores j) for i,j in 0..kmax, using the model's
    expected goals (lh, la) for this matchup as independent Poisson rates.

    Returns (grid, lh, la) where grid[i, j] is the probability of that exact
    scoreline. Mirrors exactly what the simulator rolls each game, so the
    heatmap on the front page is a faithful picture of the model's beliefs.
    """
    lh, la = pair(h, a, venue)
    ks = np.arange(kmax + 1)
    # Poisson pmf without scipy: exp(-l) * l^k / k!
    logf = np.cumsum([0.0] + [np.log(i) for i in range(1, kmax + 1)])  # log k!
    ph = np.exp(-lh + ks * np.log(lh) - logf)
    pa = np.exp(-la + ks * np.log(la) - logf)
    return np.outer(ph, pa), float(lh), float(la)


def simulate_once(rng, gf, ks, pair):
    stats = defaultdict(lambda: {"GF": 0, "GA": 0, "Pts": 0, "group": None}); gm = {}
    for _, r in gf.iterrows():
        lh, la = pair(r.home_team, r.away_team, r.venue)
        hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))
        gm[r.match_id] = (hg, ag, "home" if hg > ag else ("away" if ag > hg else "draw"))
        for t in (r.home_team, r.away_team): stats[t]["group"] = r.group
        sh, sa = stats[r.home_team], stats[r.away_team]
        sh["GF"] += hg; sh["GA"] += ag; sa["GF"] += ag; sa["GA"] += hg
        if hg > ag: sh["Pts"] += 3
        elif ag > hg: sa["Pts"] += 3
        else: sh["Pts"] += 1; sa["Pts"] += 1
    st = pd.DataFrame([{"team": t, **s} for t, s in stats.items()])
    st["GD"] = st["GF"] - st["GA"]; st["rand"] = rng.random(len(st))
    st = st.sort_values(["group", "Pts", "GD", "GF", "rand"], ascending=[True, False, False, False, False])
    st["pos"] = st.groupby("group").cumcount() + 1
    winner_of = {r.group: r.team for _, r in st[st.pos == 1].iterrows()}
    runner_of = {r.group: r.team for _, r in st[st.pos == 2].iterrows()}
    thirds = st[st.pos == 3].sort_values(["Pts", "GD", "GF", "rand"], ascending=False).head(8)
    third_of = {r.group: r.team for _, r in thirds.iterrows()}; qual = set(third_of)
    best3 = []
    for _, r in ks.iterrows():
        for side in ("home", "away"):
            slot = r["slot_" + side]
            if str(slot).startswith("Best 3rd"):
                allowed = set(re.match(r"Best 3rd \(Groups (.+)\)", slot).group(1).split("/")) & qual
                best3.append((r.match_id, side, allowed))
    best3.sort(key=lambda s: len(s[2])); assign = {}
    def _solve(i, used):
        if i == len(best3): return True
        mid, side, allowed = best3[i]
        for g in sorted(allowed):
            if g not in used:
                assign[(mid, side)] = g; used.add(g)
                if _solve(i + 1, used): return True
                used.remove(g); del assign[(mid, side)]
        return False
    _solve(0, set())
    def resolve(slot, res):
        s = str(slot)
        if s.startswith("Winner Group "): return winner_of[s.split()[-1]]
        if s.startswith("Runner-up Group "): return runner_of[s.split()[-1]]
        m = re.match(r"Winner Match (\d+)", s)
        if m: return res[int(m.group(1))]["winner"]
        m = re.match(r"Loser Match (\d+)", s)
        if m: return res[int(m.group(1))]["loser"]
    results = {}; km = {}; reached = defaultdict(set); champ = runner = None
    for _, r in ks.iterrows():
        h = third_of[assign[(r.match_id, "home")]] if (r.match_id, "home") in assign else resolve(r.slot_home, results)
        a = third_of[assign[(r.match_id, "away")]] if (r.match_id, "away") in assign else resolve(r.slot_away, results)
        reached[h].add(r["round"]); reached[a].add(r["round"])
        lh, la = pair(h, a, r.venue); hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))
        if hg == ag:
            ph = lh / (lh + la) if (lh + la) > 0 else 0.5
            ws = "home" if rng.random() < ph else "away"; pens = True
        else:
            ws = "home" if hg > ag else "away"; pens = False
        winner = h if ws == "home" else a; loser = a if ws == "home" else h
        results[r.match_id] = {"winner": winner, "loser": loser}
        km[r.match_id] = (h, a, hg, ag, ws, pens)
        if r["round"] == "Final": champ, runner = winner, loser
    # group-stage finishing position (group, pos 1..4) for every team
    finish = {row.team: (row.group, int(row.pos)) for _, row in st.iterrows()}
    return {"champion": champ, "runner_up": runner, "reached": reached,
            "group_matches": gm, "ko_matches": km, "finish": finish}


def representative_bracket(gf, ks, pair, want=None, seed=99, max_tries=600):
    """Return one coherent tournament (ideally ending with champion `want`).

    Each simulation is random, so to show 'a path to the final' we roll until we
    find one whose champion matches the overall favourite, then return that single
    consistent bracket. Falls back to the last roll if `want` never wins in time.
    """
    rng = np.random.default_rng(seed)
    last = None
    for _ in range(max_tries):
        r = simulate_once(rng, gf, ks, pair)
        last = r
        if want is None or r["champion"] == want:
            break
    return last


# knockout rounds, in order, that count toward the "how far do they go" ladder
# (the third-place playoff is a consolation match and is deliberately excluded)
REACH_LADDER = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]


def run(n, gf, ks, pair, seed=0, collect=False, progress=None):
    """Simulate n tournaments. Returns (champ_counter, group_recs, ko_recs, samples, agg).

    `samples` maps each champion seen -> one full ko_matches bracket where they won,
    so the app can show a representative bracket for the favourite without a second
    search loop. `agg` holds tournament-wide tallies aggregated over every run:
        agg["reach"][team][round]  -> times that team reached that knockout round
        agg["reach"][team]["Champion"] -> times that team won it all
        agg["finish"][team][pos]   -> times that team finished 1st/2nd/3rd/4th in its group
        agg["group"][team]         -> the team's group letter
    `progress(frac)` (optional) is called ~20x for a UI progress bar.
    """
    rng = np.random.default_rng(seed); champ = Counter(); gs = defaultdict(list); kk = defaultdict(list)
    samples = {}
    reach = defaultdict(Counter); finish = defaultdict(Counter); team_group = {}
    step = max(1, n // 20)
    for i in range(n):
        r = simulate_once(rng, gf, ks, pair); champ[r["champion"]] += 1
        if r["champion"] not in samples: samples[r["champion"]] = r["ko_matches"]
        for team, rounds in r["reached"].items():
            for rd in rounds: reach[team][rd] += 1
        reach[r["champion"]]["Champion"] += 1
        for team, (grp, pos) in r["finish"].items():
            team_group[team] = grp; finish[team][pos] += 1
        if collect:
            for mid, rec in r["group_matches"].items(): gs[mid].append(rec)
            for mid, rec in r["ko_matches"].items(): kk[mid].append(rec)
        if progress is not None and (i + 1) % step == 0: progress((i + 1) / n)
    agg = {"reach": reach, "finish": finish, "group": team_group, "n": n}
    return champ, gs, kk, samples, agg
