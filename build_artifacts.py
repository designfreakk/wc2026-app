"""Precompute the Rung 5 v2 model into small artifacts the web app loads instantly.

Trains the Poisson + ML goal models on (live, with fallback) results, blends them,
and warms the expensive per-matchup expected-goals ("base lambdas") for every
group fixture (with host advantage) and every neutral team pair. Saves the result
as a small JSON so the Streamlit app needs only numpy/pandas at runtime — no model
retraining, no sklearn/statsmodels on the server.

Run from the rung5_ensemble folder:  python app/build_artifacts.py
Re-run any time to refresh the base model with newer results.
"""
import io, json, urllib.request, datetime as dt
from collections import defaultdict, deque
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import statsmodels.api as sm, statsmodels.formula.api as smf

# Held-out backtest results (from backtest.py, cutoff 2025-06-01). These describe
# how this exact methodology performs on matches it never saw during training, so
# we surface them on the app's front page as honest accuracy metrics.
BACKTEST = {
    "cutoff": "2025-06-01", "test_n": 953, "test_competitive": 697,
    "all":         {"n": 953, "winner": 57.5, "exact": 9.7, "mae": 1.89, "model2": 79.6, "elo2": 79.0, "draws": 22.9},
    "competitive": {"n": 697, "winner": 59.7, "exact": 8.9, "mae": 1.92, "model2": 79.6, "elo2": 79.6, "draws": 21.2},
    "friendlies":  {"n": 256, "winner": 51.6, "exact": 11.7, "mae": 1.80, "model2": 79.6, "elo2": 77.4, "draws": 27.3},
}
# human-friendly labels for the model's input features
FEATURE_LABELS = {
    "is_home": "Home advantage", "competitive": "Competitive match",
    "elo": "Team strength (Elo)", "opp_elo": "Opponent strength (Elo)",
    "elo_diff": "Strength gap", "form_gf": "Recent goals scored",
    "form_ga": "Recent goals conceded", "opp_form_gf": "Opp. recent goals scored",
    "opp_form_ga": "Opp. recent goals conceded",
}

HERE = __file__.rsplit("/", 1)[0]
UA = {"User-Agent": "Mozilla/5.0"}
def _get(u, t=20): return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t)
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def load_results():
    bundled_path = HERE + "/../data/results.csv"
    bundled = pd.read_csv(bundled_path, parse_dates=["date"])
    try:
        with _get(RESULTS_URL) as r:
            live = pd.read_csv(io.StringIO(r.read().decode("utf-8")), parse_dates=["date"])
        if {"date","home_team","away_team","home_score","away_score","tournament","neutral"} <= set(live.columns) \
           and len(live) >= len(bundled) and live.date.max() >= pd.Timestamp("2024-01-01"):
            print(f"✅ live results: {len(live):,} rows through {live.date.max().date()}")
            return live, "live"
    except Exception as e:
        print(f"⚠️ live fetch failed ({type(e).__name__}) — bundled")
    return bundled, "bundled"

CUTOFF_ML="2010-01-01"; CUTOFF_POIS="2015-01-01"; FORM_N=5; K=30; BASE_ELO=1500.0
PROXY = {"UEFA Playoff A":"Scotland","UEFA Playoff B":"Scotland","UEFA Playoff C":"Scotland",
         "UEFA Playoff D":"Scotland","FIFA Playoff 1":"New Zealand","FIFA Playoff 2":"New Zealand"}
_RENAME = {"United States":"USA","Ivory Coast":"Côte d'Ivoire","Cape Verde":"Cabo Verde"}
_US=["Los Angeles","Santa Clara","Seattle","Atlanta","Miami","Boston","East Rutherford",
     "Philadelphia","Dallas","Houston","Kansas City"]
_MX=["Mexico City","Guadalajara","Monterrey"]; _CA=["Toronto","Vancouver"]
FEATURES=["is_home","competitive","elo","opp_elo","elo_diff","form_gf","form_ga","opp_form_gf","opp_form_ga"]
BLEND_W=0.5

def venue_country(v):
    if any(c in v for c in _US): return "USA"
    if any(c in v for c in _MX): return "Mexico"
    if any(c in v for c in _CA): return "Canada"
    return None
def _exp(ra,rb): return 1.0/(1.0+10**((rb-ra)/400.0))

def build_features(df):
    df=df.dropna(subset=["home_score","away_score"]); df=df[df.date>=CUTOFF_ML].sort_values("date").copy()
    df["neutral"]=df["neutral"].astype(str).str.upper().eq("TRUE")
    df["home_team"]=df["home_team"].replace(_RENAME); df["away_team"]=df["away_team"].replace(_RENAME)
    df["competitive"]=(~df["tournament"].str.contains("Friendly",case=False,na=False)).astype(int)
    elo=defaultdict(lambda:BASE_ELO); form=defaultdict(lambda:deque(maxlen=FORM_N))
    def fm(t):
        d=form[t]; return (0.0,0.0) if not d else (sum(x[0] for x in d)/len(d),sum(x[1] for x in d)/len(d))
    rows=[]
    for r in df.itertuples(index=False):
        h,a=r.home_team,r.away_team; hg,ag=int(r.home_score),int(r.away_score)
        eh,ea=elo[h],elo[a]; hgf,hga=fm(h); agf,aga=fm(a); ih=0 if r.neutral else 1
        rows.append({"goals":hg,"is_home":ih,"competitive":r.competitive,"elo":eh,"opp_elo":ea,
                     "elo_diff":eh-ea,"form_gf":hgf,"form_ga":hga,"opp_form_gf":agf,"opp_form_ga":aga})
        rows.append({"goals":ag,"is_home":0,"competitive":r.competitive,"elo":ea,"opp_elo":eh,
                     "elo_diff":ea-eh,"form_gf":agf,"form_ga":aga,"opp_form_gf":hgf,"opp_form_ga":hga})
        exp_h=_exp(eh+(100 if ih else 0),ea); sh=1.0 if hg>ag else (0.5 if hg==ag else 0.0); mg=max(1,abs(hg-ag))**0.5
        elo[h]=eh+K*mg*(sh-exp_h); elo[a]=ea+K*mg*((1-sh)-(1-exp_h)); form[h].append((hg,ag)); form[a].append((ag,hg))
    return pd.DataFrame(rows), dict(elo), {t:fm(t) for t in form}

def fit_poisson(df):
    df=df.dropna(subset=["home_score","away_score"]); df=df[df.date>=CUTOFF_POIS].copy()
    df["neutral"]=df["neutral"].astype(str).str.upper().eq("TRUE")
    df["home_team"]=df["home_team"].replace(_RENAME); df["away_team"]=df["away_team"].replace(_RENAME)
    cnt=pd.concat([df.home_team,df.away_team]).value_counts(); keep=set(cnt[cnt>=30].index)
    df=df[df.home_team.isin(keep)&df.away_team.isin(keep)]
    asof=df.date.max(); df["w"]=0.5**((asof-df.date).dt.days/(3*365.25))
    home=pd.DataFrame({"goals":df.home_score.astype(int),"team":df.home_team,"opp":df.away_team,"is_home":(~df.neutral).astype(int),"w":df.w})
    away=pd.DataFrame({"goals":df.away_score.astype(int),"team":df.away_team,"opp":df.home_team,"is_home":0,"w":df.w})
    lp=pd.concat([home,away],ignore_index=True)
    m=smf.glm("goals ~ C(team) + C(opp) + is_home",data=lp,family=sm.families.Poisson(),var_weights=lp["w"]).fit()
    p=m.params; att,deff={},{}
    for name,val in p.items():
        if name.startswith("C(team)[T."): att[name[10:-1]]=val
        elif name.startswith("C(opp)[T."): deff[name[9:-1]]=val
    return p["Intercept"],p["is_home"],att,deff

# ---- train ----
RESULTS, SRC = load_results()
print("training feature table + GB …"); long,ELO,FORM = build_features(RESULTS)
gb=HistGradientBoostingRegressor(loss="poisson",max_iter=400,learning_rate=0.05,max_leaf_nodes=31,
    min_samples_leaf=50,l2_regularization=1.0,random_state=0).fit(long[FEATURES],long["goals"])
print("fitting Poisson GLM …"); P_INT,P_HOME,P_ATT,P_DEF = fit_poisson(RESULTS)

# ---- permutation feature importance (how much each input drives the GB model) ----
print("scoring feature importance …")
_imp_idx = np.random.default_rng(0).choice(len(long), size=min(8000, len(long)), replace=False)
_pi = permutation_importance(gb, long[FEATURES].iloc[_imp_idx], long["goals"].iloc[_imp_idx],
                             n_repeats=5, random_state=0, scoring="neg_mean_poisson_deviance")
_imp = sorted(zip(FEATURES, _pi.importances_mean), key=lambda kv: kv[1], reverse=True)
_imax = max(v for _, v in _imp) or 1.0
FEATURE_IMPORTANCE = [{"feature": FEATURE_LABELS.get(f, f),
                       "importance": round(float(v / _imax), 4)} for f, v in _imp]

# training-data span actually used by the model (after the ML cutoff)
_train_df = RESULTS.dropna(subset=["home_score","away_score"])
_train_df = _train_df[_train_df.date >= CUTOFF_ML]
TRAIN_META = {"train_matches": int(len(_train_df)),
              "train_from": _train_df.date.min().date().isoformat(),
              "train_to": _train_df.date.max().date().isoformat(),
              "results_rows": int(len(RESULTS))}

gf=pd.read_csv(HERE+"/data/group_fixtures.csv"); ks=pd.read_csv(HERE+"/data/knockout_slots.csv")
teams=sorted(set(gf.home_team)|set(gf.away_team))

def poisson_eg(att,dfd,is_home):
    a=PROXY.get(att,att); d=PROXY.get(dfd,dfd)
    return float(np.exp(P_INT+P_ATT.get(a,0.0)+P_DEF.get(d,0.0)+P_HOME*is_home))
def feat(team,opp,is_home):
    t=PROXY.get(team,team); o=PROXY.get(opp,opp)
    et=ELO.get(t,BASE_ELO); eo=ELO.get(o,BASE_ELO); tf=FORM.get(t,(0.,0.)); of=FORM.get(o,(0.,0.))
    return {"is_home":is_home,"competitive":1,"elo":et,"opp_elo":eo,"elo_diff":et-eo,
            "form_gf":tf[0],"form_ga":tf[1],"opp_form_gf":of[0],"opp_form_ga":of[1]}

# ---- warm base lambdas: neutral ordered pairs + group fixtures (host adv) ----
print("warming base lambdas …")
entries=[]                                   # (key, home, away, ih_home, ih_away)
for h in teams:
    for a in teams:
        if h!=a: entries.append((f"{h}|{a}|", h, a, 0, 0))
for _,r in gf.iterrows():
    vc=venue_country(r.venue)
    entries.append((f"{r.home_team}|{r.away_team}|{r.venue}", r.home_team, r.away_team,
                    1 if r.home_team==vc else 0, 1 if r.away_team==vc else 0))
home_rows=[feat(h,a,ihh) for _,h,a,ihh,_ in entries]
away_rows=[feat(a,h,iha) for _,h,a,_,iha in entries]
ml_h=gb.predict(pd.DataFrame(home_rows)[FEATURES]).clip(0.01)
ml_a=gb.predict(pd.DataFrame(away_rows)[FEATURES]).clip(0.01)
base={}
for i,(key,h,a,ihh,iha) in enumerate(entries):
    ph=poisson_eg(h,a,ihh); pa=poisson_eg(a,h,iha)
    lh=(ph**(1-BLEND_W))*(ml_h[i]**BLEND_W); la=(pa**(1-BLEND_W))*(ml_a[i]**BLEND_W)
    base[key]=[round(float(lh),4), round(float(la),4)]

out={"meta":{"results_source":SRC,"built":dt.date.today().isoformat(),"n_pairs":len(base),
             **TRAIN_META, "backtest":BACKTEST, "feature_importance":FEATURE_IMPORTANCE,
             "blend_w":BLEND_W, "form_n":FORM_N},
     "teams":teams,"base":base}
path=HERE+"/artifacts.json"
with open(path,"w") as f: json.dump(out,f)
print(f"wrote {path}  ({len(base):,} matchups, results={SRC})")
