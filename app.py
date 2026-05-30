"""Interactive World Cup 2026 predictor — getfast.ai-style scrolling landing.

Layout: a white top bar, a dark full-bleed hero (red eyebrow + headline + a red
Simulate button), then the framed bracket card, followed by light content
sections — Used data sources, How it works, Modifications you can make, and a
closing call-to-action. Heavy model training is precomputed into artifacts.json;
only streamlit / pandas / numpy / altair are needed at runtime.
"""
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import bracket
import charts
import engine
import views

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")

# ---- styling -----------------------------------------------------------------
# getfast.ai look: clean white product canvas + Inter throughout, with a single
# dark full-bleed hero band at the top. One red accent (#fc0017) everywhere.
st.markdown(
    """
    <style>
      /* strip Streamlit chrome for a landing-page feel */
      section[data-testid="stSidebar"], button[data-testid="collapsedControl"]{display:none!important;}
      [data-testid="stToolbar"], #MainMenu, footer{display:none!important;}
      [data-testid="stHeader"]{background:transparent;height:0;}
      .stApp{overflow-x:hidden;}
      .block-container, [data-testid="stMainBlockContainer"]{padding-top:0!important;}

      /* typography — Inter for body + headings, getfast tracking */
      html, body, [class*="st-"]{ letter-spacing:-0.2px; }
      h1,h2,h3,h4{ font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif!important;
                   font-weight:600!important; letter-spacing:-0.01em; line-height:1.25!important;
                   overflow-wrap:anywhere; color:#212529; }
      h2{ font-size:1.875rem!important; } h3{ font-size:1.5rem!important; } h4{ font-size:1.25rem!important; }

      /* full-bleed white top bar */
      .topbar{ position:sticky; top:0; z-index:1000; width:100vw; margin-left:calc(50% - 50vw);
               background:#ffffff; box-shadow:0 1px 0 rgba(0,0,0,.07);
               display:flex; align-items:center; justify-content:space-between;
               padding:12px 30px; box-sizing:border-box; }
      .topbar .brand{ display:flex; align-items:center; gap:9px; font-weight:700; font-size:1.15rem; color:#111; }
      .topbar .brand .mark{ font-size:1.3rem; }
      .topbar .brand b{ color:#fc0017; font-weight:700; }
      .topbar .nav{ color:#6c757d; font-weight:500; font-size:.95rem; }

      /* full-bleed dark hero band; following content overlaps up into it */
      .hero-band{ width:100vw; margin-left:calc(50% - 50vw); background:#0e0e10;
                  text-align:center; padding:74px 16px 190px; margin-bottom:-120px; box-sizing:border-box; }
      .hero-band .eyebrow{ color:#fc0017; font-weight:700; text-transform:uppercase;
                  letter-spacing:.26em; font-size:.85rem; margin-bottom:20px; }
      .hero-band h1{ color:#ffffff!important; font-size:4.2rem; line-height:1.04; letter-spacing:-0.035em;
                  margin:0 auto; max-width:16ch; font-weight:600; }
      .hero-band .sub{ color:#aeb2ba; font-size:1.2rem; line-height:1.5; max-width:48ch; margin:22px auto 0; }
      @media (max-width:760px){ .hero-band h1{ font-size:2.6rem; } .hero-band{ padding-bottom:150px; margin-bottom:-100px; } }

      /* red pill buttons (primary), centered */
      .stButton{ text-align:center; }
      .stButton>button[kind="primary"]{ background:#fc0017; border:none; color:#fff;
                  border-radius:3rem; font-weight:700; font-size:1.1rem; padding:.8rem 2.6rem; }
      .stButton>button[kind="primary"]:hover{ background:#d80014; color:#fff; }

      /* the bracket renders in an iframe — make it a glowing white card on the seam */
      [data-testid="stIFrame"] iframe, iframe[title="st.iframe"]{
                  background:#fff; border-radius:18px; box-shadow:0 30px 80px rgba(0,0,0,.45); }

      /* section header block */
      .sec{ text-align:center; max-width:60ch; margin:0 auto 6px; }
      .sec .eyebrow{ color:#fc0017; font-weight:700; text-transform:uppercase; letter-spacing:.2em;
                  font-size:.8rem; margin-bottom:10px; }
      .sec h2{ margin:0; } .sec p{ color:#6c757d; font-size:1.05rem; margin:.5rem auto 0; }
      hr{ border:none; border-top:1px solid #e6e8eb; margin:3.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- cached heavy bits -------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return engine.load_artifacts()


@st.cache_data(ttl=900)
def get_market(teams):
    return engine.fetch_market(tuple(teams))


@st.cache_data(show_spinner=False)
def model_probs(n_pass1):
    """Pass 1: pure-model title probabilities (no market, no injuries)."""
    teams, base, meta, gf, ks = get_artifacts()
    pair = engine.make_pair(base, {t: 1.0 for t in teams}, {})
    champ, _, _, _, _ = engine.run(n_pass1, gf, ks, pair, seed=0)
    tot = sum(champ.values()) or 1
    return {t: champ.get(t, 0) / tot for t in teams}


def simulate(gamma, n_sims, avail, nonce, progress=None):
    """Run the two-pass model and stash everything for the whole page."""
    teams, base, meta, gf, ks = get_artifacts()
    market, market_src = get_market(teams)
    mp = model_probs(600)  # pass 1: cheap, cached; enough to set market multipliers
    mult = engine.calibrate(market, mp, gamma=gamma)
    pair = engine.make_pair(base, mult, avail)
    champ, gs, kk, samples, agg = engine.run(
        n_sims, gf, ks, pair, seed=1 + nonce, collect=True, progress=progress)
    tot = sum(champ.values()) or 1
    title = {t: champ.get(t, 0) / tot for t in teams}
    ranked = sorted(teams, key=lambda t: title[t], reverse=True)
    ko = samples.get(ranked[0]) or next(iter(samples.values()))
    hda = {"home": 0, "away": 0, "draw": 0}
    for recs in gs.values():
        for _, _, res in recs:
            hda[res] += 1
    grid, flh, fla = engine.scoreline_grid(pair, ranked[0], ranked[1], "")
    st.session_state["results"] = {
        "mp": mp, "market": market, "market_src": market_src, "title": title,
        "ranked": ranked, "winner": ranked[0], "runner": ranked[1],
        "ko": ko, "hda": hda, "n_sims": n_sims, "gamma": gamma,
        "avail": dict(avail), "meta": meta,
        "score_grid": grid, "final_lh": flh, "final_la": fla,
        "agg": agg,
    }


teams, base, meta, gf, ks = get_artifacts()
market, market_src = get_market(teams)
ranked_by_market = sorted(teams, key=lambda t: market.get(t, 0), reverse=True)

# Control values live in the "Modifications" section near the bottom, but the
# bracket up top needs them now. Read them from session_state (defaults on first
# load); the widgets below write the same keys, so any change reruns and the top
# picks up the new value with no lag.
gamma = st.session_state.get("ctl_gamma", 0.5)
n_sims = st.session_state.get("ctl_sims", 1000)
picks = st.session_state.get("ctl_picks", [])
avail = {t: st.session_state.get(f"av_{t}", 1.0) for t in picks}

# A Simulate button anywhere sets this flag and reruns; we consume it up here.
force = st.session_state.pop("do_run", False)
sig = (gamma, n_sims, tuple(sorted(avail.items())))
if force or st.session_state.get("results") is None or st.session_state.get("sig") != sig:
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
    bar = st.progress(0.0, text=f"Simulating {n_sims:,} tournaments…")
    simulate(gamma, n_sims, avail, st.session_state["nonce"],
             progress=lambda f: bar.progress(f, text=f"Simulating {n_sims:,} tournaments… {int(f*100)}%"))
    bar.empty()
    st.session_state["sig"] = sig

R = st.session_state["results"]
M = R["meta"]


def trigger_simulation():
    st.session_state["do_run"] = True


# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR + HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="topbar">'
    '<div class="brand"><span class="mark">🏆</span>World&nbsp;<b>Cup&nbsp;2026</b></div>'
    '<div class="nav">Predictor</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-band">'
    '<div class="eyebrow">World Cup 2026</div>'
    '<h1>Simulate the World&nbsp;Cup.</h1>'
    '<div class="sub">A football model blended with the live betting market plays '
    'all 104 matches — groups to final — thousands of times, so you can see who '
    'really lifts the trophy.</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.button("Simulate", type="primary", key="btn_hero", on_click=trigger_simulation)

# ---- the framed table: a representative bracket ------------------------------
components.html(bracket.render(R["ko"], height=640), height=700, scrolling=False)
st.caption(
    f"One plausible run all the way to the final, where **{R['winner']}** lift the "
    f"trophy ({R['title'][R['winner']]*100:.1f}% of {R['n_sims']:,} simulations). "
    "Hit Simulate again — the path changes every time."
)

# ---- explore the rest of the results (kept as tabs for now) ------------------
with st.expander("Explore more results — title race, group stage, full details"):
    t_race, t_group, t_details = st.tabs(["🥇 Title Race", "📊 Group Stage", "📋 Details"])
    with t_race:
        top = R["ranked"][:12]
        chart_df = pd.DataFrame(
            {"team": top, "win %": [R["title"][t] * 100 for t in top]}
        ).set_index("team")
        st.bar_chart(chart_df, horizontal=True)
        st.caption(
            f"Chance of winning the World Cup · {R['market_src']}. Percentages wiggle "
            "a little each run — that's the uncertainty, not a bug."
        )
        st.dataframe(charts.advancement_table(R["agg"], R["ranked"], top=24),
                     use_container_width=True)
    with t_group:
        st.dataframe(charts.group_finish_table(R["agg"]),
                     use_container_width=True, height=520)
    with t_details:
        views.render_details(R, ks)


def section(eyebrow, title, sub=""):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec"><div class="eyebrow">{eyebrow}</div>'
        f'<h2>{title}</h2>{f"<p>{sub}</p>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# USED DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════
section("The inputs", "Used data sources",
        "Everything the simulator stands on — what it is, and how much of it.")
ds = pd.DataFrame([
    {"Source": "International match results",
     "What it provides": "Goals & results used to train the goals model",
     "Coverage": f"{M['train_matches']:,} matches · {M['train_from'][:4]}–{M['train_to'][:4]}"},
    {"Source": "Live betting market",
     "What it provides": "Implied title odds, blended in via the market-trust dial",
     "Coverage": R["market_src"]},
    {"Source": "2026 tournament structure",
     "What it provides": "Group draw, fixtures & the full knockout path",
     "Coverage": "48 teams · 12 groups · 104 matches"},
    {"Source": "Team strength priors",
     "What it provides": "Per-team attack & defence ratings + home advantage",
     "Coverage": "Derived from the match results above"},
])
st.dataframe(ds, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
section("Under the hood", "How it works",
        "An ensemble football model, tilted toward the market, simulated end to end.")
st.markdown(
    "We start with a football model trained on every international match since "
    f"{M['train_from'][:4]} — about {M['train_matches']:,} games. It's really *two* "
    "models averaged: a classic **Poisson** goals model and a **machine-learning** "
    "one, each predicting how many goals a team should score given strength, recent "
    "form and home advantage.\n\n"
    "We then nudge each team toward the **live betting market** (you control how hard "
    "in *Modifications* below), because bookmakers price in news the raw stats miss. "
    "Finally we **play the whole tournament** — all 104 matches, groups to final — "
    "thousands of times, rolling realistic scorelines each game (penalty shootouts "
    "included). How often each team wins is the percentage you see."
)
c1, c2, c3, c4 = st.columns(4)
bt_c = M["backtest"]["competitive"]
c1.metric("Tournaments simulated", f"{R['n_sims']:,}", help="Per run, using the current controls.")
c2.metric("Matches trained on", f"{M['train_matches']:,}",
          help=f"Internationals from {M['train_from']} to {M['train_to']}.")
c3.metric("Winner accuracy", f"{bt_c['winner']:.0f}%",
          help=f"Out-of-sample on {bt_c['n']} competitive matches the model never saw.")
c4.metric("Goal error (MAE)", f"{bt_c['mae']:.2f}",
          help="Average miss in goals-per-team on unseen matches. Lower is better.")

st.markdown("**The model's-eye view of the final**")
st.caption(
    f"If **{R['winner']}** met **{R['runner']}** on neutral ground, here's the chance "
    "of every scoreline. The outlined square is the single most likely one — exactly "
    "what the simulator rolls each game."
)
st.altair_chart(charts.poisson_heatmap(R["score_grid"], R["winner"], R["runner"]),
                use_container_width=True)

t1, t2, t3 = st.tabs(["📈 Track record", "⚖️ Model vs market", "🎛️ What drives it"])
with t1:
    st.markdown(
        f"We hid every match after **{M['backtest']['cutoff']}**, retrained on the "
        f"rest, and predicted those **{M['backtest']['test_n']:,}** unseen games."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Picked the winner", f"{bt_c['winner']:.0f}%", help="3-way (win/draw/loss) on competitive matches.")
    m2.metric("Exact scoreline", f"{bt_c['exact']:.0f}%", help="~10% is normal for football.")
    m3.metric("Goal error (MAE)", f"{bt_c['mae']:.2f}", help="Average goals-per-team miss.")
    m4.metric("vs. Elo baseline", f"+{bt_c['model2'] - bt_c['elo2']:.1f} pts",
              help=f"Model {bt_c['model2']:.0f}% vs Elo {bt_c['elo2']:.0f}% head-to-head.")
    st.caption(
        "Honest read: on head-to-head calls we're about level with a simple Elo "
        "baseline — football is genuinely hard to predict. The model earns its keep "
        "in the *scorelines* it rolls, which is what makes the simulation realistic."
    )
with t2:
    st.altair_chart(charts.model_vs_market_scatter(R["title"], R["market"], R["ranked"]),
                    use_container_width=True)
    st.caption(
        "Each dot is a team: our simulated title chance vs the betting market's. On "
        "the dashed line we agree; above it we're more bullish, below more bearish."
    )
with t3:
    st.altair_chart(charts.feature_importance_bar(M["feature_importance"]),
                    use_container_width=True)
    st.caption(
        "What moves the goals model most, by shuffling each input and watching "
        "accuracy drop. Team strength dominates, with form and home advantage filling in."
    )

# ══════════════════════════════════════════════════════════════════════════════
# MODIFICATIONS YOU CAN MAKE
# ══════════════════════════════════════════════════════════════════════════════
section("Make it yours", "Modifications you can make",
        "Tune the dials, then re-simulate — every result above updates to match.")
mc1, mc2 = st.columns(2)
with mc1:
    st.slider("📊 Market trust", 0.0, 1.0, 0.5, 0.05, key="ctl_gamma",
              help="0 = pure football model · 1 = lean hard on live odds · 0.5 is balanced.")
with mc2:
    st.select_slider("🎲 Simulations", options=[300, 500, 1000, 2000, 3000], value=1000,
                     key="ctl_sims", help="More = smoother odds but slower. 1 000 is a good balance.")
st.multiselect("🤕 Injuries — weaken a team if a star is out", ranked_by_market,
               default=[], key="ctl_picks")
for t in st.session_state.get("ctl_picks", []):
    st.slider(f"{t} availability", 0.80, 1.05, 1.00, 0.01, key=f"av_{t}",
              help="1.00 = full strength.")

# ══════════════════════════════════════════════════════════════════════════════
# CLOSING CTA
# ══════════════════════════════════════════════════════════════════════════════
section("Your turn", "Ready to simulate and predict the World Cup?")
st.button("Simulate", type="primary", key="btn_end", on_click=trigger_simulation)
st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
