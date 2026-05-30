"""Interactive World Cup 2026 predictor — simple front page.

Shows the bracket first, then the title race. All the technical breakdowns
live on the separate "Details" page (see the sidebar nav). Heavy model
training is precomputed into artifacts.json, so this only needs
streamlit / pandas / numpy at runtime.
"""
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import bracket
import engine

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")


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
    champ, _, _ = engine.run(n_pass1, gf, ks, pair, seed=0)
    tot = sum(champ.values()) or 1
    return {t: champ.get(t, 0) / tot for t in teams}


def simulate(gamma, n_sims, avail, nonce):
    """Run the two-pass model and stash everything for both pages."""
    teams, base, meta, gf, ks = get_artifacts()
    market, market_src = get_market(teams)
    mp = model_probs(min(n_sims, 2000))
    mult = engine.calibrate(market, mp, gamma=gamma)
    pair = engine.make_pair(base, mult, avail)
    champ, gs, kk = engine.run(n_sims, gf, ks, pair, seed=1 + nonce, collect=True)
    tot = sum(champ.values()) or 1
    title = {t: champ.get(t, 0) / tot for t in teams}
    ranked = sorted(teams, key=lambda t: title[t], reverse=True)
    samp = engine.representative_bracket(gf, ks, pair, want=ranked[0], seed=99 + nonce)
    hda = {"home": 0, "away": 0, "draw": 0}
    for recs in gs.values():
        for _, _, res in recs:
            hda[res] += 1
    st.session_state["results"] = {
        "mp": mp, "market": market, "market_src": market_src, "title": title,
        "ranked": ranked, "winner": ranked[0], "runner": ranked[1],
        "ko": samp["ko_matches"], "hda": hda, "n_sims": n_sims, "gamma": gamma,
        "avail": dict(avail), "meta": meta,
    }


teams, base, meta, gf, ks = get_artifacts()
market, market_src = get_market(teams)


# ---- sidebar controls --------------------------------------------------------
st.sidebar.header("⚙️ Play with it")
gamma = st.sidebar.slider(
    "Trust the betting market", 0.0, 1.0, 0.5, 0.05,
    help="0 = pure football model. 1 = lean hard on live odds. 0.5 is balanced.",
)
n_sims = st.sidebar.select_slider(
    "Tournaments to simulate", options=[500, 1000, 2000, 3000, 5000], value=2000,
    help="More = smoother odds but slower. 2000 ≈ 10 seconds.",
)
st.sidebar.subheader("🤕 Injuries")
st.sidebar.caption("Knock a team down a notch if a star is out. 1.00 = full strength.")
ranked_by_market = sorted(teams, key=lambda t: market.get(t, 0), reverse=True)
picks = st.sidebar.multiselect("Teams to weaken", ranked_by_market, default=[])
avail = {t: st.sidebar.slider(t, 0.80, 1.05, 1.00, 0.01, key=f"av_{t}") for t in picks}
run = st.sidebar.button("▶️ Run again", type="primary", use_container_width=True)


# ---- run on first load, or when settings change / button pressed -------------
sig = (gamma, n_sims, tuple(sorted(avail.items())))
if run or st.session_state.get("results") is None or st.session_state.get("sig") != sig:
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
    with st.spinner(f"Simulating {n_sims:,} tournaments…"):
        simulate(gamma, n_sims, avail, st.session_state["nonce"])
    st.session_state["sig"] = sig

R = st.session_state["results"]


# ---- main page: bracket first, then title race -------------------------------
st.title("🏆 World Cup 2026 — who wins?")
st.markdown(
    f"#### Most likely champion: **{R['winner']}** "
    f"({R['title'][R['winner']]*100:.1f}% of simulations)"
)
st.caption(
    f"One plausible run all the way to the final, where {R['winner']} lift the "
    "trophy. Tweak the sidebar and run it again — the path changes every time."
)
components.html(bracket.render(R["ko"], height=640), height=690, scrolling=True)

st.subheader("Title race — chance of winning the World Cup")
top = R["ranked"][:12]
chart_df = pd.DataFrame(
    {"team": top, "win %": [R["title"][t] * 100 for t in top]}
).set_index("team")
st.bar_chart(chart_df, horizontal=True)

st.caption("👈 Want the details? Open **Details** in the sidebar for the model "
           "vs. market breakdown, group-stage stats, and full results.")
