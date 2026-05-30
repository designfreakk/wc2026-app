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
import charts
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
    champ, _, _, _ = engine.run(n_pass1, gf, ks, pair, seed=0)
    tot = sum(champ.values()) or 1
    return {t: champ.get(t, 0) / tot for t in teams}


def simulate(gamma, n_sims, avail, nonce, progress=None):
    """Run the two-pass model and stash everything for both pages."""
    teams, base, meta, gf, ks = get_artifacts()
    market, market_src = get_market(teams)
    mp = model_probs(600)  # pass 1: cheap, cached; enough to set market multipliers
    mult = engine.calibrate(market, mp, gamma=gamma)
    pair = engine.make_pair(base, mult, avail)
    champ, gs, kk, samples = engine.run(
        n_sims, gf, ks, pair, seed=1 + nonce, collect=True, progress=progress)
    tot = sum(champ.values()) or 1
    title = {t: champ.get(t, 0) / tot for t in teams}
    ranked = sorted(teams, key=lambda t: title[t], reverse=True)
    # representative bracket: a run where the favourite won (captured during the run)
    ko = samples.get(ranked[0]) or next(iter(samples.values()))
    hda = {"home": 0, "away": 0, "draw": 0}
    for recs in gs.values():
        for _, _, res in recs:
            hda[res] += 1
    # scoreline grid the model expects if the top two teams met (neutral venue)
    grid, flh, fla = engine.scoreline_grid(pair, ranked[0], ranked[1], "")
    st.session_state["results"] = {
        "mp": mp, "market": market, "market_src": market_src, "title": title,
        "ranked": ranked, "winner": ranked[0], "runner": ranked[1],
        "ko": ko, "hda": hda, "n_sims": n_sims, "gamma": gamma,
        "avail": dict(avail), "meta": meta,
        "score_grid": grid, "final_lh": flh, "final_la": fla,
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
    "Tournaments to simulate", options=[300, 500, 1000, 2000, 3000], value=1000,
    help="More = smoother odds but slower. 1000 is a good balance.",
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
    bar = st.progress(0.0, text=f"Simulating {n_sims:,} tournaments…")
    simulate(gamma, n_sims, avail, st.session_state["nonce"],
             progress=lambda f: bar.progress(f, text=f"Simulating {n_sims:,} tournaments… {int(f*100)}%"))
    bar.empty()
    st.session_state["sig"] = sig

R = st.session_state["results"]


# ---- main page: hero (the answer) → how it works → under the hood ------------
M = R["meta"]
st.title("🏆 World Cup 2026 — who wins?")
st.markdown(
    "We simulate the entire tournament thousands of times with a football model "
    "blended with the live betting market, then count how often each team lifts "
    "the trophy. Here's what comes out — play with the sidebar to change the story."
)

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
st.caption(
    f"Odds blend a football model with live market prices · {R['market_src']}. "
    "Because every simulation rolls random scorelines, the percentages wiggle a "
    "little each run — that's the uncertainty, not a bug."
)

# ---- how it works ------------------------------------------------------------
st.divider()
st.subheader("How it works")
st.markdown(
    "We start with a football model trained on every international match since "
    f"{M['train_from'][:4]} — about {M['train_matches']:,} games. It's really *two* "
    "models averaged: a classic **Poisson** goals model and a **machine-learning** "
    "one, each predicting how many goals a team should score given strength, recent "
    "form and home advantage.\n\n"
    "We then nudge each team toward the **live betting market** (you control how hard "
    "with the sidebar), because bookmakers price in news the raw stats miss — "
    "injuries, form, momentum.\n\n"
    "Finally we **play the whole tournament** — all 104 matches, groups to final — "
    "thousands of times, rolling realistic scorelines each game (penalty shootouts "
    "included). How often each team wins is the percentage you see above."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tournaments simulated", f"{R['n_sims']:,}", help="Per run, using the current sidebar settings.")
c2.metric("Matches trained on", f"{M['train_matches']:,}", help=f"Internationals from {M['train_from']} to {M['train_to']}.")
bt_c = M["backtest"]["competitive"]
c3.metric("Winner accuracy", f"{bt_c['winner']:.0f}%", help=f"Out-of-sample on {bt_c['n']} competitive matches the model never saw.")
c4.metric("Goal error (MAE)", f"{bt_c['mae']:.2f}", help="Average miss in goals-per-team on unseen matches. Lower is better.")

st.markdown("**The model's-eye view of the final**")
st.caption(
    f"If **{R['winner']}** met **{R['runner']}** on neutral ground, here's the chance "
    "of every scoreline. Each square is one exact result; the outlined square is the "
    "single most likely one. This is exactly what the simulator rolls each game."
)
st.altair_chart(
    charts.poisson_heatmap(R["score_grid"], R["winner"], R["runner"]),
    use_container_width=True,
)

# ---- under the hood ----------------------------------------------------------
with st.expander("🔬 Under the hood — how good is it, really?"):
    t1, t2, t3 = st.tabs(["📈 Track record", "⚖️ Model vs market", "🎛️ What drives it"])

    with t1:
        bt_a = M["backtest"]["all"]
        st.markdown(
            f"We hid every match after **{M['backtest']['cutoff']}**, retrained on the "
            f"rest, and predicted those **{M['backtest']['test_n']:,}** unseen games."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Picked the winner", f"{bt_c['winner']:.0f}%", help="3-way (win/draw/loss) on competitive matches.")
        m2.metric("Exact scoreline", f"{bt_c['exact']:.0f}%", help="Nailing the precise result is hard — ~10% is normal for football.")
        m3.metric("Goal error (MAE)", f"{bt_c['mae']:.2f}", help="Average goals-per-team miss.")
        m4.metric("vs. Elo baseline", f"+{bt_c['model2'] - bt_c['elo2']:.1f} pts",
                  help=f"Head-to-head calls: model {bt_c['model2']:.0f}% vs a simple Elo baseline {bt_c['elo2']:.0f}%.")
        st.caption(
            "Honest read: on head-to-head calls we're about level with a simple Elo "
            "baseline — football is genuinely hard to predict. The model earns its keep "
            "in the *scorelines* it rolls, which is what makes the full-tournament "
            "simulation realistic."
        )

    with t2:
        st.altair_chart(
            charts.model_vs_market_scatter(R["title"], R["market"], R["ranked"]),
            use_container_width=True,
        )
        st.caption(
            "Each dot is a team: our simulated title chance vs the betting market's. "
            "On the dashed line we agree; above it we're more bullish than the market, "
            "below it more bearish. The sidebar's *market trust* slider pulls the dots "
            "toward the line."
        )

    with t3:
        st.altair_chart(
            charts.feature_importance_bar(M["feature_importance"]),
            use_container_width=True,
        )
        st.caption(
            "What moves the goals model most, measured by shuffling each input and "
            "watching accuracy drop. The gap in team strength dominates — exactly what "
            "you'd expect — with recent form and home advantage filling in the rest."
        )

st.caption("👈 Want more? Open **Details** in the sidebar for the full model "
           "vs. market table, group-stage stats, and every match result.")
