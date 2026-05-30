"""Interactive World Cup 2026 predictor — no-sidebar, tab-based layout.

Controls live in a compact top bar (gamma + n_sims inline; injuries in a
popover). Content is split across three tabs so nothing is buried. Heavy
model training is precomputed into artifacts.json; only streamlit / pandas /
numpy / altair are needed at runtime.
"""
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import bracket
import charts
import engine
import views

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")

# Press Start 2P is a pixel font — wide and tall. Shrink heading sizes, add
# breathing room, and let long titles wrap so nothing overflows (esp. mobile).
# Also collapse the empty sidebar rail entirely.
st.markdown(
    """
    <style>
      /* hide the sidebar toggle + rail */
      section[data-testid="stSidebar"],
      button[data-testid="collapsedControl"] { display: none !important; }
      /* pixel-font guard-rails */
      h1, h2, h3, h4 { line-height: 1.5 !important; overflow-wrap: anywhere; }
      h1 { font-size: 1.7rem !important; }
      h2 { font-size: 1.15rem !important; }
      h3 { font-size: 0.95rem !important; }
      h4 { font-size: 0.8rem !important; }
      @media (max-width: 760px){
        h1 { font-size: 1.2rem !important; }
        h2 { font-size: 0.95rem !important; }
        h3, h4 { font-size: 0.8rem !important; }
      }
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
    """Run the two-pass model and stash everything for both pages."""
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
        "agg": agg,
    }


teams, base, meta, gf, ks = get_artifacts()
market, market_src = get_market(teams)


# ---- page header + control bar -----------------------------------------------
st.title("🏆 World Cup 2026 — who wins?")
st.markdown(
    "We simulate the entire tournament thousands of times with a football model "
    "blended with the live betting market, then count how often each team lifts "
    "the trophy. Tweak the controls and hit **Run** to change the story."
)

c_gamma, c_sims, c_inj, c_run = st.columns([4, 3, 2, 1])

with c_gamma:
    gamma = st.slider(
        "📊 Market trust", 0.0, 1.0, 0.5, 0.05,
        help="0 = pure football model · 1 = lean hard on live odds · 0.5 is balanced.",
    )

with c_sims:
    n_sims = st.select_slider(
        "🎲 Simulations", options=[300, 500, 1000, 2000, 3000], value=1000,
        help="More = smoother odds but slower. 1 000 is a good balance.",
    )

with c_inj:
    ranked_by_market = sorted(teams, key=lambda t: market.get(t, 0), reverse=True)
    with st.popover("🤕 Injuries", use_container_width=True):
        st.caption("Knock a team down a notch if a star is out. 1.00 = full strength.")
        picks = st.multiselect("Teams to weaken", ranked_by_market, default=[])
        avail = {t: st.slider(t, 0.80, 1.05, 1.00, 0.01, key=f"av_{t}") for t in picks}

with c_run:
    run = st.button("▶️ Run", type="primary", use_container_width=True)

# avail is {} when picks is empty (dict comprehension over empty list)
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
M = R["meta"]

st.divider()

# ---- content tabs ------------------------------------------------------------
tab_pred, tab_group, tab_model, tab_details = st.tabs(
    ["🏆 Predictions", "📊 Group Stage", "🔬 How it works", "📋 Details"]
)

# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown(
        f"#### Most likely champion: **{R['winner']}** "
        f"({R['title'][R['winner']]*100:.1f}% of simulations)"
    )
    st.caption(
        f"One plausible run all the way to the final, where {R['winner']} lift the "
        "trophy. Tweak the controls and run it again — the path changes every time."
    )
    # Bracket scales itself to fit the iframe width and reports its true height
    # back to Streamlit (see bracket.py), so it fits any window with no scrollbar.
    components.html(bracket.render(R["ko"], height=640), height=690, scrolling=False)

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

    st.subheader("How far does each team go?")
    st.caption(
        "Share of simulations in which each team reaches each knockout round. Read "
        "left to right — the deeper the green, the more likely. Top 24 by title odds."
    )
    st.dataframe(
        charts.advancement_table(R["agg"], R["ranked"], top=24),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
with tab_group:
    st.subheader("Group stage — who escapes their group?")
    st.caption(
        "For all 12 groups: chance of winning the group, finishing top two, and "
        "advancing to the knockouts (top two plus the best third-placed teams)."
    )
    st.dataframe(
        charts.group_finish_table(R["agg"]),
        use_container_width=True, height=520,
    )

# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.subheader("How it works")
    st.markdown(
        "We start with a football model trained on every international match since "
        f"{M['train_from'][:4]} — about {M['train_matches']:,} games. It's really *two* "
        "models averaged: a classic **Poisson** goals model and a **machine-learning** "
        "one, each predicting how many goals a team should score given strength, recent "
        "form and home advantage.\n\n"
        "We then nudge each team toward the **live betting market** (you control how hard "
        "with the slider above), because bookmakers price in news the raw stats miss — "
        "injuries, form, momentum.\n\n"
        "Finally we **play the whole tournament** — all 104 matches, groups to final — "
        "thousands of times, rolling realistic scorelines each game (penalty shootouts "
        "included). How often each team wins is the percentage you see above."
    )

    c1, c2, c3, c4 = st.columns(4)
    bt_c = M["backtest"]["competitive"]
    c1.metric("Tournaments simulated", f"{R['n_sims']:,}",
              help="Per run, using the current controls.")
    c2.metric("Matches trained on", f"{M['train_matches']:,}",
              help=f"Internationals from {M['train_from']} to {M['train_to']}.")
    c3.metric("Winner accuracy", f"{bt_c['winner']:.0f}%",
              help=f"Out-of-sample on {bt_c['n']} competitive matches the model never saw.")
    c4.metric("Goal error (MAE)", f"{bt_c['mae']:.2f}",
              help="Average miss in goals-per-team on unseen matches. Lower is better.")

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

    st.subheader("Under the hood — how good is it, really?")
    t1, t2, t3 = st.tabs(["📈 Track record", "⚖️ Model vs market", "🎛️ What drives it"])

    with t1:
        bt_a = M["backtest"]["all"]
        st.markdown(
            f"We hid every match after **{M['backtest']['cutoff']}**, retrained on the "
            f"rest, and predicted those **{M['backtest']['test_n']:,}** unseen games."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Picked the winner", f"{bt_c['winner']:.0f}%",
                  help="3-way (win/draw/loss) on competitive matches.")
        m2.metric("Exact scoreline", f"{bt_c['exact']:.0f}%",
                  help="Nailing the precise result is hard — ~10% is normal for football.")
        m3.metric("Goal error (MAE)", f"{bt_c['mae']:.2f}",
                  help="Average goals-per-team miss.")
        m4.metric("vs. Elo baseline", f"+{bt_c['model2'] - bt_c['elo2']:.1f} pts",
                  help=f"Head-to-head calls: model {bt_c['model2']:.0f}% vs Elo baseline {bt_c['elo2']:.0f}%.")
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
            "below it more bearish. The market-trust slider pulls the dots toward the line."
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

# ══════════════════════════════════════════════════════════════════════════════
with tab_details:
    views.render_details(R, ks)
