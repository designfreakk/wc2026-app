"""Tab render helpers for the single-page app.

The Details breakdown used to be a separate sidebar-nav page (pages/2_Details).
It's now a plain function called inside st.tabs() on the front page, so the
whole app lives on one page with no sidebar.
"""
import pandas as pd
import streamlit as st


def render_details(R, ks):
    """Details tab — the per-team blend table, home/away/draw split, and the
    full knockout results behind the sample bracket. `R` is the results dict in
    session_state; `ks` is the knockout-slots dataframe (for round ordering)."""
    # ---- what settings produced this ----------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Market trust", f"{R['gamma']:.2f}")
    c2.metric("Tournaments simulated", f"{R['n_sims']:,}")
    c3.metric("Odds source", "live" if R["market_src"].startswith("live") else "offline")
    st.caption(f"Odds: {R['market_src']} · model built {R['meta'].get('built','?')} "
               f"(results: {R['meta'].get('results_source','?')})")

    # ---- model vs market vs your blend --------------------------------------
    st.subheader("Where your settings moved the odds")
    st.caption("The model has its own opinion; the live betting market has another. "
               "Your 'market trust' dial blends them — and injuries nudge it further.")
    rows = []
    for t in R["ranked"][:16]:
        rows.append({
            "Team": t,
            "Pure model": f"{R['mp'].get(t,0)*100:.1f}%",
            "Live market": f"{R['market'].get(t,0)*100:.1f}%",
            "Your blend": f"{R['title'][t]*100:.1f}%",
            "Injury dial": f"{R['avail'].get(t,1.0):.2f}" if t in R["avail"] else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ---- group-stage flavour -------------------------------------------------
    st.subheader("Group-stage results (across all sims)")
    hda = R["hda"]
    gtot = sum(hda.values()) or 1
    g1, g2, g3 = st.columns(3)
    g1.metric("Home wins", f"{hda['home']/gtot*100:.0f}%")
    g2.metric("Away wins", f"{hda['away']/gtot*100:.0f}%")
    g3.metric("Draws", f"{hda['draw']/gtot*100:.0f}%")

    # ---- full knockout results -----------------------------------------------
    st.subheader("Knockout results (the sample bracket on the Predictions tab)")
    ko = R["ko"]
    ROUND_ORDER = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final",
                   "Third-place playoff", "Final"]
    for rnd in ROUND_ORDER:
        mids = ks[ks["round"] == rnd]["match_id"].tolist()
        rows = []
        for m in mids:
            if m not in ko:
                continue
            h, a, hg, ag, ws, pens = ko[m]
            win = h if ws == "home" else a
            tag = "  ·  pens" if pens else ""
            rows.append({"Match": f"{h}  {hg}–{ag}  {a}{tag}", "Advances": f"➡️ {win}"})
        if rows:
            st.markdown(f"**{rnd}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "How it works: an ensemble football model (a weighted Poisson attack/defence "
        "model + a gradient-boosted goal model) sets each team's expected goals, blended "
        "with live prediction-market odds, then a full 104-match tournament is simulated "
        "thousands of times. It gives each team a *chance*, not a certainty — its real "
        "strength is calibrated scorelines and draw/penalty rates, not picking winners "
        "(no model reliably beats 'higher-ranked team wins')."
    )
