"""Altair chart builders for the front-page 'how it works' visuals.

Altair ships with Streamlit, so these add no new runtime dependency. Each
function returns an alt.Chart ready for st.altair_chart(..., use_container_width=True).
"""
import numpy as np
import pandas as pd
import altair as alt

# getfast.ai-inspired palette: a single vivid red carries all meaning; near-black
# ink and cool greys stay as the neutral text & paper on a white canvas.
_AMBER = "#fc0017"   # signature red — bars, highlights, light gradient end
_BURNT = "#c5394e"   # deeper red — emphasis marks, strong gradient end
_NAVY = "#212529"    # neutral ink — text, axes, outlines
_GRID = "#dee2e6"    # cool-grey gridlines


def poisson_heatmap(grid, home, away, kmax=6):
    """Scoreline probability heatmap: P(home i : away j) from the Poisson grid.

    `grid[i, j]` is the chance of home scoring i and away scoring j. We render it
    as a labelled matrix with the model's single most likely scoreline highlighted.
    """
    rows = []
    gi, gj = np.unravel_index(int(np.argmax(grid)), grid.shape)  # modal scoreline
    for i in range(kmax + 1):
        for j in range(kmax + 1):
            rows.append({"hg": i, "ag": j, "p": float(grid[i, j]) * 100,
                         "top": (i == gi and j == gj)})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("ag:O", title=f"{away} goals", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("hg:O", title=f"{home} goals", sort="descending"),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("p:Q", title="% chance",
                        scale=alt.Scale(scheme="reds"), legend=alt.Legend(format=".0f")),
        tooltip=[alt.Tooltip("hg:O", title=f"{home}"), alt.Tooltip("ag:O", title=f"{away}"),
                 alt.Tooltip("p:Q", title="% chance", format=".1f")],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("p:Q", format=".0f"),
        color=alt.condition("datum.p > 6", alt.value("white"), alt.value("#5b5b73")),
    )
    # navy outline so the modal cell pops against the orange scale
    outline = base.transform_filter("datum.top").mark_rect(
        fill=None, stroke=_NAVY, strokeWidth=2.5)
    return (cells + text + outline).properties(height=300)


def model_vs_market_scatter(title_probs, market_probs, teams, top=18):
    """Each team as a point: our simulated title chance (y) vs the betting
    market's implied chance (x). The dashed line is perfect agreement; points
    above it are teams the model rates higher than the market, and vice versa.
    """
    ranked = sorted(teams, key=lambda t: market_probs.get(t, 0), reverse=True)[:top]
    df = pd.DataFrame({
        "team": ranked,
        "market": [market_probs.get(t, 0) * 100 for t in ranked],
        "model": [title_probs.get(t, 0) * 100 for t in ranked],
    })
    hi = max(df["market"].max(), df["model"].max()) * 1.1 + 1
    diag = pd.DataFrame({"x": [0, hi], "y": [0, hi]})
    line = alt.Chart(diag).mark_line(strokeDash=[5, 5], color=_GRID).encode(x="x:Q", y="y:Q")
    pts = alt.Chart(df).mark_circle(size=110, color=_BURNT, opacity=0.85).encode(
        x=alt.X("market:Q", title="Betting market title chance (%)",
                scale=alt.Scale(domain=[0, hi])),
        y=alt.Y("model:Q", title="Our model title chance (%)",
                scale=alt.Scale(domain=[0, hi])),
        tooltip=[alt.Tooltip("team:N", title="Team"),
                 alt.Tooltip("market:Q", title="Market %", format=".1f"),
                 alt.Tooltip("model:Q", title="Model %", format=".1f")],
    )
    labels = alt.Chart(df).mark_text(align="left", dx=7, fontSize=10, color="#5b5b73").encode(
        x="market:Q", y="model:Q", text="team:N")
    return (line + pts + labels).properties(height=340)


def _orange_css(v):
    """Inline-CSS red shade for a 0-100 percentage — a matplotlib-free stand-in
    for Styler.background_gradient (matplotlib isn't a runtime dependency).
    Near-white at 0% deepening to full red at 100%."""
    try:
        f = max(0.0, min(1.0, float(v) / 100.0))
    except (TypeError, ValueError):
        return ""
    txt = "#ffffff" if f > 0.55 else "#212529"
    return f"background-color: rgba(252,0,23,{0.05 + 0.85 * f:.3f}); color:{txt};"


_REACH_COLS = [("Round of 32", "Advance"), ("Round of 16", "Round of 16"),
               ("Quarter-final", "Quarters"), ("Semi-final", "Semis"),
               ("Final", "Final"), ("Champion", "Win it")]


def advancement_table(agg, ranked, top=24):
    """FiveThirtyEight-style 'how far does each team go' table: one row per team,
    columns = chance of reaching each knockout round, shaded on an amber gradient.
    Returns a pandas Styler ready for st.dataframe."""
    n = agg["n"] or 1
    reach = agg["reach"]
    teams = ranked[:top]
    data = {label: [100 * reach[t].get(key, 0) / n for t in teams]
            for key, label in _REACH_COLS}
    df = pd.DataFrame(data, index=teams)
    df.index.name = "Team"
    cols = [label for _, label in _REACH_COLS]
    return (df.style
            .format("{:.0f}%")
            .map(_orange_css, subset=cols))


def group_finish_table(agg):
    """Per-group escape odds: win the group, finish top 2, and overall chance of
    reaching the knockouts (top 2 + best-third). Sorted group, then by win-group %.
    Returns a pandas Styler."""
    n = agg["n"] or 1
    fin, reach, grp = agg["finish"], agg["reach"], agg["group"]
    rows = []
    for t, g in grp.items():
        rows.append({
            "Group": g, "Team": t,
            "Win group": 100 * fin[t].get(1, 0) / n,
            "Top 2": 100 * (fin[t].get(1, 0) + fin[t].get(2, 0)) / n,
            "Advance": 100 * reach[t].get("Round of 32", 0) / n,
        })
    df = (pd.DataFrame(rows)
          .sort_values(["Group", "Win group"], ascending=[True, False])
          .reset_index(drop=True))
    pct = ["Win group", "Top 2", "Advance"]
    return (df.style
            .hide(axis="index")
            .format({c: "{:.0f}%" for c in pct})
            .map(_orange_css, subset=pct))


def feature_importance_bar(items):
    """Horizontal bar of which inputs move the goals model most (relative)."""
    df = pd.DataFrame(items)
    return alt.Chart(df).mark_bar(color=_AMBER, cornerRadiusEnd=3).encode(
        x=alt.X("importance:Q", title="Relative influence", axis=alt.Axis(format=".0%")),
        y=alt.Y("feature:N", title=None, sort="-x"),
        tooltip=[alt.Tooltip("feature:N", title="Input"),
                 alt.Tooltip("importance:Q", title="Relative", format=".0%")],
    ).properties(height=max(180, 26 * len(df)))
