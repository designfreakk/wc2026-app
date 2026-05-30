"""Altair chart builders for the front-page 'how it works' visuals.

Altair ships with Streamlit, so these add no new runtime dependency. Each
function returns an alt.Chart ready for st.altair_chart(..., use_container_width=True).
"""
import numpy as np
import pandas as pd
import altair as alt

_GREEN = "#0b7"
_GRID = "#e5e7eb"


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
                        scale=alt.Scale(scheme="greens"), legend=alt.Legend(format=".0f")),
        tooltip=[alt.Tooltip("hg:O", title=f"{home}"), alt.Tooltip("ag:O", title=f"{away}"),
                 alt.Tooltip("p:Q", title="% chance", format=".1f")],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("p:Q", format=".0f"),
        color=alt.condition("datum.p > 6", alt.value("white"), alt.value("#475467")),
    )
    outline = base.transform_filter("datum.top").mark_rect(
        fill=None, stroke=_GREEN, strokeWidth=2.5)
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
    pts = alt.Chart(df).mark_circle(size=110, color=_GREEN, opacity=0.85).encode(
        x=alt.X("market:Q", title="Betting market title chance (%)",
                scale=alt.Scale(domain=[0, hi])),
        y=alt.Y("model:Q", title="Our model title chance (%)",
                scale=alt.Scale(domain=[0, hi])),
        tooltip=[alt.Tooltip("team:N", title="Team"),
                 alt.Tooltip("market:Q", title="Market %", format=".1f"),
                 alt.Tooltip("model:Q", title="Model %", format=".1f")],
    )
    labels = alt.Chart(df).mark_text(align="left", dx=7, fontSize=10, color="#475467").encode(
        x="market:Q", y="model:Q", text="team:N")
    return (line + pts + labels).properties(height=340)


def feature_importance_bar(items):
    """Horizontal bar of which inputs move the goals model most (relative)."""
    df = pd.DataFrame(items)
    return alt.Chart(df).mark_bar(color=_GREEN, cornerRadiusEnd=3).encode(
        x=alt.X("importance:Q", title="Relative influence", axis=alt.Axis(format=".0%")),
        y=alt.Y("feature:N", title=None, sort="-x"),
        tooltip=[alt.Tooltip("feature:N", title="Input"),
                 alt.Tooltip("importance:Q", title="Relative", format=".0%")],
    ).properties(height=max(180, 26 * len(df)))
