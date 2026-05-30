"""Welcome / start-here page — plain-English explanation of the whole thing."""
import streamlit as st

st.set_page_config(page_title="Welcome · WC2026", page_icon="👋", layout="centered")

st.title("👋 Welcome — how this works")
st.markdown(
    "This little app guesses how the **2026 World Cup** might play out — who reaches "
    "the final, who lifts the trophy — and lets *you* play with the assumptions. "
    "No football knowledge needed. Here's the whole thing in plain English."
)

st.header("⚽ What it actually does")
st.markdown(
    """
- It **plays the entire tournament** — all 104 matches, group stage through the
  final — like a video game simulating itself.
- It doesn't play it *once*. It plays it **thousands of times**, because football is
  full of upsets and any single tournament could go a hundred different ways.
- Then it counts: *"In what fraction of those thousands of tournaments did each team
  win?"* That fraction is the team's **chance of winning** — the number you see in the
  title-race chart.
"""
)

st.header("🎲 The simulation, simply")
st.markdown(
    """
Think of each match as a pair of weighted dice — one per team — where stronger teams
roll more goals on average. For every simulated match the app rolls the dice to get a
scoreline, decides the winner (with a penalty shootout if it's a draw in the
knockouts), and moves the winners up the bracket. Do that for all 104 matches and
you've simulated one whole World Cup. Repeat a few thousand times and patterns emerge:
the favourites win *often*, the underdogs win *occasionally* — just like real life.

This "play it thousands of times and count" approach has a name: a **Monte Carlo
simulation**.
"""
)

st.header("🧠 The clever bit — how team strength is decided")
st.markdown(
    """
How "heavy" should each team's dice be? The app blends **three sources**:

1. **A football model trained on history.** It learned, from ~70 years of past
   international matches, how many goals teams tend to score and concede against
   different opponents. (Under the hood it's two models averaged together — a classic
   statistical one and a machine-learning one — but you don't need to care which.)

2. **The live betting market.** A prediction market called *Polymarket* shows what
   thousands of people are actually betting on right now. That "wisdom of the crowd" is
   surprisingly sharp, and it updates instantly when news breaks (like a star player
   getting injured). The app nudges its own opinion toward the market — and **you control
   how much** with the *"Trust the betting market"* slider.

3. **Your tweaks.** The *Injuries* control lets you knock a team down a notch if you
   think a key player will be missing.
"""
)

st.header("📚 The datasets it uses")
st.markdown(
    """
- **Historical match results** — a public, community-maintained database of roughly
  **49,000 international football matches** going back decades. This is what the model
  learns team strengths from.
- **Live market odds** — pulled fresh from the **Polymarket** prediction market every
  time the app runs, so it reflects the latest expectations (and injury news).
- **The official 2026 schedule** — the real group fixtures and the knockout bracket
  structure (who plays whom, and where the winners go next).

If the app can't reach the internet, it quietly falls back to a saved snapshot of the
odds so it still works.
"""
)

st.header("🎛️ How to use it")
st.markdown(
    """
Everything lives in the **left sidebar** on the main page:

- **Trust the betting market** — slide toward 0 for the pure football model, toward 1
  to lean on live odds. 0.5 is a balanced middle.
- **Tournaments to simulate** — more runs = smoother, more reliable percentages (but a
  bit slower). 2,000 takes about ten seconds.
- **Injuries** — pick a team and dial its strength down to test "what if their star is
  out?"
- **▶️ Run again** — re-rolls everything. The bracket shows *one plausible path to the
  final*, so it changes every time you run it.
"""
)

st.header("🙏 One honest caveat")
st.markdown(
    """
This is a bit of fun, not a crystal ball. It gives each team a **chance**, never a
certainty — and no model reliably beats the simple guess of *"the stronger team wins."*
Its real strength is realistic *scorelines* and *draw/penalty rates* across many games.
So enjoy the predictions, argue about them with your friends — but don't bet the house. 😄
"""
)

st.success("Ready? Head back to the **main page** (top of the sidebar) and hit "
           "**Run** to see your bracket. Curious about the numbers? The **📊 Details** "
           "page breaks them down.")
