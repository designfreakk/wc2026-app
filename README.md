# 🏆 World Cup 2026 Predictor — interactive app

A point-and-click web app that lets friends play with the Rung 5 v2 model:
trust the betting market more or less, knock players out with an injury dial,
choose how many tournaments to simulate, and watch the title race move.

It's fast because all the heavy model training is **precomputed** into
`artifacts.json`. At runtime the app only fetches live odds, applies your
tweaks, and rolls the Monte Carlo dice — so it needs just
`streamlit / pandas / numpy` (no scikit-learn, no statsmodels, no retraining).

## Files

| File | What it is |
|------|------------|
| `app.py` | The Streamlit UI (what your friends see) |
| `engine.py` | Fast runtime logic — fetch odds, calibrate, simulate |
| `artifacts.json` | Precomputed expected-goals for every matchup (the trained model, frozen) |
| `build_artifacts.py` | Re-run this to refresh `artifacts.json` from the latest results — needs sklearn + statsmodels |
| `data/` | Group fixtures + knockout bracket structure |
| `requirements.txt` | Runtime dependencies for deployment |

## Run it locally

```bash
cd rung5_ensemble/app
source ../../venv/bin/activate      # the shared workspace venv has streamlit
streamlit run app.py
```

It opens at http://localhost:8501. Set options in the sidebar, click **Run**.

## Share it with friends (free) — Streamlit Community Cloud

Streamlit Community Cloud hosts public apps for free, straight from a GitHub repo.

1. **Push to GitHub.** Create a repo and commit at least the `app/` folder
   (it must include `app.py`, `engine.py`, `artifacts.json`, `requirements.txt`,
   and the `data/` folder).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app**, pick your repo/branch, and set the main file path to
   `rung5_ensemble/app/app.py` (or `app/app.py` — whatever matches your repo layout).
4. Click **Deploy**. After a minute or two you get a public `…streamlit.app` URL
   you can send to anyone — no install needed on their end.

The live odds are fetched from Polymarket at runtime; if the host can't reach it,
the app automatically falls back to a bundled bookmaker snapshot.

## Refresh the model before the tournament

The model is frozen inside `artifacts.json`. To bake in newer match results,
re-run the builder (this needs the full data-science stack, so do it locally,
then commit the new `artifacts.json`):

```bash
source ../../venv/bin/activate
python build_artifacts.py
```

Live market odds are *not* baked in — they're pulled fresh every time the app
runs, so injury news that moves the odds is captured automatically.
