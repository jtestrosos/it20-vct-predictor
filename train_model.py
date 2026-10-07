# =========================================================
# VALORANT MATCH OUTCOME PREDICTION – MODEL TRAINING SCRIPT
# Adapted from the corrected pipeline for the Flask app.
# Run this once to produce outputs/valorant_rf_bundle.joblib
# =========================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from model import TeamSymmetric  # shared with app.py so pickle resolves correctly

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RANDOM_STATE = 42
MIN_AGENT_SUPPORT = 0

os.makedirs(OUT_DIR, exist_ok=True)
KEYS = ["Tournament", "Stage", "Match Type", "Match Name", "Map"]

# ---------------------------------------------------------
# 1. DATA PREPROCESSING (one row per map)
# ---------------------------------------------------------
print("Loading and preprocessing datasets...")
df_overview = pd.read_csv(os.path.join(DATA_DIR, "overview_per_map_cleaned.csv"))
df_scores = pd.read_csv(os.path.join(DATA_DIR, "maps_scores_cleaned.csv"), encoding="utf-8")
df_draft = pd.read_csv(os.path.join(DATA_DIR, "draft_phase.csv"))

# 1.1 Base table
base = df_scores[KEYS + ["Team A", "Team B", "Team A Score", "Team B Score"]].copy()
for c in ["Team A Score", "Team B Score"]:
    base[c] = pd.to_numeric(base[c], errors="coerce")
base = base.dropna(subset=["Team A Score", "Team B Score"])
base = base.drop_duplicates(subset=KEYS)
base["y"] = (base["Team A Score"] > base["Team B Score"]).astype(int)

# 1.2 Map pick advantage
df_draft["Map"] = df_draft["Map"].str.strip()
picks = df_draft.loc[df_draft["Action"] == "pick", KEYS + ["Team"]].rename(columns={"Team": "Picker"})
picks = picks.drop_duplicates(subset=KEYS)
base = base.merge(picks, on=KEYS, how="left")
base["Pick_Advantage"] = np.select(
    [base["Picker"] == base["Team A"], base["Picker"] == base["Team B"]], [1, -1], default=0
)

# 1.3 Team rosters
ov = df_overview[df_overview["Side"] == "both"]
rosters = ov.groupby(KEYS + ["Team"])["Agents"].apply(list).reset_index()

ra = rosters.rename(columns={"Team": "Team A", "Agents": "A_agents"})
rb = rosters.rename(columns={"Team": "Team B", "Agents": "B_agents"})
base = base.merge(ra, on=KEYS + ["Team A"], how="inner")
base = base.merge(rb, on=KEYS + ["Team B"], how="inner")
base = base.reset_index(drop=True)

# 1.4 Agent features
all_agents = sorted({a for l in base["A_agents"] for a in l} | {a for l in base["B_agents"] for a in l})
agent_cols = {}
for a in all_agents:
    in_a = base["A_agents"].apply(lambda l: int(a in l))
    in_b = base["B_agents"].apply(lambda l: int(a in l))
    agent_cols[f"Agent_{a}"] = in_a - in_b
agents_df = pd.DataFrame(agent_cols)
support = (agents_df != 0).sum()
rare = support[support < MIN_AGENT_SUPPORT].index.tolist()
agents_df = agents_df.drop(columns=rare)
print(f"Dropped {len(rare)} rare agent features (<{MIN_AGENT_SUPPORT} maps): {rare}")

# 1.5 Map one-hot
maps_df = pd.get_dummies(base["Map"], prefix="Map").astype(int)

# 1.6 Final matrices
X = pd.concat([base[["Pick_Advantage"]], agents_df, maps_df], axis=1).astype(float)
y = base["y"]
flip_idx = [i for i, c in enumerate(X.columns) if c == "Pick_Advantage" or c.startswith("Agent_")]

print(f"Maps: {len(X)} | Features: {X.shape[1]}")
print(f"Team A win rate: {y.mean():.1%}")


# ---------------------------------------------------------
# 3. TRAIN AND SAVE
# ---------------------------------------------------------
model = TeamSymmetric(
    RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                           random_state=RANDOM_STATE, n_jobs=-1),
    flip_idx
)
model.fit(X, y)

agent_stats = {}
for a in all_agents:
    played_a = base[base["A_agents"].apply(lambda l: a in l)]
    played_b = base[base["B_agents"].apply(lambda l: a in l)]
    wins_a = played_a["y"].sum()
    wins_b = (1 - played_b["y"]).sum()
    total_played = len(played_a) + len(played_b)
    total_wins = wins_a + wins_b
    
    # Bayesian smoothing: add 5 wins and 10 total games to bring low-sample agents closer to 50%
    smoothed_wins = total_wins + 5
    smoothed_played = total_played + 10
    
    agent_stats[a] = round((smoothed_wins / smoothed_played) * 100, 1)

bundle = {
    "model": model,
    "features": list(X.columns),
    "agents": [c[6:] for c in X.columns if c.startswith("Agent_")],
    "maps": [c[4:] for c in X.columns if c.startswith("Map_")],
    "agent_winrates": agent_stats
}
out_path = os.path.join(OUT_DIR, "valorant_rf_bundle.joblib")
joblib.dump(bundle, out_path)
print(f"\nModel saved to {out_path}")
print(f"Agents in model: {bundle['agents']}")
print(f"Maps in model:   {bundle['maps']}")
