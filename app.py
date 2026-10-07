import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from model import TeamSymmetric  # noqa: F401 - needed for pickle to resolve the class

app = Flask(__name__)
app.secret_key = "vct_predictor_secret_key"

# ---------------------------------------------------------------------------
# Roles Configuration
# ---------------------------------------------------------------------------
ROLES_MAP = {
    "astra": "Controllers", "brimstone": "Controllers", "harbor": "Controllers", "omen": "Controllers", "viper": "Controllers", "clove": "Controllers", "miks": "Controllers",
    "jett": "Duelists", "neon": "Duelists", "phoenix": "Duelists", "raze": "Duelists", "yoru": "Duelists", "waylay": "Duelists", "iso": "Duelists", "reyna": "Duelists",
    "breach": "Initiators", "fade": "Initiators", "kayo": "Initiators", "skye": "Initiators", "sova": "Initiators", "tejo": "Initiators", "gekko": "Initiators",
    "chamber": "Sentinels", "cypher": "Sentinels", "killjoy": "Sentinels", "sage": "Sentinels", "vyse": "Sentinels", "deadlock": "Sentinels", "veto": "Sentinels"
}

# Multiplicative penalties per side when a role is entirely missing.
# Values < 1.0 reduce the team's expected performance on that half.
ATTACK_PENALTIES = {
    "Controllers": 0.75,   # Extreme - can't block sightlines for site takes
    "Initiators":  0.80,   # Extreme - blind entries, walk into traps
    "Duelists":    0.90,   # Moderate-High - stalls at choke points
    "Sentinels":   0.95,   # Low-Moderate - mainly hurts post-plant / flank watch
}

DEFENSE_PENALTIES = {
    "Sentinels":   0.75,   # Extreme - no stall for rotations
    "Controllers": 0.85,   # High - can't isolate choke points
    "Initiators":  0.90,   # Moderate - retake info suffers
    "Duelists":    0.98,   # Low - barely hurts defensive anchoring
}


def _side_multiplier(team_agents, penalty_table):
    """Return the multiplicative penalty for one half (attack or defense)."""
    team_roles = set(ROLES_MAP.get(a.lower(), "Others") for a in team_agents)
    multiplier = 1.0
    for role, factor in penalty_table.items():
        if role not in team_roles:
            multiplier *= factor
    return multiplier


def calculate_role_penalty(team_agents):
    """Return a composite penalty (0-1 range, lower = worse) averaged across halves."""
    atk = _side_multiplier(team_agents, ATTACK_PENALTIES)
    dfn = _side_multiplier(team_agents, DEFENSE_PENALTIES)
    # Average both halves - in a standard match each team plays both sides
    return (atk + dfn) / 2.0

# ---------------------------------------------------------------------------
# Load the trained model bundle
# ---------------------------------------------------------------------------
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "outputs", "valorant_rf_bundle.joblib")
bundle = None


def get_bundle():
    """Lazy-load the model bundle so the app can start even if the file is missing."""
    global bundle
    if bundle is None:
        bundle = joblib.load(BUNDLE_PATH)
    return bundle


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_win_probability(team_a_agents, team_b_agents, map_name, pick_advantage=0):
    """Return P(Team A wins) given agent lists, map, and pick advantage."""
    b = get_bundle()
    row = dict.fromkeys(b["features"], 0.0)
    for a in team_a_agents:
        key = f"Agent_{a.lower()}"
        if key in row:
            row[key] += 1
    for a in team_b_agents:
        key = f"Agent_{a.lower()}"
        if key in row:
            row[key] -= 1
    row["Pick_Advantage"] = float(pick_advantage)
    map_key = f"Map_{map_name}"
    if map_key in row:
        row[map_key] = 1.0
    prob = float(b["model"].predict_proba(pd.DataFrame([row])[b["features"]])[0, 1])
    return prob


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    bundle_data = get_bundle()
    agents = sorted(bundle_data["agents"])
    grouped_agents = {"Controllers": [], "Duelists": [], "Initiators": [], "Sentinels": [], "Others": []}
    for a in agents:
        grouped_agents[ROLES_MAP.get(a, "Others")].append(a)
    grouped_agents = {k: v for k, v in grouped_agents.items() if v}

    return render_template(
        "index.html",
        grouped_agents=grouped_agents,
        maps=sorted(bundle_data["maps"]),
        agent_winrates=bundle_data.get("agent_winrates", {})
    )


@app.route("/predict", methods=["POST"])
def predict():
    """AJAX endpoint - returns JSON with the prediction."""
    data = request.get_json()

    team_a = data.get("team_a", [])
    team_b = data.get("team_b", [])
    map_name = data.get("map", "")
    pick_adv = int(data.get("pick_advantage", 0))

    # Validation
    errors = []
    if len(team_a) != 5:
        errors.append(f"Team A must have exactly 5 agents (got {len(team_a)}).")
    if len(team_b) != 5:
        errors.append(f"Team B must have exactly 5 agents (got {len(team_b)}).")
    if not map_name:
        errors.append("Please select a map.")
    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    prob_a = predict_win_probability(team_a, team_b, map_name, pick_adv)
    
    # Extract agent win rates for adjustment
    bundle_data = get_bundle()
    winrates = bundle_data.get("agent_winrates", {})
    avg_wr_a = sum(winrates.get(a.lower(), 50.0) for a in team_a) / 5.0
    avg_wr_b = sum(winrates.get(a.lower(), 50.0) for a in team_b) / 5.0
    
    # The difference in average win rates scales the probability (scaled by 50% so it doesn't overpower the RF model)
    wr_modifier = ((avg_wr_a - avg_wr_b) / 100.0) * 0.5
    
    # Apply context-aware role multipliers (attack + defense averaged)
    mult_a = calculate_role_penalty(team_a)   # 0-1, 1.0 = no penalty
    mult_b = calculate_role_penalty(team_b)

    # Scale the base probability by the ratio of team multipliers.
    # If both teams are balanced the ratio is ~1 and prob barely moves.
    # If Team A is missing roles, mult_a < mult_b → ratio < 1 → prob drops.
    if mult_a + mult_b > 0:
        role_ratio = mult_a / (mult_a + mult_b)   # maps to 0-1 range
    else:
        role_ratio = 0.5

    # Blend: 60% RF model + 20% role context + 20% win-rate modifier headroom
    prob_a = prob_a * 0.6 + role_ratio * 0.2 + (prob_a + wr_modifier) * 0.2
    prob_a = max(0.01, min(0.99, prob_a))
    prob_b = 1 - prob_a

    return jsonify({
        "team_a_prob": round(prob_a * 100, 1),
        "team_b_prob": round(prob_b * 100, 1),
        "winner": "Team A" if prob_a > 0.5 else "Team B",
        "confidence": round(max(prob_a, prob_b) * 100, 1),
    })


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
