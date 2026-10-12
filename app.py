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


def get_role_details(team_agents):
    """Return detailed breakdown of role coverage and penalties for a team."""
    team_roles = set(ROLES_MAP.get(a.lower(), "Others") for a in team_agents)
    standard_roles = ["Controllers", "Duelists", "Initiators", "Sentinels"]
    present_roles = [r for r in standard_roles if r in team_roles]
    missing_roles = [r for r in standard_roles if r not in team_roles]

    atk_mult = 1.0
    missing_atk = []
    for role, factor in ATTACK_PENALTIES.items():
        if role not in team_roles:
            atk_mult *= factor
            missing_atk.append({"role": role, "penalty": round((1.0 - factor) * 100, 1)})

    dfn_mult = 1.0
    missing_dfn = []
    for role, factor in DEFENSE_PENALTIES.items():
        if role not in team_roles:
            dfn_mult *= factor
            missing_dfn.append({"role": role, "penalty": round((1.0 - factor) * 100, 1)})

    composite = (atk_mult + dfn_mult) / 2.0

    return {
        "attack_mult": round(atk_mult * 100, 1),
        "defense_mult": round(dfn_mult * 100, 1),
        "composite": composite,
        "composite_pct": round(composite * 100, 1),
        "present_roles": present_roles,
        "missing_roles": missing_roles,
        "missing_atk": missing_atk,
        "missing_dfn": missing_dfn,
    }


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


def get_tree_analysis(team_a_agents, team_b_agents, map_name, pick_advantage=0):
    """Extract voting consensus and probability distribution across the 300 decision trees."""
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

    df = pd.DataFrame([row])[b["features"]]
    Xv = np.asarray(df, dtype=float)
    model = b["model"]
    Xs = model._swap(Xv)

    tree_probs = [
        float(0.5 * (t.predict_proba(Xv)[0, 1] + 1.0 - t.predict_proba(Xs)[0, 1]))
        for t in model.estimator_.estimators_
    ]

    votes_a = sum(1 for p in tree_probs if p > 0.505)
    votes_b = sum(1 for p in tree_probs if p < 0.495)
    neutral = len(tree_probs) - votes_a - votes_b

    # Distribution buckets for chart histogram
    b1 = sum(1 for p in tree_probs if p < 0.35)
    b2 = sum(1 for p in tree_probs if 0.35 <= p < 0.48)
    b3 = sum(1 for p in tree_probs if 0.48 <= p <= 0.52)
    b4 = sum(1 for p in tree_probs if 0.52 < p <= 0.65)
    b5 = sum(1 for p in tree_probs if p > 0.65)

    return {
        "votes_a": votes_a,
        "votes_b": votes_b,
        "neutral": neutral,
        "total": len(tree_probs),
        "pct_a": round((votes_a / len(tree_probs)) * 100, 1),
        "pct_b": round((votes_b / len(tree_probs)) * 100, 1),
        "pct_neutral": round((neutral / len(tree_probs)) * 100, 1),
        "buckets": [b1, b2, b3, b4, b5],
        "bucket_labels": ["< 35% (Team B Favored)", "35–48% (Lean Team B)", "48–52% (Dead Heat)", "52–65% (Lean Team A)", "> 65% (Team A Favored)"]
    }


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
    """AJAX endpoint - returns JSON with the prediction and detailed estimation breakdown."""
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

    raw_rf_prob_a = predict_win_probability(team_a, team_b, map_name, pick_adv)
    raw_rf_prob_b = 1.0 - raw_rf_prob_a

    # Extract agent win rates for adjustment
    bundle_data = get_bundle()
    winrates = bundle_data.get("agent_winrates", {})

    agents_a_list = [
        {"name": a, "winrate": winrates.get(a.lower(), 50.0), "role": ROLES_MAP.get(a.lower(), "Others")}
        for a in team_a
    ]
    agents_b_list = [
        {"name": a, "winrate": winrates.get(a.lower(), 50.0), "role": ROLES_MAP.get(a.lower(), "Others")}
        for a in team_b
    ]

    avg_wr_a = sum(x["winrate"] for x in agents_a_list) / 5.0
    avg_wr_b = sum(x["winrate"] for x in agents_b_list) / 5.0
    wr_delta = avg_wr_a - avg_wr_b

    # The difference in average win rates scales the probability (scaled by 50% so it doesn't overpower the RF model)
    wr_modifier = ((avg_wr_a - avg_wr_b) / 100.0) * 0.5

    # Apply context-aware role multipliers (attack + defense averaged)
    role_a = get_role_details(team_a)
    role_b = get_role_details(team_b)
    mult_a = role_a["composite"]
    mult_b = role_b["composite"]

    # Scale the base probability by the ratio of team multipliers.
    if mult_a + mult_b > 0:
        role_ratio = mult_a / (mult_a + mult_b)   # maps to 0-1 range
    else:
        role_ratio = 0.5

    # Shift components from baseline RF
    shift_wr = wr_modifier * 0.2
    shift_role = (role_ratio - raw_rf_prob_a) * 0.2

    # Blend: 60% RF model + 20% role context + 20% win-rate modifier headroom
    prob_a = raw_rf_prob_a * 0.6 + role_ratio * 0.2 + (raw_rf_prob_a + wr_modifier) * 0.2
    prob_a = max(0.01, min(0.99, prob_a))
    prob_b = 1.0 - prob_a

    tree_info = get_tree_analysis(team_a, team_b, map_name, pick_adv)

    return jsonify({
        "team_a_prob": round(prob_a * 100, 1),
        "team_b_prob": round(prob_b * 100, 1),
        "winner": "Team A" if prob_a > 0.5 else "Team B",
        "confidence": round(max(prob_a, prob_b) * 100, 1),
        "breakdown": {
            "rf_base_a": round(raw_rf_prob_a * 100, 1),
            "rf_base_b": round(raw_rf_prob_b * 100, 1),
            "shift_wr": round(shift_wr * 100, 2),
            "shift_role": round(shift_role * 100, 2),
            "final_prob_a": round(prob_a * 100, 1),
            "final_prob_b": round(prob_b * 100, 1),
            "avg_wr_a": round(avg_wr_a, 1),
            "avg_wr_b": round(avg_wr_b, 1),
            "wr_delta": round(wr_delta, 1),
            "agents_a": agents_a_list,
            "agents_b": agents_b_list,
            "role_a": role_a,
            "role_b": role_b,
            "role_ratio_a": round(role_ratio * 100, 1),
            "role_ratio_b": round((1.0 - role_ratio) * 100, 1),
            "tree_info": tree_info,
        }
    })


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
