import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

WHL_CSV = "whl_2025.csv"
RND1_XLSX = "WHSDSC_Rnd1_matchups.xlsx"  # optional

df = pd.read_csv(WHL_CSV)

# Collapse to GAME level
games = (
    df.groupby("game_id", as_index=False)
      .agg(
          home_team=("home_team", "first"),
          away_team=("away_team", "first"),
          home_goals=("home_goals", "sum"),
          away_goals=("away_goals", "sum"),
          home_xg=("home_xg", "sum"),
          away_xg=("away_xg", "sum"),
          went_ot=("went_ot", "max"),
      )
)

# Binary outcome: did home team win?
games = games[games["home_goals"] != games["away_goals"]].copy()
games["home_win"] = (games["home_goals"] > games["away_goals"]).astype(int)

teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
assert len(teams) == 32, f"Expected 32 teams, found {len(teams)}"

team_to_idx = {t: i for i, t in enumerate(teams)}
n_teams = len(teams)

# Build Bradley–Terry design matrix (+ home-ice feature)
X = np.zeros((len(games), n_teams + 1), dtype=np.float64)

home_idx = games["home_team"].map(team_to_idx).to_numpy()
away_idx = games["away_team"].map(team_to_idx).to_numpy()
rows = np.arange(len(games))

X[rows, home_idx] = 1.0
X[rows, away_idx] = -1.0
X[:, -1] = 1.0  # home-ice feature

y = games["home_win"].to_numpy()

# Fit logistic regression (approx unregularized BT)
model = LogisticRegression(
    penalty="l2",
    C=1e6,
    solver="lbfgs",
    max_iter=5000,
    fit_intercept=False
)
model.fit(X, y)

coefs = model.coef_.ravel()
team_strength = coefs[:n_teams]
home_adv = float(coefs[-1])

# Center strengths for identifiability (rank unchanged)
team_strength = team_strength - team_strength.mean()

# Diagnostics
p_hat = model.predict_proba(X)[:, 1]
auc = roc_auc_score(y, p_hat)
ll = log_loss(y, p_hat)
acc = accuracy_score(y, (p_hat >= 0.5).astype(int))

print("Model diagnostics (in-sample):")
print(f"  AUC:      {auc:.3f}")
print(f"  LogLoss:  {ll:.3f}")
print(f"  Accuracy: {acc:.3f}")
print(f"  Home-ice advantage coefficient: {home_adv:.4f}\n")

# Power rankings
strength_df = pd.DataFrame({"team": teams, "strength": team_strength})
ranking = strength_df.sort_values("strength", ascending=False).reset_index(drop=True)
ranking["rank"] = np.arange(1, len(ranking) + 1)

print("=== WHL POWER RANKINGS (1 = best) ===")
print(ranking[["rank", "team", "strength"]].to_string(index=False))

ranking.to_csv("whl_power_rankings.csv", index=False)
print("\nSaved: whl_power_rankings.csv")

# Round 1 predictions
def win_prob(home_team: str, away_team: str) -> float:
    hi = team_to_idx[home_team]
    ai = team_to_idx[away_team]
    z = home_adv + team_strength[hi] - team_strength[ai]
    return float(1 / (1 + np.exp(-z)))

try:
    matchups = pd.read_excel(RND1_XLSX)
    matchups["home_win_prob"] = matchups.apply(lambda r: win_prob(r["home_team"], r["away_team"]), axis=1)
    if "game" in matchups.columns:
        matchups = matchups.sort_values("game")
    matchups = matchups.reset_index(drop=True)

    print("\n=== ROUND 1 HOME WIN PROBABILITIES ===")
    cols = ["home_team", "away_team", "home_win_prob"]
    if "game" in matchups.columns:
        cols = ["game"] + cols
    print(matchups[cols].to_string(index=False))

    matchups.to_csv("whl_round1_predictions.csv", index=False)
    print("\nSaved: whl_round1_predictions.csv")

except FileNotFoundError:
    print(f"\nNOTE: '{RND1_XLSX}' not found. Skipping Round 1 predictions.")