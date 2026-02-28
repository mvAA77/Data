

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

# ---------------------------
# Paths (update if needed)
# ---------------------------
WHL_CSV = "whl_2025.csv"
RND1_XLSX = "WHSDSC_Rnd1_matchups.xlsx"  # optional

# ---------------------------
# 1) Load + collapse to GAME level
# ---------------------------
df = pd.read_csv(WHL_CSV)

# Each game has many line/pairing rows; aggregate to a single game row
games = (
    df.groupby("game_id", as_index=False)
      .agg(
          home_team=("home_team", "first"),
          away_team=("away_team", "first"),
          home_goals=("home_goals", "sum"),
          away_goals=("away_goals", "sum"),
          home_xg=("home_xg", "sum"),
          away_xg=("away_xg", "sum"),
          went_ot=("went_ot", "first"),
      )
)

# Binary outcome: did home team win?
games["home_win"] = (games["home_goals"] > games["away_goals"]).astype(int)

# Sanity checks: 32 unique teams, no duplicates in list
teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
assert len(teams) == 32, f"Expected 32 teams, found {len(teams)}"

team_to_idx = {t: i for i, t in enumerate(teams)}
n_teams = len(teams)

# ---------------------------
# 2) Build Bradley–Terry design matrix
# ---------------------------
# Feature vector per game:
#   +1 at home team column
#   -1 at away team column
# plus one extra column for home-ice advantage (constant = 1)
X = np.zeros((len(games), n_teams + 1), dtype=np.float64)

home_idx = games["home_team"].map(team_to_idx).to_numpy()
away_idx = games["away_team"].map(team_to_idx).to_numpy()

rows = np.arange(len(games))
X[rows, home_idx] = 1.0
X[rows, away_idx] = -1.0
X[:, -1] = 1.0  # home-ice advantage feature

y = games["home_win"].to_numpy()

# ---------------------------
# 3) Fit logistic regression
# ---------------------------
# Very weak regularization (large C) to approximate unregularized BT model.
# (Pure unregularized isn't available directly in sklearn.)
model = LogisticRegression(
    penalty="l2",
    C=1e6,
    solver="lbfgs",
    max_iter=5000,
    fit_intercept=False,  # we already include home_adv as a feature
)
model.fit(X, y)

coefs = model.coef_.ravel()
team_strength = coefs[:n_teams]
home_adv = coefs[-1]

# Identifiability: center strengths to mean 0 (ranking unaffected)
team_strength = team_strength - team_strength.mean()

# ---------------------------
# 4) Evaluate model (quick diagnostics)
# ---------------------------
p_hat = model.predict_proba(X)[:, 1]
auc = roc_auc_score(y, p_hat)
ll = log_loss(y, p_hat)
acc = accuracy_score(y, (p_hat >= 0.5).astype(int))

print("Model diagnostics (on season games, in-sample):")
print(f"  AUC:      {auc:.3f}")
print(f"  LogLoss:  {ll:.3f}")
print(f"  Accuracy: {acc:.3f}")
print(f"  Home-ice advantage coefficient: {home_adv:.4f}\n")

# ---------------------------
# 5) Build Team Ranking Table
# ---------------------------
# Add some helpful descriptive stats (xG%, goal diff, win%)
# These do NOT determine rank (rank = team_strength), but help validate.
home_side = games[["home_team", "home_goals", "away_goals", "home_xg", "away_xg", "home_win"]].copy()
home_side.columns = ["team", "gf", "ga", "xgf", "xga", "win"]

away_side = games[["away_team", "away_goals", "home_goals", "away_xg", "home_xg", "home_win"]].copy()
away_side.columns = ["team", "gf", "ga", "xgf", "xga", "home_win"]
away_side["win"] = 1 - away_side["home_win"]
away_side = away_side.drop(columns=["home_win"])

team_stats = (
    pd.concat([home_side, away_side], ignore_index=True)
      .groupby("team", as_index=False)
      .agg(
          games=("team", "size"),
          wins=("win", "sum"),
          goals_for=("gf", "sum"),
          goals_against=("ga", "sum"),
          xg_for=("xgf", "sum"),
          xg_against=("xga", "sum"),
      )
)
team_stats["win_pct"] = team_stats["wins"] / team_stats["games"]
team_stats["goal_diff"] = team_stats["goals_for"] - team_stats["goals_against"]
team_stats["xg_diff"] = team_stats["xg_for"] - team_stats["xg_against"]
team_stats["xg_share"] = team_stats["xg_for"] / (team_stats["xg_for"] + team_stats["xg_against"])

strength_df = pd.DataFrame({"team": teams, "strength": team_strength})
ranking = team_stats.merge(strength_df, on="team", how="left")
ranking = ranking.sort_values("strength", ascending=False).reset_index(drop=True)
ranking["rank"] = np.arange(1, len(ranking) + 1)

# Display ranking (top 32)
print("=== WHL POWER RANKINGS (1 = best) ===")
print(ranking[["rank", "team", "strength", "win_pct", "xg_share", "xg_diff", "goal_diff"]].to_string(index=False))

# Save rankings for submission/use
ranking.to_csv("whl_power_rankings.csv", index=False)
print("\nSaved: whl_power_rankings.csv")

# ---------------------------
# 6) Predict Round 1 Matchup Win Probabilities (optional)
# ---------------------------
def win_prob(home_team: str, away_team: str) -> float:
    """Predict P(home wins) using fitted strengths + home_adv."""
    hi = team_to_idx[home_team]
    ai = team_to_idx[away_team]
    z = home_adv + team_strength[hi] - team_strength[ai]
    return float(1.0 / (1.0 + np.exp(-z)))

try:
    matchups = pd.read_excel(RND1_XLSX)
    # Expect columns: home_team, away_team (per provided file)
    matchups["home_win_prob"] = matchups.apply(lambda r: win_prob(r["home_team"], r["away_team"]), axis=1)
    matchups = matchups.sort_values("game").reset_index(drop=True)

    print("\n=== ROUND 1 HOME WIN PROBABILITIES ===")
    print(matchups[["game", "home_team", "away_team", "home_win_prob"]].to_string(index=False))

    matchups.to_csv("whl_round1_predictions.csv", index=False)
    print("\nSaved: whl_round1_predictions.csv")

except FileNotFoundError:
    print(f"\nNOTE: '{RND1_XLSX}' not found. Skipping Round 1 predictions.")
except Exception as e:
    print(f"\nNOTE: Could not generate Round 1 predictions due to: {e}")