import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POWER_RANKINGS_CSV = "whl_power_rankings.csv"       
LINE_DISPARITY_CSV = "whl_line_disparity_all.csv"   
OUT_PNG = "DataScienceTeam.png"


rank_df = pd.read_csv(POWER_RANKINGS_CSV)

# Expect columns at least: team, strength
required_rank = {"team", "strength"}
missing = required_rank - set(rank_df.columns)
if missing:
    raise ValueError(f"{POWER_RANKINGS_CSV} missing columns: {sorted(missing)}")

disp_df = pd.read_csv(LINE_DISPARITY_CSV)

# Expect columns: team, disparity_ratio
required_disp = {"team", "disparity_ratio"}
missing = required_disp - set(disp_df.columns)
if missing:
    raise ValueError(f"{LINE_DISPARITY_CSV} missing columns: {sorted(missing)}")


plot_df = rank_df[["team", "strength"]].merge(
    disp_df[["team", "disparity_ratio"]],
    on="team",
    how="inner"
)

if plot_df.empty:
    raise ValueError("Merge resulted in 0 rows. Check that team names match in both CSVs.")


plt.figure(figsize=(10, 6))
plt.scatter(plot_df["disparity_ratio"], plot_df["strength"], alpha=0.85)

plt.title("Team Strength vs Offensive Line Quality Disparity (WHL 2025)")
plt.xlabel("Offensive line quality disparity = Line 1 / Line 2\n(Adjusted xG per 60)")
plt.ylabel("Team strength (Bradley–Terry logistic coefficient, centered)")

# Reference line: average strength = 0
plt.axhline(0, linewidth=1)


top_strength = plot_df.sort_values("strength", ascending=False).head(5)
top_disp = plot_df.sort_values("disparity_ratio", ascending=False).head(5)

label_df = pd.concat([top_strength, top_disp]).drop_duplicates(subset=["team"])

for _, r in label_df.iterrows():
    plt.annotate(
        r["team"],
        (r["disparity_ratio"], r["strength"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8
    )

plt.figtext(
    0.01, 0.01,
    "Each point is a team. Disparity > 1 means Line 1 generates more offense than Line 2.\n"
    "We examine whether teams with more balanced lines (lower disparity) tend to be stronger overall.",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(OUT_PNG, dpi=200)
plt.close()

print(f"Saved visualization: {OUT_PNG}")