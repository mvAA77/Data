import numpy as np
import pandas as pd

WHL_CSV = "whl_2025.csv"

OUT_ALL = "whl_line_disparity_all.csv"
OUT_TOP10 = "whl_line_disparity_top10.csv"


def build_def_pairing_and_goalie_tables(df: pd.DataFrame):
    """
    Build opponent-quality tables:
      - Defensive pairing xG allowed per 60
      - Goalie xG allowed per 60

    We treat xG allowed as:
      home_def_pairing allows away_xg
      away_def_pairing allows home_xg
    """
    # Defensive pairings
    home_def = df[["home_team", "home_def_pairing", "away_xg", "toi"]].copy()
    home_def.columns = ["team", "def_pairing", "xg_allowed", "toi"]

    away_def = df[["away_team", "away_def_pairing", "home_xg", "toi"]].copy()
    away_def.columns = ["team", "def_pairing", "xg_allowed", "toi"]

    defs = pd.concat([home_def, away_def], ignore_index=True)
    defs = defs[defs["toi"] > 0].copy()
    defs["minutes"] = defs["toi"] / 60.0

    def_tbl = (
        defs.groupby(["team", "def_pairing"], as_index=False)
            .agg(xg_allowed=("xg_allowed", "sum"),
                 minutes=("minutes", "sum"))
    )
    def_tbl["xg_allowed_per60"] = def_tbl["xg_allowed"] / (def_tbl["minutes"] / 60.0)

    # Goalies
    home_g = df[["home_goalie", "away_xg", "toi"]].copy()
    home_g.columns = ["goalie", "xg_allowed", "toi"]

    away_g = df[["away_goalie", "home_xg", "toi"]].copy()
    away_g.columns = ["goalie", "xg_allowed", "toi"]

    goalies = pd.concat([home_g, away_g], ignore_index=True)
    goalies = goalies[goalies["toi"] > 0].copy()
    goalies["minutes"] = goalies["toi"] / 60.0

    g_tbl = (
        goalies.groupby("goalie", as_index=False)
              .agg(xg_allowed=("xg_allowed", "sum"),
                   minutes=("minutes", "sum"))
    )
    g_tbl["xg_allowed_per60"] = g_tbl["xg_allowed"] / (g_tbl["minutes"] / 60.0)

    return def_tbl, g_tbl


def compute_offensive_line_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Offensive performance measure per (team, off_line):
      adjusted_xg_per60

    Adjustments:
      - TOI normalization (per 60)
      - Opponent defensive pairing difficulty (xG allowed per 60)
      - Opponent goalie difficulty (xG allowed per 60)

    Interpretation:
      Producing xG against tougher defense/goalies gets more credit.
    """
    def_tbl, g_tbl = build_def_pairing_and_goalie_tables(df)

    league_def_avg = float(def_tbl["xg_allowed_per60"].mean())
    league_goalie_avg = float(g_tbl["xg_allowed_per60"].mean())

    # Build offense rows: one row per matchup segment, from offense perspective
    home_off = df[[
        "home_team", "home_off_line", "home_xg", "toi",
        "away_team", "away_def_pairing", "away_goalie"
    ]].copy()
    home_off.columns = ["team", "off_line", "xg", "toi", "opp_team", "opp_def_pairing", "opp_goalie"]

    away_off = df[[
        "away_team", "away_off_line", "away_xg", "toi",
        "home_team", "home_def_pairing", "home_goalie"
    ]].copy()
    away_off.columns = ["team", "off_line", "xg", "toi", "opp_team", "opp_def_pairing", "opp_goalie"]

    off = pd.concat([home_off, away_off], ignore_index=True)
    off = off[off["toi"] > 0].copy()

    # Join opponent defensive pairing quality
    off = off.merge(
        def_tbl.rename(columns={
            "team": "opp_team",
            "def_pairing": "opp_def_pairing",
            "xg_allowed_per60": "opp_pair_allowed_per60"
        }),
        on=["opp_team", "opp_def_pairing"],
        how="left"
    )

    # Join opponent goalie quality
    off = off.merge(
        g_tbl.rename(columns={
            "goalie": "opp_goalie",
            "xg_allowed_per60": "opp_goalie_allowed_per60"
        }),
        on="opp_goalie",
        how="left"
    )

    # Fallbacks if missing
    off["opp_pair_allowed_per60"] = off["opp_pair_allowed_per60"].fillna(league_def_avg)
    off["opp_goalie_allowed_per60"] = off["opp_goalie_allowed_per60"].fillna(league_goalie_avg)

    # Adjustment factors: tougher opponents -> smaller xg_allowed_per60 -> factor > 1
    off["pair_adj"] = league_def_avg / off["opp_pair_allowed_per60"]
    off["goalie_adj"] = league_goalie_avg / off["opp_goalie_allowed_per60"]

    off["adj_xg"] = off["xg"] * off["pair_adj"] * off["goalie_adj"]
    off["minutes"] = off["toi"] / 60.0

    line_perf = (
        off.groupby(["team", "off_line"], as_index=False)
           .agg(adj_xg=("adj_xg", "sum"),
                minutes=("minutes", "sum"))
    )
    line_perf["adj_xg_per60"] = line_perf["adj_xg"] / (line_perf["minutes"] / 60.0)
    return line_perf


def compute_line_disparity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Disparity ratio for each team:
      ratio = perf(Line1) / perf(Line2)
    """
    line_perf = compute_offensive_line_performance(df)
    line_perf["off_line"] = line_perf["off_line"].astype(str)

    
    labels = line_perf["off_line"].value_counts().index.tolist()
    if len(labels) < 2:
        raise ValueError("Could not find at least two offensive lines in the data.")

    a, b = labels[0], labels[1]
    tmp = line_perf[line_perf["off_line"].isin([a, b])].copy()

    means = tmp.groupby("off_line")["adj_xg_per60"].mean().sort_values(ascending=False)
    line1_label, line2_label = means.index[0], means.index[1]

    l1 = line_perf[line_perf["off_line"] == line1_label][["team", "adj_xg_per60"]].rename(
        columns={"adj_xg_per60": "line1_adj_xg_per60"}
    )
    l2 = line_perf[line_perf["off_line"] == line2_label][["team", "adj_xg_per60"]].rename(
        columns={"adj_xg_per60": "line2_adj_xg_per60"}
    )

    disp = l1.merge(l2, on="team", how="inner")
    disp["disparity_ratio"] = disp["line1_adj_xg_per60"] / disp["line2_adj_xg_per60"]

    # Rank from largest disparity to smallest
    disp = disp.sort_values("disparity_ratio", ascending=False).reset_index(drop=True)
    disp["rank"] = np.arange(1, len(disp) + 1)

    return disp[["rank", "team", "disparity_ratio", "line1_adj_xg_per60", "line2_adj_xg_per60"]]


def main():
    df = pd.read_csv(WHL_CSV)

    # Basic checks
    needed = {
        "home_team", "away_team",
        "home_off_line", "away_off_line",
        "home_def_pairing", "away_def_pairing",
        "home_goalie", "away_goalie",
        "home_xg", "away_xg", "toi"
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    disp_all = compute_line_disparity(df)
    disp_all.to_csv(OUT_ALL, index=False)
    print(f"Saved all-team disparity file: {OUT_ALL}")

    top10 = disp_all.head(10).copy()
    top10.to_csv(OUT_TOP10, index=False)
    print(f"Saved Top 10 disparity file: {OUT_TOP10}")

    print("\nTop 10 teams by offensive line quality disparity:")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()