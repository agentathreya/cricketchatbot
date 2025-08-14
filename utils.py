"""
utils.py
---------

Pure‑Python helper functions that are exposed to the LLM as LangChain tools.
Each function receives the full DataFrame (already loaded in app.py) and returns
a **markdown table** (or a short string) that the LLM can paste straight into the
chat response.

All functions are deliberately *stateless* – they only read from the dataframe,
they never mutate it.
"""

import pandas as pd
from tabulate import tabulate


# ----------------------------------------------------------------------
# 1️⃣  Basic Statistics Functions
# ----------------------------------------------------------------------
def get_most_runs(df: pd.DataFrame) -> str:
    """Get the top run scorers in the tournament."""
    agg = (
        df.groupby(["batter", "batting_team"])
        .agg(
            runs=("runs_batter", "sum"),
            balls=("balls_faced", "sum"),
            fours=("isFour", "sum"),
            sixes=("isSix", "sum")
        )
        .reset_index()
        .sort_values("runs", ascending=False)
        .head(10)
    )
    # Calculate strike rate safely
    agg["strike_rate"] = (agg["runs"] / agg["balls"] * 100).round(2)
    agg["strike_rate"] = agg["strike_rate"].fillna(0)
    return agg.to_markdown(index=False)


def get_most_wickets(df: pd.DataFrame) -> str:
    """Get the top wicket takers in the tournament."""
    wicket_df = df[df["isWicket"] == True]
    agg = (
        wicket_df.groupby(["bowler", "bowling_team"])
        .agg(
            wickets=("isWicket", "sum"),
            runs_conceded=("runs_total", "sum"),
            balls_bowled=("ball_no", "count"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("wickets", ascending=False)
        .head(10)
    )
    agg["economy"] = agg["economy"].round(2)
    return agg.to_markdown(index=False)


def get_most_fours_and_sixes(df: pd.DataFrame) -> str:
    """Get players with most fours and sixes."""
    agg = (
        df.groupby(["batter", "batting_team"])
        .agg(
            fours=("isFour", "sum"),
            sixes=("isSix", "sum")
        )
        .reset_index()
    )
    agg["total_boundaries"] = agg["fours"] + agg["sixes"]
    agg = agg.sort_values("total_boundaries", ascending=False).head(10)
    return agg.to_markdown(index=False)


# ----------------------------------------------------------------------
# 2️⃣  Advanced Batting Analysis
# ----------------------------------------------------------------------
def runs_by_overs_and_style(
    df: pd.DataFrame,
    overs_start: int,
    overs_end: int,
    bowling_style: str,
) -> str:
    """Aggregate runs for each batter in a specific overs window vs a bowling style."""
    mask = (
        (df["over"] >= overs_start)
        & (df["over"] <= overs_end)
        & (df["bowling_style"].str.lower() == bowling_style.lower())
    )
    sub = df[mask]

    if sub.empty:
        return f"No deliveries found for overs {overs_start}-{overs_end} vs {bowling_style}."

    agg = (
        sub.groupby(["batter", "batting_team"])
        .agg(
            balls=("ball_no", "count"),
            runs=("runs_batter", "sum"),
            fours=("isFour", "sum"),
            sixes=("isSix", "sum")
        )
        .reset_index()
        .sort_values("runs", ascending=False)
        .head(10)
    )
    # Calculate strike rate safely
    agg["strike_rate"] = (agg["runs"] / agg["balls"] * 100).round(2)
    agg["strike_rate"] = agg["strike_rate"].fillna(0)
    return agg.to_markdown(index=False)


def best_batters_death_overs(df: pd.DataFrame, bowling_type: str = "all") -> str:
    """Get best batters in death overs (16-20) vs specific bowling type."""
    mask = (df["over"] >= 16) & (df["over"] <= 20)
    
    if bowling_type.lower() != "all":
        mask &= (df["bowling_type"].str.lower() == bowling_type.lower())
    
    sub = df[mask]

    if sub.empty:
        return f"No deliveries found for death overs vs {bowling_type}."

    agg = (
        sub.groupby(["batter", "batting_team"])
        .agg(
            balls=("ball_no", "count"),
            runs=("runs_batter", "sum"),
            fours=("isFour", "sum"),
            sixes=("isSix", "sum")
        )
        .reset_index()
        .sort_values("runs", ascending=False)
        .head(10)
    )
    # Calculate strike rate safely
    agg["strike_rate"] = (agg["runs"] / agg["balls"] * 100).round(2)
    agg["strike_rate"] = agg["strike_rate"].fillna(0)
    return agg.to_markdown(index=False)


def best_batters_vs_batting_hand(df: pd.DataFrame, bat_hand: str, overs_start: int = 1, overs_end: int = 20) -> str:
    """Get best batters vs specific batting hand (LHB/RHB) in given overs."""
    mask = (
        (df["over"] >= overs_start) 
        & (df["over"] <= overs_end)
        & (df["bat_hand"].str.lower() == bat_hand.lower())
    )
    sub = df[mask]

    if sub.empty:
        return f"No deliveries found for overs {overs_start}-{overs_end} vs {bat_hand}."

    agg = (
        sub.groupby(["batter", "batting_team"])
        .agg(
            balls=("ball_no", "count"),
            runs=("runs_batter", "sum"),
            fours=("isFour", "sum"),
            sixes=("isSix", "sum")
        )
        .reset_index()
        .sort_values("runs", ascending=False)
        .head(10)
    )
    # Calculate strike rate safely
    agg["strike_rate"] = (agg["runs"] / agg["balls"] * 100).round(2)
    agg["strike_rate"] = agg["strike_rate"].fillna(0)
    return agg.to_markdown(index=False)


# ----------------------------------------------------------------------
# 3️⃣  Advanced Bowling Analysis
# ----------------------------------------------------------------------
def bowler_conceded_runs(
    df: pd.DataFrame,
    overs_start: int,
    overs_end: int,
) -> str:
    """
    Total runs (including extras) conceded by each bowler in a given overs window.
    """
    mask = (df["over"] >= overs_start) & (df["over"] <= overs_end)
    sub = df[mask]

    if sub.empty:
        return f"No deliveries found between overs {overs_start} and {overs_end}."

    agg = (
        sub.groupby(["bowler", "bowling_team"])
        .agg(
            deliveries=("ball_no", "count"),
            runs_conceded=("runs_total", "sum"),
            wides=("wides", "sum"),
            noballs=("noballs", "sum"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("runs_conceded", ascending=False)
        .head(10)
    )
    agg["economy"] = agg["economy"].round(2)
    return agg.to_markdown(index=False)


def bowler_wickets(
    df: pd.DataFrame,
    overs_start: int,
    overs_end: int,
) -> str:
    """
    Number of wickets taken by each bowler in the supplied overs window.
    """
    mask = (df["over"] >= overs_start) & (df["over"] <= overs_end) & (df["isWicket"] == True)
    sub = df[mask]

    if sub.empty:
        return f"No wickets in overs {overs_start}-{overs_end}."

    agg = (
        sub.groupby(["bowler", "bowling_team"])
        .size()
        .reset_index(name="wickets")
        .sort_values("wickets", ascending=False)
        .head(10)
    )
    return agg.to_markdown(index=False)


def best_bowlers_death_overs(df: pd.DataFrame, bat_hand: str = "all") -> str:
    """Get best bowlers in death overs (16-20) vs specific batting hand."""
    mask = (df["over"] >= 16) & (df["over"] <= 20)
    
    if bat_hand.lower() != "all":
        mask &= (df["bat_hand"].str.lower() == bat_hand.lower())
    
    sub = df[mask]

    if sub.empty:
        return f"No deliveries found for death overs vs {bat_hand}."

    # Wickets taken
    wickets_df = sub[sub["isWicket"] == True]
    wickets_agg = (
        wickets_df.groupby(["bowler", "bowling_team"])
        .size()
        .reset_index(name="wickets")
    )
    
    # Economy rate
    economy_agg = (
        sub.groupby(["bowler", "bowling_team"])
        .agg(
            deliveries=("ball_no", "count"),
            runs_conceded=("runs_total", "sum"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
    )
    
    # Merge wickets and economy
    result = economy_agg.merge(wickets_agg, on=["bowler", "bowling_team"], how="left")
    result["wickets"] = result["wickets"].fillna(0).astype(int)
    result = result.sort_values(["wickets", "economy"], ascending=[False, True]).head(10)
    result["economy"] = result["economy"].round(2)
    
    return result.to_markdown(index=False)


def best_bowlers_vs_batting_hand(df: pd.DataFrame, bat_hand: str, overs_start: int = 1, overs_end: int = 20) -> str:
    """Get best bowlers vs specific batting hand (LHB/RHB) in given overs."""
    mask = (
        (df["over"] >= overs_start) 
        & (df["over"] <= overs_end)
        & (df["bat_hand"].str.lower() == bat_hand.lower())
    )
    sub = df[mask]

    if sub.empty:
        return f"No deliveries found for overs {overs_start}-{overs_end} vs {bat_hand}."

    # Wickets taken
    wickets_df = sub[sub["isWicket"] == True]
    wickets_agg = (
        wickets_df.groupby(["bowler", "bowling_team"])
        .size()
        .reset_index(name="wickets")
    )
    
    # Economy rate
    economy_agg = (
        sub.groupby(["bowler", "bowling_team"])
        .agg(
            deliveries=("ball_no", "count"),
            runs_conceded=("runs_total", "sum"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
    )
    
    # Merge wickets and economy
    result = economy_agg.merge(wickets_agg, on=["bowler", "bowling_team"], how="left")
    result["wickets"] = result["wickets"].fillna(0).astype(int)
    result = result.sort_values(["wickets", "economy"], ascending=[False, True]).head(10)
    result["economy"] = result["economy"].round(2)
    
    return result.to_markdown(index=False)


# ----------------------------------------------------------------------
# 4️⃣  Fielding Analysis
# ----------------------------------------------------------------------
def get_fielding_stats(df: pd.DataFrame) -> str:
    """Get fielding statistics including catches, run-outs, etc."""
    # Catches
    catches_df = df[df["event_type"] == "CATCH"]
    catches_agg = (
        catches_df.groupby(["event_fielder", "bowling_team"])
        .size()
        .reset_index(name="catches")
        .sort_values("catches", ascending=False)
        .head(10)
    )
    
    # Run-outs
    runouts_df = df[df["event_type"] == "RUN_OUT"]
    runouts_agg = (
        runouts_df.groupby(["event_fielder", "bowling_team"])
        .size()
        .reset_index(name="run_outs")
        .sort_values("run_outs", ascending=False)
        .head(10)
    )
    
    # Stumpings
    stumpings_df = df[df["event_type"] == "STUMPED"]
    stumpings_agg = (
        stumpings_df.groupby(["event_fielder", "bowling_team"])
        .size()
        .reset_index(name="stumpings")
        .sort_values("stumpings", ascending=False)
        .head(10)
    )
    
    result = f"""
## Catches Taken
{catches_agg.to_markdown(index=False)}

## Run Outs Effected
{runouts_agg.to_markdown(index=False)}

## Stumpings
{stumpings_agg.to_markdown(index=False)}
"""
    return result


# ----------------------------------------------------------------------
# 5️⃣  Partnership Analysis
# ----------------------------------------------------------------------
def get_best_partnerships(df: pd.DataFrame) -> str:
    """Get the best batting partnerships in the tournament."""
    # Group by match and innings to get partnerships
    partnerships = (
        df.groupby(["match_id", "innings", "batting_partners"])
        .agg(
            runs=("runs_batter", "sum"),
            balls=("ball_no", "count"),
            fours=("isFour", "sum"),
            sixes=("isSix", "sum")
        )
        .reset_index()
        .sort_values("runs", ascending=False)
        .head(10)
    )
    
    partnerships["strike_rate"] = (partnerships["runs"] / partnerships["balls"] * 100).round(2)
    return partnerships.to_markdown(index=False)


# ----------------------------------------------------------------------
# 6️⃣  Team Analysis
# ----------------------------------------------------------------------
def team_total_runs(df: pd.DataFrame) -> str:
    """Get total runs scored by each team."""
    agg = (
        df.groupby("batting_team")
        .agg(
            matches=("match_id", "nunique"),
            total_runs=("runs_batter", "sum"),
            total_wickets=("isWicket", "sum"),
            total_fours=("isFour", "sum"),
            total_sixes=("isSix", "sum")
        )
        .reset_index()
        .sort_values("total_runs", ascending=False)
    )
    # Calculate average score safely
    agg["avg_score"] = (agg["total_runs"] / agg["matches"]).round(2)
    return agg.to_markdown(index=False)


def team_wickets_lost(df: pd.DataFrame) -> str:
    """Get wickets lost by each team."""
    agg = (
        df.groupby("batting_team")
        .agg(
            matches=("match_id", "nunique"),
            wickets_lost=("isWicket", "sum"),
            avg_wickets_per_match=("isWicket", lambda x: x.sum() / df[df["batting_team"].isin(x.index)]["match_id"].nunique())
        )
        .reset_index()
        .sort_values("wickets_lost", ascending=False)
    )
    agg["avg_wickets_per_match"] = agg["avg_wickets_per_match"].round(2)
    return agg.to_markdown(index=False)


def team_run_rate(df: pd.DataFrame) -> str:
    """Get run rate for each team."""
    agg = (
        df.groupby("batting_team")
        .agg(
            matches=("match_id", "nunique"),
            total_runs=("runs_batter", "sum"),
            total_balls=("ball_no", "count"),
            run_rate=("runs_batter", lambda x: (x.sum() / df[df["batting_team"].isin(x.index)]["ball_no"].sum()) * 6)
        )
        .reset_index()
        .sort_values("run_rate", ascending=False)
    )
    agg["run_rate"] = agg["run_rate"].round(2)
    return agg.to_markdown(index=False)


# ----------------------------------------------------------------------
# 7️⃣  Match Analysis
# ----------------------------------------------------------------------
def get_match_results(df: pd.DataFrame) -> str:
    """Get match results and winners."""
    matches = (
        df.groupby(["match_id", "batting_team", "bowling_team", "winner", "result"])
        .agg(
            total_runs=("runs_batter", "sum"),
            total_wickets=("isWicket", "sum")
        )
        .reset_index()
        .sort_values("match_id")
    )
    return matches.to_markdown(index=False)


def get_venue_stats(df: pd.DataFrame) -> str:
    """Get statistics by venue."""
    agg = (
        df.groupby("venue")
        .agg(
            matches=("match_id", "nunique"),
            total_runs=("runs_batter", "sum"),
            total_wickets=("isWicket", "sum"),
            avg_runs_per_match=("runs_batter", lambda x: x.sum() / df[df["venue"].isin(x.index)]["match_id"].nunique())
        )
        .reset_index()
        .sort_values("total_runs", ascending=False)
    )
    agg["avg_runs_per_match"] = agg["avg_runs_per_match"].round(2)
    return agg.to_markdown(index=False)


# ----------------------------------------------------------------------
# 8️⃣  Advanced Analytics
# ----------------------------------------------------------------------
def get_powerplay_stats(df: pd.DataFrame) -> str:
    """Get powerplay (1-6 overs) statistics."""
    powerplay_df = df[df["over"] <= 6]
    
    # Batting stats
    batting_stats = (
        powerplay_df.groupby("batting_team")
        .agg(
            runs=("runs_batter", "sum"),
            wickets=("isWicket", "sum"),
            run_rate=("runs_batter", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("runs", ascending=False)
    )
    batting_stats["run_rate"] = batting_stats["run_rate"].round(2)
    
    # Bowling stats
    bowling_stats = (
        powerplay_df.groupby("bowling_team")
        .agg(
            runs_conceded=("runs_total", "sum"),
            wickets=("isWicket", "sum"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("wickets", ascending=False)
    )
    bowling_stats["economy"] = bowling_stats["economy"].round(2)
    
    result = f"""
## Powerplay Batting Stats
{batting_stats.to_markdown(index=False)}

## Powerplay Bowling Stats
{bowling_stats.to_markdown(index=False)}
"""
    return result


def get_middle_overs_stats(df: pd.DataFrame) -> str:
    """Get middle overs (7-15) statistics."""
    middle_df = df[(df["over"] >= 7) & (df["over"] <= 15)]
    
    # Batting stats
    batting_stats = (
        middle_df.groupby("batting_team")
        .agg(
            runs=("runs_batter", "sum"),
            wickets=("isWicket", "sum"),
            run_rate=("runs_batter", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("runs", ascending=False)
    )
    batting_stats["run_rate"] = batting_stats["run_rate"].round(2)
    
    # Bowling stats
    bowling_stats = (
        middle_df.groupby("bowling_team")
        .agg(
            runs_conceded=("runs_total", "sum"),
            wickets=("isWicket", "sum"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("wickets", ascending=False)
    )
    bowling_stats["economy"] = bowling_stats["economy"].round(2)
    
    result = f"""
## Middle Overs Batting Stats
{batting_stats.to_markdown(index=False)}

## Middle Overs Bowling Stats
{bowling_stats.to_markdown(index=False)}
"""
    return result


def get_death_overs_stats(df: pd.DataFrame) -> str:
    """Get death overs (16-20) statistics."""
    death_df = df[(df["over"] >= 16) & (df["over"] <= 20)]
    
    # Batting stats
    batting_stats = (
        death_df.groupby("batting_team")
        .agg(
            runs=("runs_batter", "sum"),
            wickets=("isWicket", "sum"),
            run_rate=("runs_batter", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("runs", ascending=False)
    )
    batting_stats["run_rate"] = batting_stats["run_rate"].round(2)
    
    # Bowling stats
    bowling_stats = (
        death_df.groupby("bowling_team")
        .agg(
            runs_conceded=("runs_total", "sum"),
            wickets=("isWicket", "sum"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("wickets", ascending=False)
    )
    bowling_stats["economy"] = bowling_stats["economy"].round(2)
    
    result = f"""
## Death Overs Batting Stats
{batting_stats.to_markdown(index=False)}

## Death Overs Bowling Stats
{bowling_stats.to_markdown(index=False)}
"""
    return result


# ----------------------------------------------------------------------
# 9️⃣  Utility Functions (kept for compatibility)
# ----------------------------------------------------------------------
def bowler_economy(df: pd.DataFrame, overs_start: int, overs_end: int) -> str:
    """Economy rate for each bowler in a given overs window."""
    mask = (df["over"] >= overs_start) & (df["over"] <= overs_end)
    sub = df[mask]

    if sub.empty:
        return f"No deliveries found between overs {overs_start} and {overs_end}."

    agg = (
        sub.groupby(["bowler", "bowling_team"])
        .agg(
            deliveries=("ball_no", "count"),
            runs_conceded=("runs_total", "sum"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
        .sort_values("economy", ascending=True)
        .head(10)
    )
    agg["economy"] = agg["economy"].round(2)
    return agg.to_markdown(index=False)


def top_strike_rate(df: pd.DataFrame, min_balls: int = 10) -> str:
    """Top strike rates for batters with minimum balls faced."""
    agg = (
        df.groupby(["batter", "batting_team"])
        .agg(
            runs=("runs_batter", "sum"),
            balls=("balls_faced", "sum"),
            strike_rate=("runs_batter", lambda x: (x.sum() / df[df["batter"].isin(x.index)]["balls_faced"].sum()) * 100)
        )
        .reset_index()
        .query(f"balls >= {min_balls}")
        .sort_values("strike_rate", ascending=False)
        .head(10)
    )
    agg["strike_rate"] = agg["strike_rate"].round(2)
    return agg.to_markdown(index=False)


def win_probability(df: pd.DataFrame, team: str) -> str:
    """Win probability for a specific team."""
    team_matches = df[df["batting_team"] == team]["match_id"].nunique()
    team_wins = df[df["winner"] == team]["match_id"].nunique()
    
    if team_matches == 0:
        return f"No matches found for {team}."
    
    win_rate = (team_wins / team_matches) * 100
    return f"{team} has won {team_wins} out of {team_matches} matches (Win Rate: {win_rate:.1f}%)"


def best_partnerships(df: pd.DataFrame) -> str:
    """Get the best batting partnerships."""
    return get_best_partnerships(df)


# ----------------------------------------------------------------------
# 10️⃣  Team-Specific Analysis
# ----------------------------------------------------------------------
def get_team_best_batters(df: pd.DataFrame, team_name: str) -> str:
    """Get the best batters for a specific team."""
    team_df = df[df["batting_team"] == team_name]
    
    if team_df.empty:
        return f"No data found for {team_name}."
    
    agg = (
        team_df.groupby(["batter"])
        .agg(
            runs=("runs_batter", "sum"),
            balls=("balls_faced", "sum"),
            fours=("isFour", "sum"),
            sixes=("isSix", "sum")
        )
        .reset_index()
        .sort_values("runs", ascending=False)
        .head(10)
    )
    
    # Calculate strike rate safely
    agg["strike_rate"] = (agg["runs"] / agg["balls"] * 100).round(2)
    agg["strike_rate"] = agg["strike_rate"].fillna(0)
    
    return f"""
## Best Batters for {team_name}

{agg.to_markdown(index=False)}
"""


def get_team_best_bowlers(df: pd.DataFrame, team_name: str) -> str:
    """Get the best bowlers for a specific team."""
    team_df = df[df["bowling_team"] == team_name]
    
    if team_df.empty:
        return f"No bowling data found for {team_name}."
    
    # Wickets taken
    wickets_df = team_df[team_df["isWicket"] == True]
    wickets_agg = (
        wickets_df.groupby(["bowler"])
        .size()
        .reset_index(name="wickets")
    )
    
    # Economy rate
    economy_agg = (
        team_df.groupby(["bowler"])
        .agg(
            deliveries=("ball_no", "count"),
            runs_conceded=("runs_total", "sum"),
            economy=("runs_total", lambda x: (x.sum() / len(x)) * 6)
        )
        .reset_index()
    )
    
    # Merge wickets and economy
    result = economy_agg.merge(wickets_agg, on="bowler", how="left")
    result["wickets"] = result["wickets"].fillna(0).astype(int)
    result = result.sort_values(["wickets", "economy"], ascending=[False, True]).head(10)
    result["economy"] = result["economy"].round(2)
    
    return f"""
## Best Bowlers for {team_name}

{result.to_markdown(index=False)}
"""


def get_team_overall_stats(df: pd.DataFrame, team_name: str) -> str:
    """Get overall statistics for a specific team."""
    batting_df = df[df["batting_team"] == team_name]
    bowling_df = df[df["bowling_team"] == team_name]
    
    if batting_df.empty and bowling_df.empty:
        return f"No data found for {team_name}."
    
    # Batting stats
    batting_stats = {
        "Total Runs": batting_df["runs_batter"].sum(),
        "Total Wickets Lost": batting_df["isWicket"].sum(),
        "Total Fours": batting_df["isFour"].sum(),
        "Total Sixes": batting_df["isSix"].sum(),
        "Matches Played": batting_df["match_id"].nunique()
    }
    
    # Bowling stats
    bowling_stats = {
        "Total Wickets Taken": bowling_df["isWicket"].sum(),
        "Total Runs Conceded": bowling_df["runs_total"].sum(),
        "Total Extras": bowling_df["wides"].sum() + bowling_df["noballs"].sum()
    }
    
    # Calculate averages
    if batting_stats["Matches Played"] > 0:
        batting_stats["Average Score"] = (batting_stats["Total Runs"] / batting_stats["Matches Played"]).round(2)
    
    return f"""
## Overall Statistics for {team_name}

### Batting Statistics
- **Total Runs**: {batting_stats['Total Runs']}
- **Total Wickets Lost**: {batting_stats['Total Wickets Lost']}
- **Total Fours**: {batting_stats['Total Fours']}
- **Total Sixes**: {batting_stats['Total Sixes']}
- **Matches Played**: {batting_stats['Matches Played']}
- **Average Score**: {batting_stats.get('Average Score', 'N/A')}

### Bowling Statistics
- **Total Wickets Taken**: {bowling_stats['Total Wickets Taken']}
- **Total Runs Conceded**: {bowling_stats['Total Runs Conceded']}
- **Total Extras**: {bowling_stats['Total Extras']}
"""


# ----------------------------------------------------------------------
# 11️⃣  Utility Functions (kept for compatibility)
# ----------------------------------------------------------------------
