"""
fastf1_dataset.py

Builds a driver/team/circuit dataset from real F1 session data using the
FastF1 library, then scores an upcoming race the same way the GRID/READ
dashboard does -- except every number comes from actual timing data
instead of hand-estimated weights.

Requires internet access to F1's live timing service (FastF1 pulls from
livetiming.formula1.com / the Ergast archive). Run this on your own
machine, not inside a network-sandboxed environment.

Install:
    pip install fastf1 pandas scikit-learn

Usage:
    python fastf1_dataset.py --season 2026 --through-round 12 --out dataset.csv
    python fastf1_dataset.py --season 2026 --through-round 12 --predict Hungary
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fastf1
except ImportError:
    sys.exit("Missing dependency. Run: pip install fastf1 pandas scikit-learn")


CACHE_DIR = Path(__file__).parent / "f1_cache"


def setup_cache():
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


# ---------------------------------------------------------------------------
# 1. Pull per-round race + qualifying results into one tidy table
# ---------------------------------------------------------------------------

def get_completed_rounds(season: int, through_round: int | None = None) -> pd.DataFrame:
    """Event schedule for the season, optionally capped at a round number."""
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    if through_round is not None:
        schedule = schedule[schedule["RoundNumber"] <= through_round]
    return schedule


def build_round_results(season: int, round_number: int, event_name: str) -> pd.DataFrame | None:
    """Race + qualifying results for a single round, merged into one row per driver."""
    try:
        race = fastf1.get_session(season, round_number, "R")
        race.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"  [skip] Round {round_number} ({event_name}) race data unavailable: {e}")
        return None

    if race.results is None or race.results.empty:
        print(f"  [skip] Round {round_number} ({event_name}) has no result data yet.")
        return None

    race_res = race.results[[
        "DriverNumber", "Abbreviation", "FullName", "TeamName",
        "GridPosition", "Position", "Points", "Status",
    ]].copy()
    race_res = race_res.rename(columns={
        "Position": "FinishPosition",
        "GridPosition": "GridPosition",
    })

    # Qualifying is optional -- some events (sprint weekends) shuffle grids differently.
    try:
        quali = fastf1.get_session(season, round_number, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        quali_res = quali.results[["Abbreviation", "Position"]].rename(
            columns={"Position": "QualiPosition"}
        )
        race_res = race_res.merge(quali_res, on="Abbreviation", how="left")
    except Exception:
        race_res["QualiPosition"] = np.nan

    race_res.insert(0, "Season", season)
    race_res.insert(1, "RoundNumber", round_number)
    race_res.insert(2, "EventName", event_name)
    return race_res


def build_season_dataset(season: int, through_round: int | None = None, with_circuit_profile: bool = True) -> pd.DataFrame:
    setup_cache()
    schedule = get_completed_rounds(season, through_round)

    rows = []
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        if rnd == 0:
            continue  # testing entries sometimes show up as round 0
        print(f"Fetching round {rnd}: {event['EventName']}...")
        result = build_round_results(season, rnd, event["EventName"])
        if result is None:
            continue

        if with_circuit_profile:
            # Telemetry pull is slow -- only do it once per round, then
            # broadcast the same circuit numbers to every driver's row.
            profile = build_circuit_profile(season, rnd)
            if profile:
                for key, val in profile.items():
                    result[key] = val

        rows.append(result)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_multi_season_dataset(
    start_season: int,
    end_season: int,
    through_round: int | None = None,
    with_circuit_profile: bool = True,
) -> pd.DataFrame:
    """
    Loop build_season_dataset across a range of years (inclusive) and stack
    the results. through_round only applies to the LAST season in the range
    (e.g. the current, in-progress one) -- every earlier season is pulled in
    full, since those seasons are already complete.
    """
    all_seasons = []
    for year in range(start_season, end_season + 1):
        is_final_season = (year == end_season)
        cap = through_round if is_final_season else None

        print(f"\n=== Season {year} {'(through round ' + str(cap) + ')' if cap else '(full season)'} ===")
        try:
            season_df = build_season_dataset(year, cap, with_circuit_profile=with_circuit_profile)
        except Exception as e:
            print(f"  [skip season] {year} failed entirely: {e}")
            continue

        if season_df.empty:
            print(f"  [skip season] {year} returned no rows.")
            continue

        all_seasons.append(season_df)

    if not all_seasons:
        return pd.DataFrame()

    combined = pd.concat(all_seasons, ignore_index=True)
    print(f"\nPulled {len(all_seasons)} season(s), {len(combined)} total rows.")
    return combined


# ---------------------------------------------------------------------------
# 2. Circuit speed profile, derived from actual lap speed-trap data
# ---------------------------------------------------------------------------

def build_circuit_profile(season: int, round_number: int) -> dict | None:
    """
    Objective circuit characteristics pulled from lap telemetry:
      - straight_line_index: avg speed-trap reading (proxy for how much the
        track rewards power / low downforce)
      - technical_index: inverse of avg lap speed (slower avg speed -> more
        low-speed technical corners)
      - gear_changes: avg gear changes per lap (proxy for corner density)
    """
    try:
        race = fastf1.get_session(season, round_number, "R")
        race.load(telemetry=True, weather=False, messages=False)
        laps = race.laps.pick_quicklaps()
        if laps.empty:
            return None

        straight_line_index = laps["SpeedST"].mean()  # speed trap
        avg_lap_speed = (race.get_circuit_info().length / laps["LapTime"].dt.total_seconds().mean()) if hasattr(race, "get_circuit_info") else np.nan

        gear_changes = []
        for _, lap in laps.head(5).iterrows():  # sample a handful of laps for speed
            try:
                car = lap.get_car_data()
                gear_changes.append((car["nGear"].diff().abs() > 0).sum())
            except Exception:
                continue

        return {
            "straight_line_index": float(straight_line_index) if pd.notna(straight_line_index) else None,
            "avg_lap_speed_kmh": float(avg_lap_speed) if pd.notna(avg_lap_speed) else None,
            "avg_gear_changes_per_lap": float(np.mean(gear_changes)) if gear_changes else None,
        }
    except Exception as e:
        print(f"  [circuit profile skipped] {e}")
        return None


# ---------------------------------------------------------------------------
# 3. Rolling driver/team form
# ---------------------------------------------------------------------------

def add_rolling_form(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    df = df.sort_values(["Abbreviation", "Season", "RoundNumber"]).copy()
    df["FinishPosition"] = pd.to_numeric(df["FinishPosition"], errors="coerce")
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")

    df["RollingAvgFinish"] = (
        df.groupby("Abbreviation")["FinishPosition"]
          .transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    )
    df["RollingPointsPerRace"] = (
        df.groupby("Abbreviation")["Points"]
          .transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    )
    return df


# ---------------------------------------------------------------------------
# 4. Simple scoring model for an upcoming round
#    (same shape as the dashboard's model, but every input is measured,
#    not estimated)
# ---------------------------------------------------------------------------

def score_upcoming_race(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.sort_values(["Season", "RoundNumber"]).groupby("Abbreviation").tail(1).copy()

    latest["form_score"] = latest["RollingPointsPerRace"].rank(pct=True) * 100
    latest["team_pace_score"] = (
        latest.groupby("TeamName")["RollingPointsPerRace"].transform("mean").rank(pct=True) * 100
    )

    latest["predicted_score"] = (
        latest["form_score"].fillna(0) * 0.55
        + latest["team_pace_score"].fillna(0) * 0.45
    )

    return latest[[
        "Abbreviation", "FullName", "TeamName",
        "RollingAvgFinish", "RollingPointsPerRace", "predicted_score",
    ]].sort_values("predicted_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a multi-season F1 dataset from FastF1 and score the next race.")
    parser.add_argument("--start-season", type=int, default=2022, help="First season to pull (inclusive)")
    parser.add_argument("--end-season", type=int, default=2026, help="Last season to pull (inclusive) -- this is the season --through-round applies to")
    parser.add_argument("--through-round", type=int, default=None, help="Cap the END season at this round number (earlier seasons are pulled in full)")
    parser.add_argument("--out", type=str, default="fastf1_dataset.csv")
    parser.add_argument("--predict", type=str, default=None, help="Print a predicted order for the next race (informational label only)")
    parser.add_argument("--no-circuit-profile", action="store_true", help="Skip telemetry pull for circuit speed profile (much faster)")
    args = parser.parse_args()

    if args.start_season > args.end_season:
        sys.exit("--start-season must be <= --end-season")

    print(f"Building dataset for seasons {args.start_season}-{args.end_season}...")
    df = build_multi_season_dataset(
        args.start_season,
        args.end_season,
        args.through_round,
        with_circuit_profile=not args.no_circuit_profile,
    )

    if df.empty:
        print("No data retrieved. Check your internet connection / FastF1's data availability for these seasons.")
        return

    df = add_rolling_form(df)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")

    if args.predict:
        print(f"\nPredicted order heading into {args.predict} (form + team pace only, no circuit-fit yet):")
        ranked = score_upcoming_race(df)
        print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()