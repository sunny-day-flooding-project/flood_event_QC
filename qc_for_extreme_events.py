"""
Extreme Event QC Orchestrator

This script ingests a configuration YAML file, loads an existing QC results CSV,
and cross-references recorded flood events against filtered NWS significant events,
NWS issued advisories, and observed elevated water levels.
"""

import os
import argparse
import time
from datetime import datetime
import yaml
import pandas as pd

# External script modules
from pull_nws_advisories import fetch_extreme_events, generate_pillow_calendar
from storms_from_waves_wtr_levels import extract_storm_events


def log(message: str) -> None:
    """
    Print console messages with a timestamp.

    Parameters
    ----------
    message : str
        The message string to print to the console.

    Returns
    -------
    None
    """
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}")


def check_overlap(
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    reference_df: pd.DataFrame,
    ref_start_col: str,
    ref_end_col: str,
) -> bool:
    """
    Check if a specific time window overlaps with any window in a reference dataframe.

    Parameters
    ----------
    event_start : pd.Timestamp
        The start time of the specific event (timezone aware UTC).
    event_end : pd.Timestamp
        The end time of the specific event (timezone aware UTC).
    reference_df : pd.DataFrame
        The dataframe containing reference events.
    ref_start_col : str
        The column name in reference_df corresponding to the start time.
    ref_end_col : str
        The column name in reference_df corresponding to the end time.

    Returns
    -------
    bool
        True if there is at least one overlapping event, False otherwise.
    """
    if reference_df.empty:
        return False

    # Two time windows overlap if Window A starts before Window B ends,
    # and Window A ends after Window B starts.
    overlap_mask = (event_start <= reference_df[ref_end_col]) & (
        event_end >= reference_df[ref_start_col]
    )
    return overlap_mask.any()


def run_orchestrator(config_path: str) -> None:
    """
    Read the runtime YAML configuration, gather external data based on constraints,
    evaluate timeline overlaps, and save the verified output files.

    Parameters
    ----------
    config_path : str
        The file path to the runtime configuration YAML file.

    Returns
    -------
    None
    """
    script_start_time = time.time()

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_csv = config["input_csv"]
    met_year = int(config["met_year"])
    nws_offices = config.get("nws_offices", [])
    station_pairs = config.get("station_pairs", [])

    log(f"--- Starting QC Orchestration for Met Year {met_year} ---")
    log(f"Active Weather Offices: {', '.join(nws_offices)}")
    log(f"Configured Marine Gauges: {len(station_pairs)} pairs loaded.")

    target_dir = os.path.dirname(input_csv)
    if not target_dir:
        target_dir = "."

    log(
        f"Targeting directory for inputs and outputs: {os.path.abspath(target_dir)}"
    )

    years_to_process = [met_year, met_year + 1]

    # Load foundational QC dataset and enforce timezone-aware UTC types
    log(f"Loading base QC data from {input_csv}...")
    qc_df = pd.read_csv(input_csv)
    qc_df["start_time_UTC"] = pd.to_datetime(qc_df["start_time_UTC"], utc=True)
    qc_df["end_time_UTC"] = pd.to_datetime(qc_df["end_time_UTC"], utc=True)

    # ---------------------------------------------------------
    # 1. Process Significant Weather Events (Filtered by Office)
    # ---------------------------------------------------------
    log("Processing Significant Weather Events...")
    sig_events_list = []

    for year in years_to_process:
        filename = f"Significant_Events_{year}.csv"
        filepath = os.path.join(target_dir, filename)

        if os.path.exists(filepath):
            temp_df = pd.read_csv(filepath)
            sig_events_list.append(temp_df)
        else:
            log(f"  -> Warning: {filename} not found in {target_dir}.")

    if sig_events_list:
        master_sig_df = pd.concat(sig_events_list, ignore_index=True)

        # Filter records to only include the configured NWS offices
        if "Office" in master_sig_df.columns:
            initial_count = len(master_sig_df)
            sig_df = master_sig_df[
                master_sig_df["Office"].isin(nws_offices)
            ].copy()
            log(
                f"  -> Filtered out {initial_count - len(sig_df)} events matching other NWS offices."
            )
        else:
            log(
                "  -> Warning: 'Office' column missing from Significant Events CSV. No filtering applied."
            )
            sig_df = master_sig_df

        if not sig_df.empty:
            # Add 5 hours to shift EST to UTC, then localize
            sig_df["StartDate_UTC"] = (
                pd.to_datetime(sig_df["StartDate"]) + pd.Timedelta(hours=5)
            ).dt.tz_localize("UTC")
            sig_df["EndDate_UTC"] = (
                pd.to_datetime(sig_df["EndDate"]) + pd.Timedelta(hours=5)
            ).dt.tz_localize("UTC")
        else:
            sig_df = pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])
    else:
        sig_df = pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])

    # ---------------------------------------------------------
    # 2. Process NWS Advisories (Configured Offices Loop)
    # ---------------------------------------------------------
    log("Processing NWS Advisories...")
    advisory_list = []

    for station in nws_offices:
        log(f"  -> Fetching advisories for {station}...")
        adv_df = fetch_extreme_events(station=station, start_year=met_year)

        if not adv_df.empty:
            advisory_list.append(adv_df)

            adv_csv_path = os.path.join(
                target_dir, f"{station.lower()}_events_{met_year}.csv"
            )
            adv_df.to_csv(adv_csv_path, index=False)

            try:
                generate_pillow_calendar(adv_df, station, met_year, target_dir)
            except Exception as e:
                log(
                    f"  -> Warning: Could not generate calendar for {station}. Error: {e}"
                )

    if advisory_list:
        advisories_df = pd.concat(advisory_list, ignore_index=True)
        advisories_df["issue"] = pd.to_datetime(
            advisories_df["issue"], utc=True
        )
        advisories_df["expire"] = pd.to_datetime(
            advisories_df["expire"], utc=True
        )
    else:
        advisories_df = pd.DataFrame(columns=["issue", "expire"])

    # ---------------------------------------------------------
    # 3. Process Elevated Water Levels (Configured Pairs Loop)
    # ---------------------------------------------------------
    log("Processing Elevated Water Levels...")
    storms_list = []

    for year in years_to_process:
        for wave_station, wl_station in station_pairs:
            log(
                f"  -> Extracting storm events for Year: {year} | Buoy: {wave_station} | Gauge: {wl_station}..."
            )

            try:
                storm_df = extract_storm_events(
                    year=str(year),
                    wave_station_id=str(wave_station),
                    wl_station_id=str(wl_station),
                    output_dir=target_dir,
                )
                if not storm_df.empty:
                    storms_list.append(storm_df)
            except Exception as e:
                log(
                    f"  -> WARNING: Failed to extract events for pair {wave_station}/{wl_station}. API timeout/error. Details: {e}"
                )

    if storms_list:
        elevated_wtr_df = pd.concat(storms_list, ignore_index=True)
        elevated_wtr_df["StartDate_UTC"] = pd.to_datetime(
            elevated_wtr_df["StartDate"], utc=True
        )
        elevated_wtr_df["EndDate_UTC"] = pd.to_datetime(
            elevated_wtr_df["EndDate"], utc=True
        )
    else:
        elevated_wtr_df = pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])

    # ---------------------------------------------------------
    # 4. Compare Overlaps and Append Columns
    # ---------------------------------------------------------
    log("Evaluating event overlaps and generating final QC indicators...")

    sig_weather_col = []
    advisory_col = []
    elev_wtr_col = []

    for idx, row in qc_df.iterrows():
        ev_start = row["start_time_UTC"]
        ev_end = row["end_time_UTC"]

        sig_weather_col.append(
            check_overlap(
                ev_start, ev_end, sig_df, "StartDate_UTC", "EndDate_UTC"
            )
        )
        advisory_col.append(
            check_overlap(ev_start, ev_end, advisories_df, "issue", "expire")
        )
        elev_wtr_col.append(
            check_overlap(
                ev_start,
                ev_end,
                elevated_wtr_df,
                "StartDate_UTC",
                "EndDate_UTC",
            )
        )

    # Append boolean columns representing overlaps
    qc_df["significant_weather"] = sig_weather_col
    qc_df["advisory_issued"] = advisory_col
    qc_df["obs_elevated_wtr_level"] = elev_wtr_col

    # Require at least 2 conditions to be met for an extreme event
    qc_df["meets_criteria_for_extreme_event"] = (
        qc_df[
            ["significant_weather", "advisory_issued", "obs_elevated_wtr_level"]
        ].sum(axis=1)
        >= 2
    )

    # ---------------------------------------------------------
    # 5. Save Output
    # ---------------------------------------------------------
    output_path = os.path.join(target_dir, "qc_results_with_extreme_events.csv")
    qc_df.to_csv(output_path, index=False)

    elapsed_time = round(time.time() - script_start_time, 2)
    log("--- Orchestration Complete! ---")
    log(f"Total Execution Time: {elapsed_time} seconds")
    log(f"Final dataset saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orchestrate coastal flood QC data against external storm/weather events via YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the runtime config.yaml file (default: config.yaml).",
    )

    args = parser.parse_args()
    run_orchestrator(config_path=args.config)
