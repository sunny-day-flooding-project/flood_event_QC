"""
Master Event QC Orchestrator

This script ingests a configuration YAML file, pulls extreme weather events,
NWS advisories, and elevated water levels, and merges them into a unified
master timeline. It then cross-references this master list against the three
criteria to determine which combined events meet the threshold of an extreme event.
"""

import os
import argparse
import time
from datetime import datetime
from typing import List, Tuple
import yaml
import pandas as pd

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
    # Fetch the current local time and format it as Hours:Minutes:Seconds
    current_time = datetime.now().strftime("%H:%M:%S")
    # Print the formatted time alongside the provided message
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
    # Return early if the reference dataframe has no events to compare against
    if reference_df.empty:
        return False

    # Create a boolean mask to find overlaps:
    # An overlap occurs if the event starts before the reference ends AND ends after the reference starts
    overlap_mask = (event_start <= reference_df[ref_end_col]) & (
        event_end >= reference_df[ref_start_col]
    )
    
    # Return True if any overlaps were found in the entire mask
    return bool(overlap_mask.any())


def merge_time_windows(
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Merge overlapping time intervals into continuous, distinct time windows.

    Parameters
    ----------
    intervals : List[Tuple[pd.Timestamp, pd.Timestamp]]
        A list of start and end timestamp tuples representing individual events.

    Returns
    -------
    List[Tuple[pd.Timestamp, pd.Timestamp]]
        A merged list of distinct start and end timestamp tuples where no
        windows overlap.
    """
    # If the input list is empty, there is nothing to merge, so return an empty list
    if not intervals:
        return []

    # Sort the intervals chronologically based on their start times
    intervals.sort(key=lambda x: x[0])
    
    # Initialize the merged list with the very first interval
    merged = [intervals[0]]

    # Loop through the remaining intervals starting from the second item
    for current_start, current_end in intervals[1:]:
        # Retrieve the start and end of the most recently merged window
        last_start, last_end = merged[-1]

        # Check if the current interval's start falls within the most recent window
        if current_start <= last_end:
            # If so, expand the end time of the most recent window to encompass both
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # If they do not overlap, add the current interval as a brand new window
            merged.append((current_start, current_end))

    # Return the condensed list of mutually exclusive time intervals
    return merged


def run_orchestrator(config_path: str) -> None:
    """
    Read the runtime YAML configuration, gather external data based on constraints,
    merge event timelines into a master schedule, evaluate overlaps, and save outputs.

    Parameters
    ----------
    config_path : str
        The file path to the runtime configuration YAML file.

    Returns
    -------
    None
    """
    # Record the initial script execution time for performance logging
    script_start_time = time.time()

    # Validate that the provided configuration file path actually exists
    if not os.path.exists(config_path):
        # Raise a fatal error if the configuration file is missing
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )

    # Open the YAML file in read-only mode
    with open(config_path, "r") as f:
        # Parse the YAML configuration into a Python dictionary safely
        config = yaml.safe_load(f)

    # Extract required runtime variables from the parsed configuration
    met_year = int(config["met_year"])
    nws_offices = config.get("nws_offices", [])
    station_pairs = config.get("station_pairs", [])

    # Setup file path routing
    target_dir = "."
    output_path = "master_extreme_events_list.csv"
    
    # Check if a custom write_path was provided in the config
    if "write_path" in config:
        # Set the final output path directly to what was specified
        output_path = config["write_path"]
        # Derive the directory from the provided file path to store intermediate files
        target_dir = os.path.dirname(output_path) or "."
        
    # Ensure the target directory exists so file writes don't throw errors
    os.makedirs(target_dir, exist_ok=True)

    # Output startup logs detailing current runtime settings
    log(f"--- Starting Master Event QC Orchestration for Met Year {met_year} ---")
    log(f"Active Weather Offices: {', '.join(nws_offices)}")
    log(f"Configured Marine Gauges: {len(station_pairs)} pairs loaded.")
    log(f"Targeting directory for inputs and outputs: {os.path.abspath(target_dir)}")

    # Define the scope of years based on the met_year (current and next)
    years_to_process = [met_year, met_year + 1]

    # --- 1. Process Significant Local Weather Events ---
    log("Fetching local significant weather events...")
    sig_events_list = []
    
    # Iterate through the targeted years to check for significant events
    for year in years_to_process:
        # Construct the expected filename for local events
        filename = f"Significant_Events_{year}.csv"
        # Bind the filename to our designated working directory
        filepath = os.path.join(target_dir, filename)

        # Check if the local CSV data exists
        if os.path.exists(filepath):
            # Read the CSV into a pandas DataFrame
            temp_df = pd.read_csv(filepath)
            # Add the DataFrame to our list for later concatenation
            sig_events_list.append(temp_df)
        else:
            # Log a warning if the file is missing so the user is aware
            log(f"  -> Warning: {filename} not found in {target_dir}.")

    # Check if we successfully loaded any significant events
    if sig_events_list:
        # Combine all yearly DataFrames into a single master DataFrame
        master_sig_df = pd.concat(sig_events_list, ignore_index=True)

        # Ensure the 'Office' column exists before trying to filter by it
        if "Office" in master_sig_df.columns:
            # Capture the row count prior to filtering
            initial_count = len(master_sig_df)
            # Filter the DataFrame to only keep events from the configured NWS offices
            sig_df = master_sig_df[master_sig_df["Office"].isin(nws_offices)].copy()
            # Log how many irrelevant events were dropped
            log(f"  -> Filtered out {initial_count - len(sig_df)} events from other offices.")
        else:
            # Warn if we can't filter the data
            log("  -> Warning: 'Office' column missing. No filtering applied.")
            # Bypass filtering if the column doesn't exist
            sig_df = master_sig_df

        # If after filtering we still have records
        if not sig_df.empty:
            # Convert StartDate to datetime, shift 5 hours to convert EST to UTC, and localize the timezone
            sig_df["StartDate_UTC"] = (
                pd.to_datetime(sig_df["StartDate"]) + pd.Timedelta(hours=5)
            ).dt.tz_localize("UTC")
            # Apply the exact same timezone transformation to the EndDate
            sig_df["EndDate_UTC"] = (
                pd.to_datetime(sig_df["EndDate"]) + pd.Timedelta(hours=5)
            ).dt.tz_localize("UTC")
        else:
            # Initialize an empty structure if all records were filtered out
            sig_df = pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])
    else:
        # Initialize an empty structure if no CSV files were loaded at all
        sig_df = pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])

    # --- 2. Process NWS Advisories ---
    log("Downloading and processing NWS Advisories...")
    advisory_list = []
    
    # Iterate over each requested weather office
    for station in nws_offices:
        log(f"  -> Fetching advisories for {station}...")
        # Hit the external NWS fetcher method to get advisory events
        adv_df = fetch_extreme_events(station=station, start_year=met_year)

        # Check if the API returned any actionable events
        if not adv_df.empty:
            # Append the successful pull to our ongoing list
            advisory_list.append(adv_df)

            # Define where to cache these events locally
            adv_csv_path = os.path.join(
                target_dir, f"{station.lower()}_events_{met_year}.csv"
            )
            # Save the advisory data to disk
            adv_df.to_csv(adv_csv_path, index=False)

            # Try to generate a visual calendar output using Pillow
            try:
                generate_pillow_calendar(adv_df, station, met_year, target_dir)
            except Exception as e:
                # Catch and log any visual generation errors so it doesn't crash the script
                log(f"  -> Warning: Could not generate calendar for {station}. Error: {e}")

    # Check if we successfully gathered any NWS advisories
    if advisory_list:
        # Concatenate all office advisories into one master advisory DataFrame
        advisories_df = pd.concat(advisory_list, ignore_index=True)
        # Force the 'issue' start times into strict UTC timestamps
        advisories_df["issue"] = pd.to_datetime(advisories_df["issue"], utc=True)
        # Force the 'expire' end times into strict UTC timestamps
        advisories_df["expire"] = pd.to_datetime(advisories_df["expire"], utc=True)
    else:
        # Ensure a defined empty structure if the API pulls failed or were empty
        advisories_df = pd.DataFrame(columns=["issue", "expire"])

    # --- 3. Process Elevated Water Levels ---
    log("Processing elevated water level data...")
    storms_list = []
    
    # Loop over the necessary years
    for year in years_to_process:
        # Loop through each configured buoy/gauge pair in the YAML
        for wave_station, wl_station in station_pairs:
            log(f"  -> Extracting storm events for Year: {year} | Buoy: {wave_station} | Gauge: {wl_station}...")
            # Try to execute the extraction logic
            try:
                # Call the external method to derive storms from wave/water level data
                storm_df = extract_storm_events(
                    year=str(year),
                    wave_station_id=str(wave_station),
                    wl_station_id=str(wl_station),
                    output_dir=target_dir,
                )
                # Append if the method returned results
                if not storm_df.empty:
                    storms_list.append(storm_df)
            except Exception as e:
                # Catch external processing failures without breaking the loop
                log(f"  -> WARNING: Failed to extract events for pair {wave_station}/{wl_station}. Details: {e}")

    # Check if water level storms were found
    if storms_list:
        # Merge all station/year records into a single elevated water DataFrame
        elevated_wtr_df = pd.concat(storms_list, ignore_index=True)
        # Enforce UTC timezone awareness on the start date
        elevated_wtr_df["StartDate_UTC"] = pd.to_datetime(elevated_wtr_df["StartDate"], utc=True)
        # Enforce UTC timezone awareness on the end date
        elevated_wtr_df["EndDate_UTC"] = pd.to_datetime(elevated_wtr_df["EndDate"], utc=True)
    else:
        # Create an empty DataFrame template if no storm records were derived
        elevated_wtr_df = pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])

    # --- 4. Master Timeline Consolidation ---
    log("Consolidating all recorded events into a master timeline...")
    all_intervals = []
    
    # If we have significant weather events, zip their start and end dates and append them to the interval list
    if not sig_df.empty:
        all_intervals.extend(zip(sig_df["StartDate_UTC"], sig_df["EndDate_UTC"]))
    
    # If we have NWS advisories, zip their issue and expire dates and append them
    if not advisories_df.empty:
        all_intervals.extend(zip(advisories_df["issue"], advisories_df["expire"]))
        
    # If we have elevated water events, zip their start and end dates and append them
    if not elevated_wtr_df.empty:
        all_intervals.extend(zip(elevated_wtr_df["StartDate_UTC"], elevated_wtr_df["EndDate_UTC"]))

    # Pass the aggregate list into our merging function to collapse overlapping windows
    master_intervals = merge_time_windows(all_intervals)
    
    # Construct our final Master DataFrame from the continuous merged intervals
    master_df = pd.DataFrame(master_intervals, columns=["start_time_UTC", "end_time_UTC"])

    # --- 5. Evaluate Overlapping Criteria ---
    log("Evaluating overlapping criteria for master timeline...")
    # Initialize separate lists to hold the boolean cross-checks
    sig_weather_col = []
    advisory_col = []
    elev_wtr_col = []

    # Iterate through every unique merged event row in the new master DataFrame
    for _, row in master_df.iterrows():
        # Unpack the bounds of the current event window
        ev_start = row["start_time_UTC"]
        ev_end = row["end_time_UTC"]

        # Check if the master event window overlaps with ANY significant weather event
        sig_weather_col.append(
            check_overlap(ev_start, ev_end, sig_df, "StartDate_UTC", "EndDate_UTC")
        )
        # Check if the master event window overlaps with ANY NWS advisory
        advisory_col.append(
            check_overlap(ev_start, ev_end, advisories_df, "issue", "expire")
        )
        # Check if the master event window overlaps with ANY elevated water level event
        elev_wtr_col.append(
            check_overlap(ev_start, ev_end, elevated_wtr_df, "StartDate_UTC", "EndDate_UTC")
        )

    # Attach the boolean lists back into the master dataframe as explicit columns
    master_df["significant_weather"] = sig_weather_col
    master_df["advisory_issued"] = advisory_col
    master_df["obs_elevated_wtr_level"] = elev_wtr_col

    # Identify "extreme events" by checking if at least 2 of the 3 boolean columns are True for that row
    master_df["meets_criteria_for_extreme_event"] = (
        master_df[["significant_weather", "advisory_issued", "obs_elevated_wtr_level"]].sum(axis=1) >= 2
    )

    # Export the finalized, labeled master dataset to the specified write_path
    master_df.to_csv(output_path, index=False)

    # Calculate total script execution time by diffing current time with our start variable
    elapsed_time = round(time.time() - script_start_time, 2)
    
    # Log the final completion telemetry
    log("--- Orchestration Complete! ---")
    log(f"Total Master Events Evaluated: {len(master_df)}")
    log(f"Total Execution Time: {elapsed_time} seconds")
    log(f"Master timeline dataset saved to: {output_path}")


if __name__ == "__main__":
    # Setup command line argument parsing for the script
    parser = argparse.ArgumentParser(
        description="Extract and combine all storm events into a unified master timeline using YAML configuration."
    )
    # Add a mandatory/optional argument for routing the configuration path
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the runtime config.yaml file (default: config.yaml).",
    )

    # Parse arguments provided by the user in the CLI
    args = parser.parse_args()
    # Trigger the primary processing orchestration
    run_orchestrator(config_path=args.config)