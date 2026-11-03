"""
Extreme event cross-reference for QC'd flood events.

Checks time windows against three independent signs of an extreme
coastal event:

1. ``significant_weather``: a significant weather event listed by a
   configured NWS office (``NWS_significant_event_lists/``).
2. ``advisory_issued``: an NWS coastal or tropical watch, warning, or
   advisory from a configured office (see :mod:`flood_qc.nws_advisories`).
3. ``obs_elevated_wtr_level``: a wave-driven storm at a configured NDBC
   buoy (see :mod:`flood_qc.wave_storms`).

A window with at least two of the three gets
``meets_criteria_for_extreme_event = True``.

Two subcommands use the same configuration file:

``tag``
    Adds the criteria columns to every event in a QC results CSV and
    saves ``qc_results_with_extreme_events.csv`` next to it.
``master``
    Merges all overlapping source windows into one timeline for the
    meteorological year and saves it with the criteria columns.

Functions
---------
load_config
    Read and check the YAML configuration file.
check_overlap
    Check whether a time window overlaps any window in a table.
merge_time_windows
    Merge overlapping time windows into distinct windows.
load_significant_events
    Load the NWS significant events for the configured offices.
load_nws_advisories
    Download the NWS advisories for the configured offices.
load_wave_storms
    Find the wave-driven storms at the configured buoys.
load_all_sources
    Load all three criteria sources.
add_criteria_columns
    Add the three criteria columns and the 2-of-3 result to a table.
tag_qc_events
    Run the ``tag`` subcommand.
build_master_list
    Run the ``master`` subcommand.
main
    Command-line entry point.

Examples
--------
Run from the repository root::

    python -m flood_qc.extreme_events tag
    python -m flood_qc.extreme_events master --config my_config.yaml
"""

import argparse
import os
import time
from typing import List, Tuple

import pandas as pd
import yaml

from .nws_advisories import fetch_nws_advisories, plot_advisory_calendar
from .utils import log
from .wave_storms import extract_storm_events

# Repository root (the folder that contains the flood_qc package)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration file used when --config is not given
DEFAULT_CONFIG = os.path.join(REPO_DIR, "extreme_event_example_config.yaml")

# Folder with the running record of Significant_Events_<year>.csv files
SIG_EVENTS_DIR = os.path.join(REPO_DIR, "NWS_significant_event_lists")

# Hours added to Eastern Standard Time to get UTC
EST_TO_UTC = pd.Timedelta(hours=5)

# Names of the three criteria columns, in output order
CRITERIA_COLUMNS = [
    "significant_weather",
    "advisory_issued",
    "obs_elevated_wtr_level",
]

# Number of criteria a window must meet to count as an extreme event
MIN_CRITERIA_MET = 2


def load_config(config_path: str) -> dict:
    """
    Read and check the YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path of the YAML configuration file.

    Returns
    -------
    dict
        The configuration, with ``met_year`` converted to int and
        ``nws_offices`` and ``station_pairs`` defaulting to empty lists.

    Raises
    ------
    FileNotFoundError
        If `config_path` does not exist.
    """
    # Fail clearly if the file is missing
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )

    # Parse the YAML file into a dictionary
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # The meteorological year is required and must be a whole number
    config["met_year"] = int(config["met_year"])
    # Offices and station pairs are optional
    config["nws_offices"] = config.get("nws_offices") or []
    config["station_pairs"] = config.get("station_pairs") or []
    return config


def check_overlap(
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    reference_df: pd.DataFrame,
    ref_start_col: str,
    ref_end_col: str,
) -> bool:
    """
    Check whether a time window overlaps any window in a table.

    Parameters
    ----------
    event_start : pandas.Timestamp
        Start of the window to check (timezone-aware UTC).
    event_end : pandas.Timestamp
        End of the window to check (timezone-aware UTC).
    reference_df : pandas.DataFrame
        Table of reference windows.
    ref_start_col : str
        Name of the start-time column in `reference_df`.
    ref_end_col : str
        Name of the end-time column in `reference_df`.

    Returns
    -------
    bool
        True if the window overlaps at least one reference window
        (touching endpoints count as overlapping).
    """
    # Nothing to overlap with
    if reference_df.empty:
        return False

    # Two windows overlap when each starts before the other ends
    overlap_mask = (event_start <= reference_df[ref_end_col]) & (
        event_end >= reference_df[ref_start_col]
    )
    return bool(overlap_mask.any())


def merge_time_windows(
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Merge overlapping time windows into distinct windows.

    Parameters
    ----------
    intervals : list of tuple of (pandas.Timestamp, pandas.Timestamp)
        (start, end) windows. The list is sorted in place.

    Returns
    -------
    list of tuple of (pandas.Timestamp, pandas.Timestamp)
        Windows in start-time order, with no two windows overlapping.
    """
    # Nothing to merge
    if not intervals:
        return []

    # Put the windows in start-time order
    intervals.sort(key=lambda x: x[0])

    # Start with the earliest window
    merged = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        # The most recent merged window
        last_start, last_end = merged[-1]

        # Starts before the merged window ends: extend the merged window
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        # Starts after it ends: begin a new merged window
        else:
            merged.append((current_start, current_end))

    return merged


def load_significant_events(
    met_year: int, nws_offices: List[str]
) -> pd.DataFrame:
    """
    Load the NWS significant events for the configured offices.

    Reads ``Significant_Events_<met_year>.csv`` and
    ``Significant_Events_<met_year + 1>.csv`` from
    ``NWS_significant_event_lists/`` and keeps the rows whose
    ``Office`` is in `nws_offices`.

    Parameters
    ----------
    met_year : int
        First year of the meteorological year.
    nws_offices : list of str
        NWS WFO IDs to keep (e.g. ``["ILM", "MHX"]``).

    Returns
    -------
    pandas.DataFrame
        The kept events with added ``StartDate_UTC`` and
        ``EndDate_UTC`` columns. Has only those two (empty) columns if
        no events were found.
    """
    log("Processing Significant Weather Events...")
    sig_events_list = []

    # Read the files for both calendar years in the meteorological year
    for year in [met_year, met_year + 1]:
        filename = f"Significant_Events_{year}.csv"
        filepath = os.path.join(SIG_EVENTS_DIR, filename)

        if os.path.exists(filepath):
            sig_events_list.append(pd.read_csv(filepath))
        # Warn about a missing year; its events count as not found
        else:
            log(f"  -> Warning: {filename} not found in {SIG_EVENTS_DIR}.")

    # Empty result if neither file exists
    if not sig_events_list:
        return pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])

    # Combine both years into one table
    master_sig_df = pd.concat(sig_events_list, ignore_index=True)

    # Keep only the configured offices
    if "Office" in master_sig_df.columns:
        initial_count = len(master_sig_df)
        sig_df = master_sig_df[
            master_sig_df["Office"].isin(nws_offices)
        ].copy()
        log(
            f"  -> Filtered out {initial_count - len(sig_df)} events "
            "matching other NWS offices."
        )
    # Without an Office column every event is kept
    else:
        log(
            "  -> Warning: 'Office' column missing from Significant Events "
            "CSV. No filtering applied."
        )
        sig_df = master_sig_df

    # Empty result if no events are left after filtering
    if sig_df.empty:
        return pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])

    # The CSV times are Eastern Standard Time: add 5 hours and mark UTC
    sig_df["StartDate_UTC"] = (
        pd.to_datetime(sig_df["StartDate"]) + EST_TO_UTC
    ).dt.tz_localize("UTC")
    sig_df["EndDate_UTC"] = (
        pd.to_datetime(sig_df["EndDate"]) + EST_TO_UTC
    ).dt.tz_localize("UTC")

    return sig_df


def load_nws_advisories(
    met_year: int, nws_offices: List[str], output_dir: str
) -> pd.DataFrame:
    """
    Download the NWS advisories for the configured offices.

    For each office, also saves the advisories to
    ``<office>_events_<met_year>.csv`` and draws a calendar PNG in
    `output_dir`.

    Parameters
    ----------
    met_year : int
        First year of the meteorological year (May 1 - April 30).
    nws_offices : list of str
        NWS WFO IDs to download.
    output_dir : str
        Folder for the per-office CSVs and calendars.

    Returns
    -------
    pandas.DataFrame
        All offices' advisories with UTC ``issue`` and ``expire``
        columns. Has only those two (empty) columns if none were found.
    """
    log("Processing NWS Advisories...")
    advisory_list = []

    for station in nws_offices:
        # Download this office's advisories
        log(f"  -> Fetching advisories for {station}...")
        adv_df = fetch_nws_advisories(station=station, start_year=met_year)

        # Skip offices with no advisories
        if adv_df.empty:
            continue

        advisory_list.append(adv_df)

        # Save this office's advisories to CSV
        adv_csv_path = os.path.join(
            output_dir, f"{station.lower()}_events_{met_year}.csv"
        )
        adv_df.to_csv(adv_csv_path, index=False)

        # Draw this office's calendar; a drawing error is not fatal
        try:
            plot_advisory_calendar(adv_df, station, met_year, output_dir)
        except Exception as e:
            log(
                f"  -> Warning: Could not generate calendar for {station}. "
                f"Error: {e}"
            )

    # Empty result if no office had advisories
    if not advisory_list:
        return pd.DataFrame(columns=["issue", "expire"])

    # Combine all offices and make sure the times are UTC
    advisories_df = pd.concat(advisory_list, ignore_index=True)
    advisories_df["issue"] = pd.to_datetime(advisories_df["issue"], utc=True)
    advisories_df["expire"] = pd.to_datetime(advisories_df["expire"], utc=True)
    return advisories_df


def load_wave_storms(
    met_year: int, station_pairs: List[List[str]], output_dir: str
) -> pd.DataFrame:
    """
    Find the wave-driven storms at the configured buoys.

    Processes both full calendar years in the meteorological year
    (`met_year` and `met_year + 1`) for every buoy and gauge pair.

    Parameters
    ----------
    met_year : int
        First year of the meteorological year.
    station_pairs : list of list of str
        [NDBC buoy ID, NOAA tide gauge ID] pairs.
    output_dir : str
        Folder for the per-buoy storm table CSVs.

    Returns
    -------
    pandas.DataFrame
        All storms with added UTC ``StartDate_UTC`` and ``EndDate_UTC``
        columns. Has only those two (empty) columns if none were found.
    """
    log("Processing Elevated Water Levels...")
    storms_list = []

    # Process every year and buoy/gauge pair
    for year in [met_year, met_year + 1]:
        for wave_station, wl_station in station_pairs:
            log(
                f"  -> Extracting storm events for Year: {year} | "
                f"Buoy: {wave_station} | Gauge: {wl_station}..."
            )

            try:
                # Find the storms for this pair and year
                storm_df = extract_storm_events(
                    year=str(year),
                    wave_station_id=str(wave_station),
                    wl_station_id=str(wl_station),
                    output_dir=output_dir,
                )
                if not storm_df.empty:
                    storms_list.append(storm_df)
            # A failed download skips this pair but not the others
            except Exception as e:
                log(
                    f"  -> WARNING: Failed to extract events for pair "
                    f"{wave_station}/{wl_station}. API timeout/error. "
                    f"Details: {e}"
                )

    # Empty result if no storms were found
    if not storms_list:
        return pd.DataFrame(columns=["StartDate_UTC", "EndDate_UTC"])

    # Combine all pairs and years and add UTC time columns
    elevated_wtr_df = pd.concat(storms_list, ignore_index=True)
    elevated_wtr_df["StartDate_UTC"] = pd.to_datetime(
        elevated_wtr_df["StartDate"], utc=True
    )
    elevated_wtr_df["EndDate_UTC"] = pd.to_datetime(
        elevated_wtr_df["EndDate"], utc=True
    )
    return elevated_wtr_df


def load_all_sources(
    config: dict, output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three criteria sources.

    Parameters
    ----------
    config : dict
        Configuration from :func:`load_config`.
    output_dir : str
        Folder for the intermediate advisory and storm files.

    Returns
    -------
    sig_df : pandas.DataFrame
        Significant events, from :func:`load_significant_events`.
    advisories_df : pandas.DataFrame
        NWS advisories, from :func:`load_nws_advisories`.
    elevated_wtr_df : pandas.DataFrame
        Wave-driven storms, from :func:`load_wave_storms`.
    """
    # Load each source in turn
    sig_df = load_significant_events(config["met_year"], config["nws_offices"])
    advisories_df = load_nws_advisories(
        config["met_year"], config["nws_offices"], output_dir
    )
    elevated_wtr_df = load_wave_storms(
        config["met_year"], config["station_pairs"], output_dir
    )
    return sig_df, advisories_df, elevated_wtr_df


def add_criteria_columns(
    events_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    advisories_df: pd.DataFrame,
    elevated_wtr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the three criteria columns and the 2-of-3 result to a table.

    Parameters
    ----------
    events_df : pandas.DataFrame
        Windows to check, with UTC ``start_time_UTC`` and
        ``end_time_UTC`` columns. Changed in place.
    sig_df : pandas.DataFrame
        Significant events, with ``StartDate_UTC`` and ``EndDate_UTC``.
    advisories_df : pandas.DataFrame
        NWS advisories, with ``issue`` and ``expire``.
    elevated_wtr_df : pandas.DataFrame
        Wave-driven storms, with ``StartDate_UTC`` and ``EndDate_UTC``.

    Returns
    -------
    pandas.DataFrame
        `events_df` with ``significant_weather``, ``advisory_issued``,
        ``obs_elevated_wtr_level``, and
        ``meets_criteria_for_extreme_event`` columns added.
    """
    log("Evaluating event overlaps and generating final QC indicators...")

    # One list of True/False values per criterion
    sig_weather_col = []
    advisory_col = []
    elev_wtr_col = []

    for _, row in events_df.iterrows():
        ev_start = row["start_time_UTC"]
        ev_end = row["end_time_UTC"]

        # Does the window overlap a significant weather event?
        sig_weather_col.append(
            check_overlap(
                ev_start, ev_end, sig_df, "StartDate_UTC", "EndDate_UTC"
            )
        )
        # Does it overlap an NWS advisory?
        advisory_col.append(
            check_overlap(ev_start, ev_end, advisories_df, "issue", "expire")
        )
        # Does it overlap a wave-driven storm?
        elev_wtr_col.append(
            check_overlap(
                ev_start,
                ev_end,
                elevated_wtr_df,
                "StartDate_UTC",
                "EndDate_UTC",
            )
        )

    # Attach the criteria columns
    events_df["significant_weather"] = sig_weather_col
    events_df["advisory_issued"] = advisory_col
    events_df["obs_elevated_wtr_level"] = elev_wtr_col

    # An extreme event meets at least two of the three criteria
    events_df["meets_criteria_for_extreme_event"] = (
        events_df[CRITERIA_COLUMNS].sum(axis=1) >= MIN_CRITERIA_MET
    )
    return events_df


def tag_qc_events(config: dict) -> str:
    """
    Run the ``tag`` subcommand.

    Adds the extreme event criteria to every event in the QC results
    CSV named by ``input_csv`` and saves
    ``qc_results_with_extreme_events.csv`` in the same folder.

    Parameters
    ----------
    config : dict
        Configuration from :func:`load_config`. Must contain
        ``input_csv``.

    Returns
    -------
    str
        Path of the saved CSV.
    """
    script_start_time = time.time()

    # Read the settings this subcommand needs
    input_csv = config["input_csv"]
    met_year = config["met_year"]

    log(f"--- Starting QC Orchestration for Met Year {met_year} ---")
    log(f"Active Weather Offices: {', '.join(config['nws_offices'])}")
    log(
        f"Configured Marine Gauges: {len(config['station_pairs'])} pairs "
        "loaded."
    )

    # Outputs go in the same folder as the input CSV
    target_dir = os.path.dirname(input_csv) or "."
    log(
        "Targeting directory for inputs and outputs: "
        f"{os.path.abspath(target_dir)}"
    )

    # Load the QC results and parse the event times as UTC
    log(f"Loading base QC data from {input_csv}...")
    qc_df = pd.read_csv(input_csv)
    qc_df["start_time_UTC"] = pd.to_datetime(qc_df["start_time_UTC"], utc=True)
    qc_df["end_time_UTC"] = pd.to_datetime(qc_df["end_time_UTC"], utc=True)

    # Load the criteria sources and tag each event
    sig_df, advisories_df, elevated_wtr_df = load_all_sources(
        config, target_dir
    )
    qc_df = add_criteria_columns(qc_df, sig_df, advisories_df, elevated_wtr_df)

    # Save the tagged QC results
    output_path = os.path.join(
        target_dir, "qc_results_with_extreme_events.csv"
    )
    qc_df.to_csv(output_path, index=False)

    elapsed_time = round(time.time() - script_start_time, 2)
    log("--- Orchestration Complete! ---")
    log(f"Total Execution Time: {elapsed_time} seconds")
    log(f"Final dataset saved to: {output_path}")
    return output_path


def build_master_list(config: dict) -> str:
    """
    Run the ``master`` subcommand.

    Merges every significant event, advisory, and storm window into one
    timeline of distinct windows, adds the extreme event criteria, and
    saves the result to ``write_path`` (or
    ``master_extreme_events_list.csv`` in the current folder if
    ``write_path`` is not set).

    Parameters
    ----------
    config : dict
        Configuration from :func:`load_config`.

    Returns
    -------
    str
        Path of the saved CSV.
    """
    script_start_time = time.time()
    met_year = config["met_year"]

    # Use write_path if given, otherwise the default file name
    output_path = config.get("write_path", "master_extreme_events_list.csv")
    # Intermediate files go in the same folder as the output
    target_dir = os.path.dirname(output_path) or "."
    os.makedirs(target_dir, exist_ok=True)

    log(
        "--- Starting Master Event QC Orchestration for Met Year "
        f"{met_year} ---"
    )
    log(f"Active Weather Offices: {', '.join(config['nws_offices'])}")
    log(
        f"Configured Marine Gauges: {len(config['station_pairs'])} pairs "
        "loaded."
    )
    log(
        "Targeting directory for inputs and outputs: "
        f"{os.path.abspath(target_dir)}"
    )

    # Load the criteria sources
    sig_df, advisories_df, elevated_wtr_df = load_all_sources(
        config, target_dir
    )

    # Collect every (start, end) window from all three sources
    log("Consolidating all recorded events into a master timeline...")
    all_intervals = []
    if not sig_df.empty:
        all_intervals.extend(
            zip(sig_df["StartDate_UTC"], sig_df["EndDate_UTC"])
        )
    if not advisories_df.empty:
        all_intervals.extend(
            zip(advisories_df["issue"], advisories_df["expire"])
        )
    if not elevated_wtr_df.empty:
        all_intervals.extend(
            zip(
                elevated_wtr_df["StartDate_UTC"],
                elevated_wtr_df["EndDate_UTC"],
            )
        )

    # Merge overlapping windows into distinct windows
    master_intervals = merge_time_windows(all_intervals)
    master_df = pd.DataFrame(
        master_intervals, columns=["start_time_UTC", "end_time_UTC"]
    )

    # Check each merged window against the three sources
    master_df = add_criteria_columns(
        master_df, sig_df, advisories_df, elevated_wtr_df
    )

    # Save the master list
    master_df.to_csv(output_path, index=False)

    elapsed_time = round(time.time() - script_start_time, 2)
    log("--- Orchestration Complete! ---")
    log(f"Total Master Events Evaluated: {len(master_df)}")
    log(f"Total Execution Time: {elapsed_time} seconds")
    log(f"Master timeline dataset saved to: {output_path}")
    return output_path


def main():
    """
    Run the ``tag`` or ``master`` subcommand from the command line.

    Returns
    -------
    None
    """
    # --config is shared by both subcommands
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to the runtime config YAML (default: "
        "extreme_event_example_config.yaml in the repository).",
    )

    # Define the main parser and its two subcommands
    parser = argparse.ArgumentParser(
        description="Cross-reference flood events against NWS significant "
        "events, NWS advisories, and wave-driven storms."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "tag",
        parents=[config_parser],
        help="Add extreme event criteria to each event in input_csv.",
    )
    subparsers.add_parser(
        "master",
        parents=[config_parser],
        help="Build a master list of extreme events and save it to "
        "write_path.",
    )
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Run the chosen subcommand
    if args.command == "tag":
        tag_qc_events(config)
    else:
        build_master_list(config)


# Run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
