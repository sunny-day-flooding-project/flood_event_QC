"""
Wave-driven storm detection from NDBC buoys and NOAA tide gauges.

Finds storms at an NDBC wave buoy (significant wave height above a
threshold for longer than a minimum duration), then looks up the peak
water level and storm surge for each storm at a paired NOAA CO-OPS tide
gauge.

Adapted from ``reference/StormsFromWaveandWaterLevel.m`` by Liliana
Velasquez Montoya (2017), later modified by Elizabeth Farquhar (2026).

Functions
---------
extract_storm_events
    Find a year's storms for one buoy and tide gauge pair.
main
    Command-line entry point.

Examples
--------
Run from the repository root::

    python -m flood_qc.wave_storms 2025 41110 8658163 --outdir ./storms
"""

import argparse
import os

import pandas as pd
from ndbc_api import NdbcApi
from noaa_coops import Station

from .utils import log

# Multiply meters by this to get feet
METERS_TO_FEET = 1 / 0.3048


def extract_storm_events(
    year: str,
    wave_station_id: str,
    wl_station_id: str,
    hs_threshold_meters: float = 1.5,
    min_duration_hours: float = 8.0,
    output_dir: str = ".",
) -> pd.DataFrame:
    """
    Find a year's storms for one buoy and tide gauge pair.

    A storm is a continuous run of buoy readings with significant wave
    height (``WVHT``) above `hs_threshold_meters` that lasts longer
    than `min_duration_hours`. For each storm, the maximum observed
    water level and the maximum surge (observed minus predicted) at the
    tide gauge are recorded. Results are also saved to
    ``StormTable_<buoy>_<year>_<threshold>_<duration>.csv`` in
    `output_dir`.

    Parameters
    ----------
    year : str
        The four-digit calendar year to process (e.g. ``"2025"``).
    wave_station_id : str
        NDBC wave buoy station ID (e.g. ``"41110"``).
    wl_station_id : str
        NOAA CO-OPS tide gauge station ID (e.g. ``"8658163"``).
    hs_threshold_meters : float, optional
        Significant wave height (m) that waves must exceed. Default is
        1.5.
    min_duration_hours : float, optional
        Minimum storm length in hours (the storm must be longer than
        this). Default is 8.0.
    output_dir : str, optional
        Folder to save the storm table CSV in. Default is the current
        folder.

    Returns
    -------
    pandas.DataFrame
        One row per storm with ``StartDate``, ``EndDate`` (UTC),
        ``MaxHs_ft``, ``Duration_Hr``, ``MaxWL_ftNAVD``, and
        ``Surge_ft`` columns. Empty if no storms were found or data
        could not be retrieved.
    """
    log(
        f"Processing Year: {year} | Wave Station: {wave_station_id} | "
        f"WL Station: {wl_station_id}"
    )

    # ----- Wave buoy data -----
    log("  -> Fetching NDBC wave data...")
    api = NdbcApi()

    # Download the buoy's standard meteorological data for the year
    df_waves = api.get_data(
        station_id=wave_station_id,
        mode="stdmet",
        start_time=f"{year}-01-01",
        end_time=f"{year}-12-31",
    )

    # Stop if the buoy has no data for the year
    if df_waves.empty:
        log(
            "  -> Warning: No wave data found for the specified year and "
            "station."
        )
        return pd.DataFrame()

    # Move the index (which may hold the timestamps) into a column
    df_waves = df_waves.reset_index()

    # Make the column names upper case to avoid case mismatches
    df_waves.columns = [str(col).upper() for col in df_waves.columns]

    # Find the timestamp column; ndbc-api versions differ in how they
    # return it, so try several ways
    time_col = None

    # 1. A column that is already a datetime type
    for col in df_waves.columns:
        if pd.api.types.is_datetime64_any_dtype(df_waves[col]):
            time_col = col
            break

    # 2. A column with a typical timestamp name
    if time_col is None:
        for col in ["TIME", "DATE", "DATETIME"]:
            if col in df_waves.columns:
                time_col = col
                # Parse it in case it is stored as text
                df_waves[time_col] = pd.to_datetime(df_waves[time_col])
                break

    # 3. Build it from separate year/month/day/hour/minute columns
    if time_col is None:
        # Map the NDBC column names to the names pandas expects
        rename_map = {}
        for col in df_waves.columns:
            if col in ["#YY", "YY", "YYYY", "YEAR"]:
                rename_map[col] = "year"
            elif col in ["MM", "MONTH"]:
                rename_map[col] = "month"
            elif col in ["DD", "DAY"]:
                rename_map[col] = "day"
            elif col in ["HH", "HOUR"]:
                rename_map[col] = "hour"
            elif col in ["MIN", "MM_1", "MINUTE"]:
                rename_map[col] = "minute"

        df_waves.rename(columns=rename_map, inplace=True)

        # Use whichever parts are present (year, month, day at least)
        dt_cols = [
            c
            for c in ["year", "month", "day", "hour", "minute"]
            if c in df_waves.columns
        ]
        if len(dt_cols) >= 3:
            df_waves["assembled_time"] = pd.to_datetime(df_waves[dt_cols])
            time_col = "assembled_time"

    if time_col:
        # Index the readings by time, in time order
        df_waves.index = df_waves[time_col]
        df_waves = df_waves.sort_index()
    # Stop if no timestamp could be found
    else:
        log(
            "  -> Error: Could not determine time index from NDBC data.",
            "error",
        )
        log(f"     Available columns: {list(df_waves.columns)}", "error")
        return pd.DataFrame()

    # Stop if the data has no significant wave height column
    if "WVHT" not in df_waves.columns:
        log("  -> Warning: No 'WVHT' (wave height) column found in data.")
        return pd.DataFrame()

    # Keep the wave heights, dropping missing readings
    wave_heights = df_waves[["WVHT"]].dropna()

    # ----- Storm detection -----
    log("  -> Identifying storm events...")

    # True for readings above the wave height threshold
    is_high = wave_heights["WVHT"] > hs_threshold_meters

    # Give each run of consecutive high readings its own group number:
    # the count of low readings so far only changes between runs
    storm_groups = (~is_high).cumsum()[is_high]

    storm_records = []

    # Check each run of high readings
    for _, group_df in wave_heights[is_high].groupby(storm_groups):
        # The run starts and ends at its first and last readings
        start_time = group_df.index[0]
        end_time = group_df.index[-1]

        # Length of the run in hours
        duration_hours = (end_time - start_time).total_seconds() / 3600.0

        # Keep runs longer than the minimum duration as storms
        if duration_hours > min_duration_hours:
            storm_records.append(
                {
                    "StartDate": start_time,
                    "EndDate": end_time,
                    "Duration_Hr": duration_hours,
                    "MaxHs_m": group_df["WVHT"].max(),
                }
            )

    log(
        f"  -> Found {len(storm_records)} storm events. Fetching annual "
        "water level data..."
    )

    # Stop if there are no storms to look up
    if not storm_records:
        return pd.DataFrame()

    # ----- Tide gauge data (one request for the whole year) -----
    wl_station = Station(id=wl_station_id)

    # First and last day of the year in the format CO-OPS expects
    start_str = f"{year}0101"
    end_str = f"{year}1231"

    try:
        # Observed hourly water levels (m, NAVD88, UTC)
        df_obs = wl_station.get_data(
            begin_date=start_str,
            end_date=end_str,
            product="hourly_height",
            datum="NAVD",
            units="metric",
            time_zone="gmt",
        )

        # Predicted (astronomical tide) water levels for the same period
        df_pred = wl_station.get_data(
            begin_date=start_str,
            end_date=end_str,
            product="predictions",
            datum="NAVD",
            units="metric",
            time_zone="gmt",
        )

        # Line up observed and predicted levels by time
        df_combined = pd.DataFrame(
            {
                "observed": df_obs["v"].astype(float),
                "predicted": df_pred["v"].astype(float),
            }
        )
        # Surge is the observed level minus the predicted tide
        df_combined["surge"] = (
            df_combined["observed"] - df_combined["predicted"]
        )

    # Stop if the tide gauge data could not be downloaded
    except Exception as e:
        log(
            f"  -> Error: Could not fetch annual WL data. Reason: {e}", "error"
        )
        return pd.DataFrame()

    # ----- Storm summaries -----
    final_output = []

    for event in storm_records:
        # Select the tide gauge readings during the storm
        df_storm = df_combined.loc[event["StartDate"] : event["EndDate"]]

        # Peak water level and peak surge (need not occur together)
        if not df_storm.empty:
            max_wl_m = df_storm["observed"].max()
            max_surge_m = df_storm["surge"].max()
        # No gauge readings during the storm
        else:
            max_wl_m = float("nan")
            max_surge_m = float("nan")

        # Record the storm with heights converted to feet
        final_output.append(
            {
                "StartDate": event["StartDate"].strftime("%Y-%m-%d %H:%M:%S"),
                "EndDate": event["EndDate"].strftime("%Y-%m-%d %H:%M:%S"),
                "MaxHs_ft": event["MaxHs_m"] * METERS_TO_FEET,
                "Duration_Hr": round(event["Duration_Hr"], 2),
                "MaxWL_ftNAVD": max_wl_m * METERS_TO_FEET,
                "Surge_ft": max_surge_m * METERS_TO_FEET,
            }
        )

    # ----- Save the storm table -----
    output_table = pd.DataFrame(final_output)

    if not output_table.empty:
        # Name the file by buoy, year, threshold, and minimum duration
        filename = (
            f"StormTable_{wave_station_id}_{year}_{hs_threshold_meters}_"
            f"{min_duration_hours}.csv"
        )
        output_path = os.path.join(output_dir, filename)

        # Save it
        output_table.to_csv(output_path, index=False)
        log(f"  -> Processing complete. Saved to: {output_path}")
    else:
        log("  -> No storms met the criteria for this year/station combo.")

    return output_table


def main():
    """
    Find one year's storms for a buoy and tide gauge pair.

    Reads the year, station IDs, thresholds, and output folder from the
    command line, then prints the first few storms found.

    Returns
    -------
    None
    """
    # Define the command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract wave-driven storm events from NDBC buoy and "
        "NOAA tide gauge data."
    )
    parser.add_argument(
        "year", type=str, help="The 4-digit year to process (e.g., 2024)."
    )
    parser.add_argument(
        "wave_station",
        type=str,
        help="NDBC wave buoy station ID (e.g., 44056).",
    )
    parser.add_argument(
        "wl_station",
        type=str,
        help="NOAA tide gauge station ID (e.g., 8651370).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Significant wave height threshold in meters (default: 1.5).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Minimum storm duration in hours (default: 8).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to save the output CSV (default: current "
        "directory).",
    )
    args = parser.parse_args()

    # Find the storms
    result_df = extract_storm_events(
        year=args.year,
        wave_station_id=args.wave_station,
        wl_station_id=args.wl_station,
        hs_threshold_meters=args.threshold,
        min_duration_hours=args.duration,
        output_dir=args.outdir,
    )

    # Show the first few storms
    if not result_df.empty:
        print("\nPreview of extracted storms:")
        print(result_df.head())


# Run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
