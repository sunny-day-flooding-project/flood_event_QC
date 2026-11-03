"""
Storm Event Extractor for NDBC Buoys and NOAA Tide Gauges.

Identifies storm events at NDBC wave buoys and calculates the corresponding
water level and storm surge using NOAA CO-OPS tide gauges.

Adapted from StormsFromWaveandWaterLevel.m by Liliana Velasquez Montoya (2017),
previously modified by Elizabeth Farquhar (2026).

"""

import argparse
import pandas as pd
from ndbc_api import NdbcApi
from noaa_coops import Station


def extract_storm_events(
    year: str,
    wave_station_id: str,
    wl_station_id: str,
    hs_threshold_meters: float = 1.5,
    min_duration_hours: float = 8.0,
    output_dir: str = ".",
) -> pd.DataFrame:
    """
    Extracts storm events and corresponding water levels for a given year and station pair.
    """
    print(
        f"Processing Year: {year} | Wave Station: {wave_station_id} | WL Station: {wl_station_id}"
    )

    # --- Data Retrieval: Wave Buoy ---
    print("  -> Fetching NDBC wave data...")
    api = NdbcApi()

    # Fetch standard meteorological data ('stdmet') for the specified year
    df_waves = api.get_data(
        station_id=wave_station_id,
        mode="stdmet",
        start_time=f"{year}-01-01",
        end_time=f"{year}-12-31",
    )

    # Check if data exists for this station/year combination
    if df_waves.empty:
        print(
            "  -> Warning: No wave data found for the specified year and station."
        )
        return pd.DataFrame()

    # Move the index into standard columns
    df_waves = df_waves.reset_index()

    # Standardize column names to uppercase to avoid case-sensitivity issues
    df_waves.columns = [str(col).upper() for col in df_waves.columns]

    time_col = None

    # 1. Search for a column that is explicitly a pandas datetime type
    for col in df_waves.columns:
        if pd.api.types.is_datetime64_any_dtype(df_waves[col]):
            time_col = col
            break

    # 2. Search by typical name if not found by type
    if time_col is None:
        for col in ["TIME", "DATE", "DATETIME"]:
            if col in df_waves.columns:
                time_col = col
                # Parse just in case it was left as a string
                df_waves[time_col] = pd.to_datetime(df_waves[time_col])
                break

    # 3. Fallback: Manually build it from individual raw columns
    if time_col is None:
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

        dt_cols = [
            c
            for c in ["year", "month", "day", "hour", "minute"]
            if c in df_waves.columns
        ]
        if len(dt_cols) >= 3:
            df_waves["assembled_time"] = pd.to_datetime(df_waves[dt_cols])
            time_col = "assembled_time"

    if time_col:
        # Set our found/built column as the true DatetimeIndex and ensure it is sorted chronologically
        df_waves.index = df_waves[time_col]
        df_waves = df_waves.sort_index()
    else:
        print("  -> Error: Could not determine time index from NDBC data.")
        print(f"     Available columns: {list(df_waves.columns)}")
        return pd.DataFrame()

    # Extract Significant Wave Height (WVHT)
    if "WVHT" not in df_waves.columns:
        print("  -> Warning: No 'WVHT' (wave height) column found in data.")
        return pd.DataFrame()

    wave_heights = df_waves[["WVHT"]].dropna()

    # --- Storm Identification Logic ---
    print("  -> Identifying storm events...")

    # Create a boolean mask where wave height exceeds our threshold
    is_high = wave_heights["WVHT"] > hs_threshold_meters

    # Group contiguous blocks of high waves by identifying where the condition flips
    storm_groups = (~is_high).cumsum()[is_high]

    storm_records = []

    # Iterate through each contiguous block of high waves
    for group_id, group_df in wave_heights[is_high].groupby(storm_groups):

        # Calculate start, end, and duration
        start_time = group_df.index[0]
        end_time = group_df.index[-1]

        # Calculate duration in hours
        duration_hours = (end_time - start_time).total_seconds() / 3600.0

        # Keep the event only if it meets the minimum duration threshold
        if duration_hours > min_duration_hours:
            storm_records.append(
                {
                    "StartDate": start_time,
                    "EndDate": end_time,
                    "Duration_Hr": duration_hours,
                    "MaxHs_m": group_df["WVHT"].max(),
                }
            )

    print(
        f"  -> Found {len(storm_records)} storm events. Fetching water level data..."
    )

    # --- Data Retrieval: Water Levels ---

    print(
        f"  -> Found {len(storm_records)} storm events. Fetching annual water level data..."
    )
    
    if not storm_records:
        return pd.DataFrame()

    # --- Data Retrieval: Water Levels (Annual Batch) ---
    wl_station = Station(id=wl_station_id)
    
    # We can fetch exactly one year of hourly data per API request.
    start_str = f"{year}0101"
    end_str = f"{year}1231"

    try:
        # Fetch observed and predicted hourly heights for the entire year
        df_obs = wl_station.get_data(
            begin_date=start_str,
            end_date=end_str,
            product="hourly_height",
            datum="NAVD",
            units="metric",
            time_zone="gmt",
        )

        df_pred = wl_station.get_data(
            begin_date=start_str,
            end_date=end_str,
            product="predictions",
            datum="NAVD",
            units="metric",
            time_zone="gmt",
        )

        # Combine DataFrames on their datetime index and calculate surge for the whole year
        df_combined = pd.DataFrame(
            {
                "observed": df_obs["v"].astype(float),
                "predicted": df_pred["v"].astype(float),
            }
        )
        df_combined["surge"] = df_combined["observed"] - df_combined["predicted"]

    except Exception as e:
        print(f"  -> Error: Could not fetch annual WL data. Reason: {e}")
        return pd.DataFrame()

    # --- Slicing Data & Unit Conversions ---
    final_output = []
    METERS_TO_FEET = 1 / 0.3048

    for event in storm_records:
        # Instead of pinging the API, just slice the local dataframe to the storm's exact window
        df_storm = df_combined.loc[event["StartDate"] : event["EndDate"]]

        if not df_storm.empty:
            max_wl_m = df_storm["observed"].max()
            max_surge_m = df_storm["surge"].max()
        else:
            max_wl_m = float("nan")
            max_surge_m = float("nan")

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

    # --- File Output ---
    output_table = pd.DataFrame(final_output)

    if not output_table.empty:
        # Construct filename and path
        filename = f"StormTable_{wave_station_id}_{year}_{hs_threshold_meters}_{min_duration_hours}.csv"
        output_path = f"{output_dir}/{filename}"

        # Save to disk
        output_table.to_csv(output_path, index=False)
        print(f"  -> Processing complete. Saved to: {output_path}")
    else:
        print("  -> No storms met the criteria for this year/station combo.")

    return output_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract storm events from NDBC and NOAA APIs."
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
        help="Significant wave height threshold in meters.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Minimum storm duration in hours.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to save the output CSV.",
    )

    args = parser.parse_args()

    result_df = extract_storm_events(
        year=args.year,
        wave_station_id=args.wave_station,
        wl_station_id=args.wl_station,
        hs_threshold_meters=args.threshold,
        min_duration_hours=args.duration,
        output_dir=args.outdir,
    )

    if not result_df.empty:
        print("\nPreview of extracted storms:")
        print(result_df.head())
