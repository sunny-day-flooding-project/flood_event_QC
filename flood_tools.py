import concurrent.futures
import os
import random
import re
import shutil
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytz
import requests


def _log(message, level="info"):
    """
    Prints a timestamped message to stdout or stderr.

    Parameters
    ----------
    message : str
        The message to be logged.
    level : str, optional
        The log level. "info" (default) prints to stdout,
        "error" prints to stderr.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {level.upper()}: {message}"

    if level.lower() == "error":
        # Print to standard error
        print(formatted_message, file=sys.stderr)
    else:
        # Print to standard output
        print(formatted_message, file=sys.stdout)


class EventTracker:
    """
    A client for fetching data from the Sunny Day Flooding Project API
    with built-in flood event summary and plotting.

    Attributes
    ----------
    BASE_URL : str
        The base API endpoint for fetching water level data.
    authorization : Tuple[str, str]
        The (username, password) tuple for API authentication.
    min_date : datetime.date | datetime.datetime
        The overall start date for all data queries.
    max_date : datetime.date | datetime.datetime
        The overall end date for all data queries.
    max_workers : int
        The number of concurrent threads to use for fetching.
    timeout : int
        The request timeout in seconds.
    """

    BASE_URL = (
        "https://api-sunnydayflood.apps.cloudapps.unc.edu/get_water_level"
    )

    def __init__(
        self,
        authorization: Tuple[str, str],
        min_date: date | datetime,
        max_date: date | datetime,
        max_workers: int = 16,
        timeout: int = 60,
    ):
        """
        Initializes the EventTracker.

        Parameters
        ----------
        authorization : Tuple[str, str]
            A (username, password) tuple for API authentication.
        min_date : datetime.date | datetime.datetime
            The overall start date for all data queries (inclusive).
        max_date : datetime.date | datetime.datetime
            The overall end date for all data queries (inclusive).
        max_workers : int, optional
            The number of concurrent threads to use for fetching, by
            default 16.
        timeout : int, optional
            The request timeout in seconds, by default 60.

        Raises
        ------
        ValueError
            If the authorization tuple is not provided or is incomplete.
        """
        if not authorization or not all(authorization):
            raise ValueError(
                "Authorization (username, password) tuple is required."
            )
        if not min_date or not max_date:
            raise ValueError("min_date and max_date are required.")

        self.authorization = authorization
        self.min_date = min_date
        self.max_date = max_date
        self.max_workers = max_workers
        self.timeout = timeout

        _log(
            f"EventTracker initialized. Max workers: {self.max_workers}, Timeout: {self.timeout}s"
        )
        _log(f"Date Range Set: {self.min_date} to {self.max_date}")

    def get_data(
        self, sensor_ids: str | List[str], chunk_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves data using a chunking method and a robust fallback.

        Parameters
        ----------
        sensor_ids : str | List[str]
            A single location name (e.g., "Down East") or a list of
            specific sensor ID strings.
        chunk_days : int, optional
            The number of days to fetch in each parallel request, by
            default 30.

        Returns
        -------
        Optional[pd.DataFrame]
            A combined, deduplicated, and sorted pandas DataFrame
            containing all successfully retrieved data, or None if no
            data could be fetched.
        """

        # Input validation and setup
        sensor_ids_list = self._resolve_sensor_ids(sensor_ids)
        if sensor_ids_list is None:
            return None  # Error already printed by helper

        # Create chunks
        date_chunks = self._create_date_chunks(chunk_days)
        if not date_chunks:
            _log(
                "No date chunks to process (check min/max dates).",
                level="error",
            )
            return None

        total_tasks = len(sensor_ids_list) * len(date_chunks)
        _log(
            f"Total date range split into {len(date_chunks)} chunks of ~{chunk_days} days each."
        )
        _log(
            f"Starting chunked download for {len(sensor_ids_list)} sensors... ({total_tasks} total requests)"
        )

        all_dataframes = []

        # Create Session, Executor, and Run Tasks
        with requests.Session() as session:
            session.auth = self.authorization

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

                # Submit Primary (Chunked) Tasks
                futures = self._submit_chunk_tasks(
                    executor, session, sensor_ids_list, date_chunks
                )

                # Process Primary Results
                primary_data, failed_tasks = self._process_chunk_results(
                    futures
                )
                all_dataframes.extend(primary_data)

            # Handle Failed Tasks (The Fallback)
            if failed_tasks:
                fallback_data = self._handle_failed_chunks(
                    session, failed_tasks
                )
                all_dataframes.extend(fallback_data)

        _log("Data retrieval complete.")

        # Combine and Return Data
        return self._process_and_combine_data(all_dataframes)

    def pull_data_gen_csvs_and_plots(
        self,
        location: str | List[str],
        chunk_days: int = 30,
        output_dir: str = "data",
        full_csv_name: str = "flood_events.csv",
        abbr_csv_name: str = "abbr_flood_events.csv",
        outage_csv_name: str = "sensor_outages.csv",
        plot_folder: str = "flood_plots",
    ) -> Optional[pd.DataFrame]:
        """
        Runs the full data pipeline: fetch, process, and output.

        Parameters
        ----------
        location : str | List[str]
            A single location name (e.g., "Down East") or a list of
            specific sensor ID strings.
        chunk_days : int, optional
            The number of days to fetch in each parallel request,
            by default 30.
        output_dir : str, optional
            Directory to save outputs of method to.
        full_csv_name : str, optional
            Filename for the detailed, row-by-row flood event CSV.
        abbr_csv_name : str, optional
            Filename for the abbreviated, one-row-per-event CSV.
        outage_csv_name : str, optional
            Filename for the sensor outage log CSV.
        plot_folder : str, optional
            Name of the directory to save flood plots to.

        Returns
        -------
        Optional[pd.DataFrame]
            The downloaded and combined DataFrame, or None if the
            data pull fails.
        """
        os.makedirs(output_dir, exist_ok=True)
        full_csv_name = os.path.join(output_dir, full_csv_name)
        abbr_csv_name = os.path.join(output_dir, abbr_csv_name)
        outage_csv_name = os.path.join(output_dir, outage_csv_name)
        plot_folder = os.path.join(output_dir, plot_folder)

        _log(f"Starting data pull for location: '{location}'")
        download_data = self.get_data(location, chunk_days)

        if download_data is None or download_data.empty:
            _log(
                "Data download failed or returned no data. Stopping processing.",
                level="error",
            )
            return None

        _log(f"Data pull successful. Shape: {download_data.shape}")

        _log("Generating CSVs...")
        try:
            self._gen_flood_tracker(download_data, full_csv_name)
            self._gen_abbr_flood_event_csv(download_data, abbr_csv_name)
            _log("CSV files created successfully.")
        except Exception as e:
            _log(f"An error occurred during CSV generation: {e}", level="error")

        _log("Checking for sensor outages...")
        try:
            self._find_outages(download_data, outage_csv_name)
            self._check_for_outage_during_flood(outage_csv_name, abbr_csv_name)
            _log("Sensor outage logs generated and CSVs appended.")
        except Exception as e:
            _log(f"An error occurred during outage check: {e}", level="error")

        _log("Plotting flood events...")
        try:
            self._plot_and_save_flood_plots(
                download_data, abbr_csv_name, plot_folder
            )
            _log("Plotting completed.")
        except Exception as e:
            _log(f"An error occurred during plotting: {e}", level="error")

        return download_data

    def _sensor_list_generator(self, location_name: str) -> List[str]:
        """
        Provides a sensor identity list for a given location.

        Parameters
        ----------
        location_name : str
            Name of sensor location.

        Returns
        -------
        List[str]
            List of sensor ids as strings.

        Raises
        ------
        ValueError
            If the location_name is not recognized.
        """

        if location_name.lower() == "carolina beach":
            sensor_ids = ["CB_01", "CB_02", "CB_03"]
        elif location_name.lower() == "beaufort":
            sensor_ids = ["BF_01"]
        elif location_name.lower() == "down east":
            sensor_ids = ["DE_01", "DE_02", "DE_03", "DE_04"]
        elif location_name.lower() == "new bern":
            sensor_ids = ["NB_01", "NB_02"]
        elif location_name.lower() == "north river":
            sensor_ids = ["NR_01"]
        else:
            raise ValueError(f"Location name '{location_name}' not recognized.")
        return sensor_ids

    def _resolve_sensor_ids(
        self, sensor_ids_input: str | List[str]
    ) -> Optional[List[str]]:
        """
        Resolves a potential location name string into a list of sensor
        IDs.

        If input is a list, it is returned as-is.
        If input is a string, it's passed to `_sensor_list_generator`.

        Parameters
        ----------
        sensor_ids_input : str | List[str]
            The user-provided sensor input.

        Returns
        -------
        Optional[List[str]]
            A list of sensor ID strings, or None if a string location
            name is not recognized.
        """
        if isinstance(sensor_ids_input, str):
            try:
                _log(f"Resolving location name: '{sensor_ids_input}'")
                return self._sensor_list_generator(sensor_ids_input)
            except ValueError as e:
                _log(f"Error: {e}", level="error")
                _log("Stopping data retrieval.", level="error")
                return None

        # If not a string, assume it's a List[str] and return it
        return sensor_ids_input

    def _create_date_chunks(self, chunk_days: int) -> List[Tuple[date, date]]:
        """
        Splits the instance's date range (self.min_date to self.max_date)
        into n-day chunks.

        Parameters
        ----------
        chunk_days : int
            The maximum number of days for each chunk.

        Returns
        -------
        List[Tuple[datetime.date, datetime.date]]
            A list of (start_date, end_date) tuples.
        """
        date_chunks = []
        current_start_date = self.min_date

        # While current time is before end date
        while current_start_date <= self.max_date:
            current_end_date = current_start_date + timedelta(
                days=chunk_days - 1
            )
            if current_end_date > self.max_date:
                current_end_date = self.max_date
            date_chunks.append((current_start_date, current_end_date))
            current_start_date += timedelta(days=chunk_days)
        return date_chunks

    def _fetch_data(
        self,
        session: requests.Session,
        start_date: date | datetime,
        end_date: date | datetime,
        sensor_id: str,
    ) -> Optional[Dict[str, Any] | List[Any]]:
        """
        Perms a single, raw API request to the water level endpoint.

        Parameters
        ----------
        session : requests.Session
            The authenticated requests session to use for the API call.
        start_date : datetime.date | datetime.datetime
            The start date for the API query (inclusive).
        end_date : datetime.date | datetime.datetime
            The end date for the API query (exclusive).
        sensor_id : str
            The specific sensor ID to query.

        Returns
        -------
        Optional[Dict[str, Any] | List[Any]]
            The raw JSON response from the API or None if the request
            fails.
        """

        query_params = {
            "min_date": start_date.strftime("%Y-%m-%d"),
            "max_date": end_date.strftime("%Y-%m-%d"),
            "sensor_ID": sensor_id,
        }

        try:
            response = session.get(
                self.BASE_URL, params=query_params, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            _log(
                f"HTTP error for sensor {sensor_id} ({start_date} to {end_date}): {http_err}",
                level="error",
            )
        except requests.exceptions.JSONDecodeError:
            _log(
                f"Error: Server returned OK but failed to decode JSON for {sensor_id}.",
                level="error",
            )
        except requests.exceptions.RequestException as req_err:
            _log(
                f"Request error for sensor {sensor_id} ({start_date} to {end_date}): {req_err}",
                level="error",
            )

        return None

    def _fetch_chunk(
        self,
        session: requests.Session,
        sensor_id: str,
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches and processes data for a single sensor over a date range
        chunk.

        Parameters
        ----------
        session : requests.Session
            The authenticated requests session.
        sensor_id : str
            The sensor ID for the chunk.
        start_date : datetime.date | datetime.datetime
            The first day of the chunk (inclusive).
        end_date : datetime.date | datetime.datetime
            The last day of the chunk (inclusive).

        Returns
        -------
        Optional[pd.DataFrame]
            A processed DataFrame for the chunk, or None if processing
            fails.
        """

        exclusive_end_date = end_date + timedelta(days=1)

        json_data = self._fetch_data(
            session=session,
            start_date=start_date,
            end_date=exclusive_end_date,
            sensor_id=sensor_id,
        )

        if not json_data:
            return None

        try:
            df = pd.DataFrame(json_data)
            if df.empty:
                return None

            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["date_surveyed"] = pd.to_datetime(df["date_surveyed"], utc=True)
            return df

        except KeyError as e:
            _log(
                f"Error processing DataFrame for {sensor_id}: Missing expected column {e}.",
                level="error",
            )
        except Exception as e:
            _log(
                f"An unexpected error occurred processing data for {sensor_id}: {e}",
                level="error",
            )

        return None

    def _fetch_day(
        self,
        session: requests.Session,
        sensor_id: str,
        target_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches and processes data for a single sensor for a single day.

        Parameters
        ----------
        session : requests.Session
            The authenticated requests session.
        sensor_id : str
            The sensor ID for the request.
        target_date : datetime.date
            The specific day to fetch data for (inclusive).

        Returns
        -------
        Optional[pd.DataFrame]
            A processed DataFrame for the single day, or None if
            processing fails.
        """

        exclusive_end_date = target_date + timedelta(days=1)

        json_data = self._fetch_data(
            session=session,
            start_date=target_date,
            end_date=exclusive_end_date,
            sensor_id=sensor_id,
        )

        if not json_data:
            return None

        try:
            df = pd.DataFrame(json_data)
            if df.empty:
                return None

            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["date_surveyed"] = pd.to_datetime(df["date_surveyed"], utc=True)
            return df

        except KeyError as e:
            _log(
                f"Error processing DataFrame for {sensor_id} on {target_date}: Missing column {e}.",
                level="error",
            )
        except Exception as e:
            _log(
                f"Unexpected error processing data for {sensor_id} on {target_date}: {e}",
                level="error",
            )

        return None

    def _submit_chunk_tasks(
        self,
        executor: ThreadPoolExecutor,
        session: requests.Session,
        sensor_ids: List[str],
        date_chunks: List[Tuple[date, date]],
    ) -> Dict[Future, Tuple[str, date, date]]:
        """
        Submits all sensor/date_chunk tasks to the executor.

        Parameters
        ----------
        executor : ThreadPoolExecutor
            The executor instance to submit tasks to.
        session : requests.Session
            The authenticated session to pass to the task.
        sensor_ids : List[str]
            The list of sensor IDs to fetch.
        date_chunks : List[Tuple[datetime.date, datetime.date]]
            The list of date range tuples.

        Returns
        -------
        Dict[Future, Tuple[str, datetime.date, datetime.date]]
            A dictionary mapping the submitted Future objects to a
            tuple of (sensor_id, start_chunk, end_chunk) for tracking.
        """
        futures = {}
        for sensor_id in sensor_ids:
            for start_chunk, end_chunk in date_chunks:
                task = executor.submit(
                    self._fetch_chunk,
                    session,
                    sensor_id,
                    start_chunk,
                    end_chunk,
                )
                futures[task] = (sensor_id, start_chunk, end_chunk)
        return futures

    def _process_chunk_results(
        self, futures: Dict[Future, Tuple[str, date, date]]
    ) -> Tuple[List[pd.DataFrame], Set[Tuple[str, date, date]]]:
        """
        Processes completed futures as they finish.

        Sorts results into successfully fetched DataFrames and a set
        of task info tuples for failed tasks.

        Parameters
        ----------
        futures : Dict[Future, Tuple[str, datetime.date, datetime.date]]
            The dictionary of futures returned by `_submit_chunk_tasks`.

        Returns
        -------
        Tuple[List[pd.DataFrame], Set[Tuple[str, ...]]]
            A tuple containing:
            - A list of all successfully fetched DataFrames.
            - A set of (sensor_id, start_date, end_date) tuples for
              all tasks that failed and require fallback.
        """
        all_dataframes = []
        failed_tasks = set()

        for future in as_completed(futures):
            task_info = futures[future]
            task_id = f"{task_info[0]} ({task_info[1]} to {task_info[2]})"
            try:
                df = future.result()
                if df is not None:
                    all_dataframes.append(df)
                else:
                    # Handles non-exception failures
                    # (e.g., HTTP 404, empty JSON)
                    _log(
                        f"Task failed, queueing for 1-day fallback: {task_id}",
                        level="info",
                    )
                    failed_tasks.add(task_info)
            except Exception as e:
                # Handles exceptions raised during the task execution
                _log(
                    f"Task error, queueing for 1-day fallback: {task_id}: {e}",
                    level="error",
                )
                failed_tasks.add(task_info)

        return all_dataframes, failed_tasks

    def _handle_failed_chunks(
        self,
        session: requests.Session,
        failed_tasks: Set[Tuple[str, date, date]],
    ) -> List[pd.DataFrame]:
        """
        Manages the fallback process for failed multi-day chunks.

        Parameters
        ----------
        session : requests.Session
            The authenticated requests session.
        failed_tasks : Set[Tuple[str, datetime.date, datetime.date]]
            A set of (sensor_id, start_date, end_date) tuples.

        Returns
        -------
        List[pd.DataFrame]
            A list of all DataFrames successfully retrieved during
            fallback.
        """
        if not failed_tasks:
            return []

        _log(
            f"Handling {len(failed_tasks)} failed chunks by fetching 1-day increments..."
        )
        all_fallback_data = []

        daily_tasks_to_run = []
        for sensor_id, start_date, end_date in failed_tasks:
            _log(
                f"Queueing 1-day fallback for {sensor_id} from {start_date} to {end_date}"
            )
            total_days = (end_date - start_date).days
            for i in range(total_days + 1):
                current_date = start_date + timedelta(days=i)
                daily_tasks_to_run.append((sensor_id, current_date))

        if not daily_tasks_to_run:
            return []

        _log(f"Total fallback 1-day requests: {len(daily_tasks_to_run)}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_day, session, sensor_id, target_date
                ): f"{sensor_id} on {target_date}"
                for (sensor_id, target_date) in daily_tasks_to_run
            }

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        all_fallback_data.append(df)
                except Exception as e:
                    _log(
                        f"Error in fallback task {task_id}: {e}", level="error"
                    )

        _log("Fallback data retrieval complete.")
        return all_fallback_data

    def _process_and_combine_data(
        self, all_dataframes: List[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        Combines, de-duplicates, and sorts the final list of DataFrames.

        Parameters
        ----------
        all_dataframes : List[pd.DataFrame]
            A list of all DataFrames retrieved (both primary and
            fallback).

        Returns
        -------
        Optional[pd.DataFrame]
            The final, cleaned DataFrame, or None if no data was found.
        """
        if not all_dataframes:
            _log(
                "No data was successfully retrieved for any sensor.",
                level="error",
            )
            return None

        try:
            _log(f"Combining {len(all_dataframes)} DataFrame chunks...")
            combined_data = pd.concat(all_dataframes, ignore_index=True)

            _log("Dropping duplicates and sorting by date...")
            combined_data.drop_duplicates(
                subset=["date", "sensor_ID"], inplace=True
            )
            combined_data.sort_values(by=["date"], inplace=True)
            combined_data.reset_index(drop=True, inplace=True)

            _log(f"Final DataFrame shape: {combined_data.shape}")
            return combined_data
        except Exception as e:
            _log(f"Failed to concatenate DataFrames: {e}", level="error")
            return None

    def _reassign_abbr_flood_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reassigns flood event numbers for the abbreviated CSV.

        Ensures that overlapping flood events are merged and
        numbered sequentially.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of abbreviated flood events.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with reassigned flood event numbers.
        """
        df = df.copy()

        df["start_time_UTC"] = pd.to_datetime(df["start_time_UTC"], utc=True)
        df["end_time_UTC"] = pd.to_datetime(df["end_time_UTC"], utc=True)
        df = df.sort_values(by=["start_time_UTC"]).reset_index(drop=True)

        if df.empty:
            return df

        if "flood_event" not in df.columns:
            df["flood_event"] = 0

        # Get the columns as typed Series objects
        start_times = df["start_time_UTC"]
        end_times = df["end_time_UTC"]

        # Initialize variables with the first row (at position 0)
        current_event_number = 1
        current_end_time = end_times.iloc[0]

        # Set the value for the first row
        df.loc[0, "flood_event"] = current_event_number

        # Iterate by int position from the second row (position 1) onward
        for i in range(1, len(df)):
            # Get data directly from the typed Series
            start_time = start_times.iloc[i]
            end_time = end_times.iloc[i]

            # Check if the current event overlaps
            if start_time < current_end_time:
                # Assign the same flood event number
                df.loc[i, "flood_event"] = current_event_number
                current_end_time = max(current_end_time, end_time)
            else:
                # Assign a new flood event number
                current_event_number += 1
                df.loc[i, "flood_event"] = current_event_number
                # Update the current end time
                current_end_time = end_time

        df["flood_event"] = df["flood_event"].astype(int)
        return df

    def _gen_abbr_flood_event_csv(
        self,
        dataframe: pd.DataFrame,
        csv_filename: str = "abbr_flood_events.csv",
    ):
        """
        Generates an abbreviated (one row per event) flood event CSV.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The raw water level DataFrame from the API.
        csv_filename : str, optional
            The name of the CSV file to save.
        """
        # Read the existing CSV file if it exists
        try:
            existing_data = pd.read_csv(csv_filename)
            last_event_number = (
                existing_data["flood_event"].max() + 1
                if "flood_event" in existing_data.columns
                and not existing_data.empty
                else 0
            )
        except FileNotFoundError:
            column_names = [
                "flood_event",
                "sensor_ID",
                "start_time_UTC",
                "end_time_UTC",
                "start_time_EST",
                "end_time_EST",
                "duration_(hours)",
                "max_road_water_level_(ft)",
                "max_road_water_level_(m)",
            ]
            existing_data = pd.DataFrame(columns=column_names)
            last_event_number = 0

        # Initialize variables
        flood_start_time = None
        max_water_level = 0
        flood_events = []
        eastern = pytz.timezone("EST")

        sorted_dataframe = dataframe.sort_values(by=["date"])
        sensors = sorted_dataframe["sensor_ID"].unique().tolist()

        for sensor in sensors:
            filtered_dataframe = sorted_dataframe[
                sorted_dataframe["sensor_ID"] == sensor
            ]

            for _, row in filtered_dataframe.iterrows():
                water_level = row["road_water_level_adj"]
                timestamp = row["date"]

                # Start of a new flood event
                if water_level > 0.02 and flood_start_time is None:
                    flood_start_time = timestamp
                    max_water_level = water_level

                # During a flood event, track max water level
                if (
                    flood_start_time is not None
                    and water_level > max_water_level
                ):
                    max_water_level = water_level

                # End of a flood event
                if water_level < 0 and flood_start_time is not None:
                    flood_end_time = timestamp
                    start_time_est = flood_start_time.astimezone(eastern)
                    end_time_est = flood_end_time.astimezone(eastern)

                    flood_events.append(
                        {
                            "flood_event": last_event_number,
                            "sensor_ID": sensor,
                            "start_time_UTC": flood_start_time,
                            "end_time_UTC": flood_end_time,
                            "start_time_EST": start_time_est,
                            "end_time_EST": end_time_est,
                            "duration_(hours)": round(
                                (end_time_est - start_time_est).total_seconds()
                                / 3600,
                                3,
                            ),
                            "max_road_water_level_(ft)": round(
                                max_water_level, 3
                            ),
                            "max_road_water_level_(m)": round(
                                max_water_level / 3.28, 3
                            ),
                        }
                    )

                    # Reset variables
                    last_event_number += 1
                    flood_start_time = None
                    max_water_level = 0

        if not flood_events:
            _log("No new flood events found.")
            return

        flood_event_df = pd.DataFrame(flood_events)

        if existing_data.empty:
            updated_data = self._reassign_abbr_flood_numbers(flood_event_df)
        else:
            merged_df = pd.concat(
                [existing_data, flood_event_df], ignore_index=True
            )
            # Ensure UTC times are strings for deduplication, as loaded
            # CSVs may be strings
            merged_df["start_time_UTC"] = merged_df["start_time_UTC"].astype(
                str
            )
            filtered_df = merged_df.drop_duplicates(
                subset=["sensor_ID", "start_time_UTC"], keep="first"
            )
            updated_data = self._reassign_abbr_flood_numbers(filtered_df)

        updated_data.to_csv(csv_filename, index=False)

    def _reassign_flood_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reassigns flood event numbers for the detailed flood event CSV.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of detailed flood events (row-by-row).

        Returns
        -------
        pd.DataFrame
            The same DataFrame with reassigned flood event numbers.
        """
        df = df.copy()

        # Convert time columns to Timestamp
        df["start_time_UTC"] = pd.to_datetime(df["start_time_UTC"], utc=True)
        df["end_time_UTC"] = pd.to_datetime(df["end_time_UTC"], utc=True)
        df["time_UTC"] = pd.to_datetime(df["time_UTC"], utc=True)

        # Create a dictionary to map alphanumeric sensor IDs to numeric
        # values
        sensor_id_list = df["sensor_ID"].unique()
        sensor_id_mapping = {
            sensor_id: i for i, sensor_id in enumerate(sensor_id_list)
        }

        df["Sensor_ID_numeric"] = df["sensor_ID"].map(sensor_id_mapping)

        # Sort the dataframe by time_UTC and sensor ID
        df = df.sort_values(by=["time_UTC", "Sensor_ID_numeric"])
        df = df.reset_index(drop=True)
        df = df.drop(columns=["Sensor_ID_numeric"])

        # Dataframe of rows indicating the end of flood events
        complete_rows = df[df["duration_(hours)"].notnull()]

        if complete_rows.empty:
            _log("No complete flood events to re-number.")
            return df

        # Initialize flood numbers and assigned event dictionary
        flood_event_number = 1
        last_assigned_event = {}
        current_end_time = complete_rows.iloc[0]["end_time_UTC"]

        # Iterate over each complete row (end of flood event)
        for index, row in complete_rows.iterrows():
            if row["start_time_UTC"] < current_end_time:
                # Part of the current combined event
                if row["sensor_ID"] not in last_assigned_event:
                    df.loc[
                        (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index
                elif row["sensor_ID"] in last_assigned_event:
                    df.loc[
                        (df.index > last_assigned_event[row["sensor_ID"]])
                        & (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index

                current_end_time = max(current_end_time, row["end_time_UTC"])
            else:
                # Start of a new combined event
                flood_event_number += 1

                if row["sensor_ID"] not in last_assigned_event:
                    df.loc[
                        (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index
                elif row["sensor_ID"] in last_assigned_event:
                    df.loc[
                        (df.index > last_assigned_event[row["sensor_ID"]])
                        & (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index

                current_end_time = row["end_time_UTC"]

        return df

    def _gen_flood_tracker(
        self, dataframe: pd.DataFrame, csv_filename: str = "flood_events.csv"
    ):
        """
        Generates a detailed (row-by-row) flood event CSV file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            A string representing the name of the created CSV file.
        """
        # Read the existing CSV file if it exists
        try:
            existing_data = pd.read_csv(csv_filename)
            last_event_number = (
                existing_data["flood_event"].max() + 1
                if "flood_event" in existing_data.columns
                and not existing_data.empty
                else 0
            )
        except FileNotFoundError:
            column_names = [
                "flood_event",
                "sensor_ID",
                "time_UTC",
                "time_EST",
                "road_water_level",
                "sensor_water_level",
                "start_time_UTC",
                "end_time_UTC",
                "start_time_EST",
                "end_time_EST",
                "duration_(hours)",
                "max_road_water_level_(ft)",
                "max_road_water_level_(m)",
            ]
            existing_data = pd.DataFrame(columns=column_names)
            last_event_number = 0

        # Initialize variables
        flood_start_time = None
        flood_end_time = None
        max_water_level = 0
        flood_events = []
        eastern = pytz.timezone("EST")

        sorted_dataframe = dataframe.sort_values(by=["date"])
        sensors = sorted_dataframe["sensor_ID"].unique().tolist()

        for sensor in sensors:
            filtered_dataframe = sorted_dataframe[
                sorted_dataframe["sensor_ID"] == sensor
            ]

            for _, row in filtered_dataframe.iterrows():
                water_level = row["road_water_level_adj"]
                sensor_wtr_level = row["sensor_water_level_adj"]
                timestamp = row["date"]

                # Start of a new flood event
                if water_level > 0.02 and flood_start_time is None:
                    flood_start_time = timestamp
                    max_water_level = water_level

                # Track max water level
                if (
                    flood_start_time is not None
                    and water_level > max_water_level
                ):
                    max_water_level = water_level

                # End of a flood event (only set the end time)
                if water_level < 0 and flood_start_time is not None:
                    flood_end_time = timestamp

                if flood_start_time is not None and flood_end_time is None:
                    # During a flood, before it ends
                    flood_events.append(
                        {
                            "flood_event": last_event_number,
                            "sensor_ID": sensor,
                            "time_UTC": timestamp,
                            "time_EST": timestamp.astimezone(eastern),
                            "road_water_level": water_level,
                            "sensor_water_level": sensor_wtr_level,
                            "start_time_UTC": None,
                            "end_time_UTC": None,
                            "start_time_EST": None,
                            "end_time_EST": None,
                            "duration_(hours)": None,
                            "max_road_water_level_(ft)": None,
                            "max_road_water_level_(m)": None,
                        }
                    )
                elif (
                    flood_start_time is not None and flood_end_time is not None
                ):

                    # Define EST times just before they are used.
                    start_time_est = flood_start_time.astimezone(eastern)
                    end_time_est = flood_end_time.astimezone(eastern)

                    # At the end of a flood
                    flood_events.append(
                        {
                            "flood_event": last_event_number,
                            "sensor_ID": sensor,
                            "time_UTC": timestamp,
                            "time_EST": timestamp.astimezone(eastern),
                            "road_water_level": water_level,
                            "sensor_water_level": sensor_wtr_level,
                            "start_time_UTC": flood_start_time,
                            "end_time_UTC": flood_end_time,
                            "start_time_EST": start_time_est,
                            "end_time_EST": end_time_est,
                            "duration_(hours)": round(
                                (end_time_est - start_time_est).total_seconds()
                                / 3600,
                                3,
                            ),
                            "max_road_water_level_(ft)": round(
                                max_water_level, 3
                            ),
                            "max_road_water_level_(m)": round(
                                max_water_level / 3.28, 3
                            ),
                        }
                    )

                    # Reset variables
                    last_event_number += 1
                    flood_start_time = None
                    flood_end_time = None
                    max_water_level = 0

        if not flood_events:
            _log("No new detailed flood data to track.")
            return

        flood_event_df = pd.DataFrame(flood_events)

        if existing_data.empty:
            updated_data = self._reassign_flood_numbers(flood_event_df)
        else:
            merged_df = pd.concat(
                [existing_data, flood_event_df], ignore_index=True
            )
            merged_df["time_UTC"] = merged_df["time_UTC"].astype(str)
            filtered_df = merged_df.drop_duplicates(
                subset=["sensor_ID", "time_UTC"], keep="first"
            )
            updated_data = self._reassign_flood_numbers(filtered_df)

        updated_data.to_csv(csv_filename, index=False)

    def _find_outages(
        self, dataframe: pd.DataFrame, csv_filename: str = "sensor_outages.csv"
    ) -> Optional[pd.DataFrame]:
        """
        Generates a list of outages and saves them to a CSV file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            A string representing the name of the created CSV file.

        Returns
        -------
        Optional[pd.DataFrame]
            A DataFrame of outages, or None if no outages are found.
        """
        sensor_id_list = dataframe["sensor_ID"].unique().tolist()
        outages = []
        eastern = pytz.timezone("EST")

        for sensor_id in sensor_id_list:
            filtered_dataframe = dataframe[dataframe["sensor_ID"] == sensor_id]
            sorted_dataframe = filtered_dataframe.sort_values(
                by="date"
            ).reset_index(drop=True)

            # --- NEW: Capture lat/lon for the astral daylight calculation ---
            lat = (
                sorted_dataframe["lat"].iloc[0]
                if "lat" in sorted_dataframe.columns
                else 0.0
            )
            lon = (
                sorted_dataframe["lng"].iloc[0]
                if "lng" in sorted_dataframe.columns
                else 0.0
            )

            # Use diff() to find time gaps
            time_differences = sorted_dataframe["date"].diff()

            # Identify gaps larger than 1 hour
            gaps = time_differences > pd.Timedelta(hours=1)
            indices_of_gaps = gaps[gaps].index

            for i in indices_of_gaps:
                outage_start_time = sorted_dataframe.loc[i - 1, "date"]
                outage_end_time = sorted_dataframe.loc[i, "date"]
                start_time_est = outage_start_time.astimezone(eastern)
                end_time_est = outage_end_time.astimezone(eastern)

                outages.append(
                    {
                        "outage_number": None,
                        "sensor_ID": sensor_id,
                        "start_time_UTC": outage_start_time,
                        "end_time_UTC": outage_end_time,
                        "start_time_EST": start_time_est,
                        "end_time_EST": end_time_est,
                        "duration_(hours)": round(
                            (end_time_est - start_time_est).total_seconds()
                            / 3600,
                            3,
                        ),
                        "latitude": lat,  # NEW
                        "longitude": lon,  # NEW
                    }
                )

        if not outages:
            _log("No sensor outages found.")
            return None

        outage_dataframe = pd.DataFrame(outages).sort_values(
            by="start_time_UTC"
        )

        # Assign outage numbers sequentially
        outage_dataframe["outage_number"] = range(1, len(outage_dataframe) + 1)

        outage_dataframe.to_csv(csv_filename, index=False)
        return outage_dataframe

    def _check_for_outage_during_flood(
        self,
        outage_csv: str = "sensor_outages.csv",
        abbr_flood_csv: str = "abbr_flood_events.csv",
    ):
        """
        Compares flood events to outages and flags overlaps.

        Uses a `pd.merge` for efficient comparison.

        Parameters
        ----------
        outage_csv : str, optional
            Filename of the outages CSV.
        abbr_flood_csv : str, optional
            Filename of the abbreviated flood events CSV.
        """
        try:
            abbr_floods = pd.read_csv(abbr_flood_csv)
            outage_dataframe = pd.read_csv(outage_csv)
        except FileNotFoundError as e:
            _log(
                f"Error: Cannot check for outages, file not found: {e.filename}",
                level="error",
            )
            return

        if abbr_floods.empty or outage_dataframe.empty:
            _log("No flood or outage data to compare.")
            return

        # Ensure date columns are in datetime format for comparison
        abbr_floods["start_time_UTC"] = pd.to_datetime(
            abbr_floods["start_time_UTC"]
        )
        abbr_floods["end_time_UTC"] = pd.to_datetime(
            abbr_floods["end_time_UTC"]
        )
        outage_dataframe["start_time_UTC"] = pd.to_datetime(
            outage_dataframe["start_time_UTC"]
        )

        # Initialize flag columns
        outage_dataframe["during_flood_event"] = "No"
        outage_dataframe["flood_event_number"] = pd.NA
        abbr_floods["outage"] = "No"

        # Merge on sensor_ID to compare all outages with all floods for
        # that sensor
        merged = pd.merge(
            outage_dataframe,
            abbr_floods,
            on="sensor_ID",
            suffixes=("_outage", "_flood"),
        )

        # Create a mask to find where outage start time is within a
        # flood event
        mask = (
            merged["start_time_UTC_outage"] >= merged["start_time_UTC_flood"]
        ) & (merged["start_time_UTC_outage"] <= merged["end_time_UTC_flood"])

        # Get the original indices of the rows that match
        outage_indices_to_flag = merged.loc[mask, "outage_number"].unique()
        flood_event_nums_to_flag = merged.loc[mask, "flood_event"].unique()

        # Update the outage DataFrame
        matching_outages = merged.loc[
            mask, ["outage_number", "flood_event"]
        ].set_index("outage_number")

        # Update the original outage_dataframe
        for outage_num, flood_event_num in matching_outages[
            "flood_event"
        ].items():
            outage_dataframe.loc[
                outage_dataframe["outage_number"] == outage_num,
                "during_flood_event",
            ] = "Yes"
            outage_dataframe.loc[
                outage_dataframe["outage_number"] == outage_num,
                "flood_event_number",
            ] = flood_event_num

        # Update the flood DataFrame
        abbr_floods.loc[
            abbr_floods["flood_event"].isin(flood_event_nums_to_flag), "outage"
        ] = "Yes"

        # Save the updated DataFrames back to CSVs
        abbr_floods.to_csv(abbr_flood_csv, index=False)
        outage_dataframe.to_csv(outage_csv, index=False)

    def _plot_and_save_flood_plots(
        self,
        sunnyd_data: pd.DataFrame,
        csv_filename: str = "abbr_flood_events.csv",
        plot_folder: str = "flood_plots",
    ):
        """
        Plots flood events listed in the abbreviated flood event CSV.

        Parameters
        ----------
        sunnyd_data : pd.DataFrame
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            Filename of the abbreviated flood events CSV file.
        plot_folder : str, optional
            Name of the directory to save flood plots to.
        """

        # Create a folder to save the plots if it doesn't exist
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
            _log(f"Folder {plot_folder} created successfully.")

        try:
            abbr_dataframe = pd.read_csv(csv_filename)
        except FileNotFoundError:
            _log(
                f"Error: Cannot create plots, file not found: {csv_filename}",
                level="error",
            )
            return

        if abbr_dataframe.empty:
            _log("No flood events in CSV to plot.")
            return

        # Iterate over each row (event) in the DataFrame
        for _, row in abbr_dataframe.iterrows():
            sensor_id = row["sensor_ID"]
            event_number = row["flood_event"]

            # Check if file already exists
            plot_filename = os.path.join(
                plot_folder, f"flood_event_{event_number}_{sensor_id}.png"
            )
            if os.path.exists(plot_filename):
                _log(f"Plot already exists: {plot_filename}")
                continue

            flood_start_time = pd.to_datetime(row["start_time_UTC"], utc=True)
            flood_end_time = pd.to_datetime(row["end_time_UTC"], utc=True)

            plot_start_time = flood_start_time - timedelta(days=1)
            plot_end_time = flood_end_time + timedelta(days=1)

            # Filter the main DataFrame to get data for this plot
            filtered_df = sunnyd_data[
                (sunnyd_data["date"] >= plot_start_time)
                & (sunnyd_data["date"] <= plot_end_time)
                & (sunnyd_data["sensor_ID"] == sensor_id)
            ]

            if filtered_df.empty:
                _log(
                    f"No raw data found for event {event_number} ({sensor_id}). Skipping plot.",
                    level="error",
                )
                continue

            plt.figure(figsize=(12, 10))

            plt.plot(
                filtered_df["date"],
                filtered_df["road_water_level_adj"],
                label="Water Level",
                linestyle="-",
                color="#427e93",
            )

            plt.plot(
                filtered_df["date"],
                np.zeros_like(filtered_df["date"]),
                label="Roadway",
                linestyle="-",
                linewidth=2,
                color="black",
            )

            # Identify the indices within the flood event
            flood_indices = (filtered_df["date"] >= flood_start_time) & (
                filtered_df["date"] <= flood_end_time
            )

            plt.plot(
                filtered_df.loc[flood_indices, "date"],
                filtered_df.loc[flood_indices, "road_water_level_adj"],
                marker="x",
                markersize=5,
                linestyle="None",
                color="#cc0000",
                label="Flood Event Data Points",
            )

            plt.ylim(-2.5, 2)
            plt.xticks(rotation=45)
            plt.yticks(np.arange(-2.5, 2.25, 0.25))
            plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            plt.grid(
                which="major",
                axis="y",
                linestyle="-",
                linewidth="0.5",
                color="gray",
            )
            plt.grid(
                which="minor",
                axis="y",
                linestyle="-",
                linewidth="0.5",
                color="black",
            )
            plt.xlabel("Date")
            plt.ylabel("Water Level (ft above road)")
            plt.title(f"Flood Event {event_number} for {sensor_id}")
            plt.legend(loc="upper right")
            plt.tight_layout()

            plt.savefig(plot_filename)
            plt.close()

    # ===== Static Methods ========

    @staticmethod
    def num_of_flood_days_by_start(csv_name="abbr_flood_events.csv"):
        """Counts unique flood days based on the start date in UTC.

        Parameters
        ----------
        csv_name : str, optional
            The name of the CSV file containing flood events.
            Default is "abbr_flood_events.csv".

        Returns
        -------
        int
            The number of unique flooding days. Returns None if the
            file is not found.
        """

        # Read dataframe
        try:
            read_df = pd.read_csv(csv_name)
        except FileNotFoundError:
            _log(f"Error: File '{csv_name}' not found.", level="error")
            return

        # Convert 'start_time_UTC' to Pandas datetime object
        read_df["start_time_UTC"] = pd.to_datetime(read_df["start_time_UTC"])

        # Extract date part from datetime column
        read_df["date"] = read_df["start_time_UTC"].dt.date

        # Count the number of unique dates
        num_days = read_df["date"].nunique()

        _log(f"Number of days of flooding: {num_days}")
        return num_days

    @staticmethod
    def num_of_flood_days(timezone="EST", csv_name="abbr_flood_events.csv"):
        """Counts unique flood days spanning start and end dates.

        Calculates the total number of unique calendar days a flood
        occurred, considering the full range from start to end time.
        The calculation can be performed in either EST or UTC.

        Parameters
        ----------
        timezone : str, optional
            The timezone to use for date calculation ('EST' or 'UTC').
            Default is "EST".
        csv_name : str, optional
            The name of the CSV file containing flood events.
            Default is "abbr_flood_events.csv".

        Returns
        -------
        int
            The total number of unique flooding days. Returns None
            on file not found or invalid timezone.
        """

        # Read dataframe
        try:
            read_df = pd.read_csv(csv_name)
        except FileNotFoundError:
            _log(f"Error: File '{csv_name}' not found.", level="error")
            return

        if timezone == "UTC":

            # Convert 'start_time_UTC' to Pandas datetime object
            read_df["start_time"] = pd.to_datetime(read_df["start_time_UTC"])
            read_df["end_time"] = pd.to_datetime(read_df["end_time_UTC"])

        elif timezone == "EST":

            # Convert 'start_time_EST' to Pandas datetime object
            read_df["start_time"] = pd.to_datetime(read_df["start_time_EST"])
            read_df["end_time"] = pd.to_datetime(read_df["end_time_EST"])

        else:
            _log(
                f"Error: Timezone '{timezone}' not recognized. Use 'EST' or 'UTC'.",
                level="error",
            )
            return

        # Extract date part from datetime column
        read_df["start_date"] = read_df["start_time"].dt.date
        read_df["end_date"] = read_df["end_time"].dt.date

        # Initialize an empty list to store all dates
        all_dates = []

        # Iterate over each row and append all dates in the range
        for index, row in read_df.iterrows():
            start_date = row["start_date"]
            end_date = row["end_date"]
            date_range = pd.date_range(start=start_date, end=end_date)
            all_dates.extend(date_range)

        # Convert the list of dates to a Pandas Series
        all_dates = pd.Series(all_dates, name="combined_date")

        num_days = all_dates.nunique()

        _log(f"Number of days of flooding: {num_days}")
        return num_days

    @staticmethod
    def avg_errors(
        error_csv, save_file_as=None, csv_filename="abbr_flood_events.csv"
    ):
        """Calculates and appends average errors to the flood events CSV.

        Reads a separate CSV with error data, calculates the 7-day
        rolling average of errors ('Mean', '5th Pct', '95th Pct')
        prior to each flood's end time, and appends these
        averages to the main flood events DataFrame. Saves the
        result to a new or overwritten CSV.

        Parameters
        ----------
        error_csv : str
            Name of the CSV file containing error data.
        save_file_as : str, optional
            File path to save the updated DataFrame. If None,
            it overwrites the file specified by `csv_filename`.
            Default is None.
        csv_filename : str, optional
            Name of the abbreviated flood events CSV to read and
            update. Default is "abbr_flood_events.csv".

        Returns
        -------
        bool
            True if the file was saved successfully, None if an
            error (e.g., FileNotFoundError) occurred.
        """
        # Read CSV files
        try:
            flood_df = pd.read_csv(csv_filename)
            other_df = pd.read_csv(error_csv)
        except FileNotFoundError as e:
            _log(f"Error: File not found. {e}", level="error")
            return

        # Convert 'flood_end_time' to datetime
        flood_df["end_time_UTC"] = pd.to_datetime(
            flood_df["end_time_UTC"], utc=True
        )

        # Tell Pandas to expect mixed formats to safely parse month abbreviations
        other_df["Date"] = pd.to_datetime(
            other_df["Date"], format="mixed", utc=True
        )

        # Loop through each row in flood_df
        averages = []
        for index, row in flood_df.iterrows():
            # Define the time range for a week before the flood end time
            start_time = row["end_time_UTC"] - pd.Timedelta(days=7)
            end_time = row["end_time_UTC"]

            # Select rows from other_df that fall within the defined
            # time range
            selected_rows = other_df[
                (other_df["Date"] >= start_time)
                & (other_df["Date"] <= end_time)
            ]

            # Calculate the average of desired columns
            avg_values = (
                selected_rows[["Mean", "5th Pct", "95th Pct"]].mean().round(4)
            )
            averages.append(avg_values.values)

        # Convert the list of averages to DataFrame
        averages_df = pd.DataFrame(
            averages,
            columns=["Mean of Mean", "Mean of 5th Pct", "Mean of 95th Pct"],
        )

        # Concatenate averages_df with flood_df
        flood_df_with_averages = pd.concat([flood_df, averages_df], axis=1)

        if save_file_as is None:
            save_file = csv_filename
        else:
            save_file = save_file_as

        # Save the result
        flood_df_with_averages.to_csv(save_file, index=False)

        _log(f"File with averages saved successfully to '{save_file}'.")
        return True
    
    def pull_qced_time_series_bulk(
        self,
        qc_csv_path: str,
        output_csv_name: str = "qced_detailed_events.csv",
        chunk_days: int = 30,
        time_buffer_hours: float = 0.0
    ) -> Optional[pd.DataFrame]:
        """
        Takes a QCed abbreviated flood event CSV, uses the robust get_data
        method to pull the full time series for all unique sensors, and 
        cross-references it to extract detailed data for the QCed events.
        """
        try:
            qc_df = pd.read_csv(qc_csv_path)
            qc_df["start_time_UTC"] = pd.to_datetime(qc_df["start_time_UTC"], utc=True)
            qc_df["end_time_UTC"] = pd.to_datetime(qc_df["end_time_UTC"], utc=True)
        except Exception as e:
            _log(f"Error loading QC CSV: {e}", level="error")
            return None

        if qc_df.empty:
            _log("QC CSV is empty.", level="error")
            return None

        # 1. Extract unique sensors to fetch bulk data for
        unique_sensors = qc_df["sensor_ID"].unique().tolist()
        _log(f"Extracting detailed events for {len(qc_df)} events across sensors: {unique_sensors}")

        # 2. Leverage existing robust get_data method for the set date range
        bulk_df = self.get_data(sensor_ids=unique_sensors, chunk_days=chunk_days)
        
        if bulk_df is None or bulk_df.empty:
            _log("Bulk data retrieval failed or returned empty.", level="error")
            return None

        # 3. Cross-reference the bulk data with the QC event times
        _log("Cross-referencing bulk data with QC timestamps...")
        detailed_rows = []

        for _, row in qc_df.iterrows():
            sensor_id = row["sensor_ID"]
            start_time = row["start_time_UTC"]
            end_time = row["end_time_UTC"]
            event_num = row.get("flood_event", pd.NA)

            mask_start = start_time - timedelta(hours=time_buffer_hours)
            mask_end = end_time + timedelta(hours=time_buffer_hours)

            # Slice the bulk dataframe for this specific event and sensor
            event_df = bulk_df[
                (bulk_df["sensor_ID"] == sensor_id) &
                (bulk_df["date"] >= mask_start) & 
                (bulk_df["date"] <= mask_end)
            ].copy()

            if not event_df.empty:
                event_df["flood_event"] = event_num
                detailed_rows.append(event_df)
            else:
                _log(f"No bulk data matched Event {event_num} ({sensor_id}). Likely an outage.", level="info")

        if not detailed_rows:
            _log("No detailed time series data matched any QCed events.", level="error")
            return None

        # 4. Combine and format to match standard output
        combined_df = pd.concat(detailed_rows, ignore_index=True)
        combined_df.sort_values(by=["flood_event", "date"], inplace=True)

        eastern = pytz.timezone("EST")
        combined_df["time_UTC"] = combined_df["date"]
        combined_df["time_EST"] = combined_df["time_UTC"].dt.tz_convert(eastern)

        # Standardize columns
        final_cols = {
            "flood_event": combined_df["flood_event"],
            "sensor_ID": combined_df["sensor_ID"],
            "time_UTC": combined_df["time_UTC"],
            "time_EST": combined_df["time_EST"],
            "road_water_level": combined_df.get("road_water_level_adj", pd.Series(dtype=float)),
            "sensor_water_level": combined_df.get("sensor_water_level_adj", pd.Series(dtype=float)),
        }
        
        final_df = pd.DataFrame(final_cols)
        final_df.to_csv(output_csv_name, index=False)
        
        _log(f"Successfully wrote detailed time series to {output_csv_name}")
        return final_df


class ImageHandler:
    """
    Handles the retrieval and copying of images from an archive
    based on flood event data.

    Parameters
    ----------
    sdfp_image_drive : str or os.PathLike
        The root file path to the image archive drive.

    Attributes
    ----------
    drive : str
        The root file path to the image archive drive.
    copy_destination_dir : str
        The destination folder path set by a public method call.
        This is used by the internal `_copy_file` method.

    """

    def __init__(self, sdfp_image_drive):
        """
        Initializes the ImageHandler with the path to the image drive.

        Parameters
        ----------
        sdfp_image_drive : str or os.PathLike
            The root file path to the image archive drive.
        """
        self.drive = sdfp_image_drive
        _log(
            f"Image Handler initialized. Image Drive: {self.drive}",
            level="info",
        )

    def pull_flood_event_images(
        self,
        destination_folder,
        flood_event_csv_path="abbr_flood_events.csv",
        time_buffer_hours=0,
        max_workers=None,
    ):
        """
        Pulls all images associated with flood events into a single
        folder.

        This is the main public method. It performs the following steps:
        - Creates the destination folder.
        - Loads and formats flood events from the provided CSV.
        - Iterates through each event, spanning multiple days if
          necessary.
        - Applies an optional time buffer to the start and end of
          each event.
        - Lists all files in the relevant archive directories.
        - Filters files to match the event's (buffered) time range.
        - Compiles a unique list of all files to be copied.
        - Uses a thread pool to copy all files to the destination
          folder.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            The path to the folder where all images will be copied.
            It will be created if it does not exist.
        flood_event_csv_path : str or os.PathLike, optional
            The file path to the CSV containing flood event data.
            Defaults to 'abbr_flood_events.csv'.
        time_buffer_hours : int or float, optional
            The number of hours to add as a buffer *before* the
            start_time and *after* the end_time of each event.
            For example, a value of 1 will pull images from
            1 hour before the event started until 1 hour after
            it ended. Defaults to 0 (no buffer).
        max_workers : int, optional
            The maximum number of worker threads to use for copying
            files.
            If None, the default for ThreadPoolExecutor is used.

        """
        _log(
            f"Starting flood event image pull. Destination: {destination_folder}",
            level="info",
        )

        if time_buffer_hours > 0:
            _log(
                f"Applying {time_buffer_hours}-hour buffer to all event times.",
                level="info",
            )

        # Create destination folder if it does not already exist
        if not self._setup_destination_dir(destination_folder):
            return

        try:
            # Load flood events from abbreviated csv
            flood_events = self._format_abbreviated_events_for_pull(
                flood_event_csv_path
            )
            _log(
                f"Loaded {len(flood_events)} flood events from {flood_event_csv_path}",
                level="info",
            )
        except Exception as e:
            _log(
                f"FATAL: Failed to load/format flood events from {flood_event_csv_path}. Reason: {e}",
                level="error",
            )
            return

        if flood_events.empty:
            _log("No flood events found to process.", level="info")
            return

        _log(
            "Gathering file list for all events... (This may take a moment)",
            level="info",
        )
        total_events = len(flood_events)
        event_report_interval = max(1, total_events // 10)
        tasks = []

        for i, row in enumerate(flood_events.itertuples(name="Event")):
            if (
                i == 0
                or (i + 1) == total_events
                or (i + 1) % event_report_interval == 0
            ):
                try:
                    # Try to log with details
                    _log(
                        f"  Processing event {i + 1}/{total_events}: Camera {row.camera_ID} @ {row.start_time_UTC.date()}",
                        level="info",
                    )
                except Exception:
                    # Failsafe log if row attributes are weird
                    _log(
                        f"  Processing event {i + 1}/{total_events}",
                        level="info",
                    )

            try:
                # Set the camera ID
                camera_id = row.camera_ID

                # Apply the time buffer, if any
                buffer = timedelta(hours=time_buffer_hours)
                start_time = row.start_time_UTC - buffer
                end_time = row.end_time_UTC + buffer

                # set the current time to begin iterating through folders
                current_date = start_time.date()
                # set the end date (possible that the flood event
                # occurred over more than one date)
                end_date = end_time.date()

                while current_date <= end_date:
                    # list the files in the current date iteration's
                    # corresponding folder
                    file_list = self._list_files_in_archive_date_dir(
                        current_date, camera_id
                    )

                    # filter the file list to between the start and end
                    # time (using end time rather than end date accounts
                    # for multiday events)
                    filtered_files = self._filter_files_by_start_and_end_time(
                        file_list, start_time, end_time
                    )

                    # add the file paths to the task list to be copied
                    tasks.extend(filtered_files)

                    # continue to the next day
                    current_date += timedelta(days=1)

            except AttributeError as e:
                _log(
                    f"Skipping event row: Missing expected CSV column. Error: {e}",
                    level="error",
                )
            except Exception as e:
                _log(
                    f"Skipping event row due to unexpected error: {e}",
                    level="error",
                )

        _log(
            f"Finished event processing. Found {len(tasks)} files (pre-deduplication).",
            level="info",
        )

        # De-duplicate the list in case events overlapped and found the
        # same file
        tasks = sorted(list(set(tasks)))

        # --- NEW: Check for existing files to prevent redundant copying ---
        _log(
            "Checking for existing images to avoid re-pulling...", level="info"
        )
        existing_files = set(os.listdir(destination_folder))
        new_tasks = [
            t for t in tasks if os.path.basename(t) not in existing_files
        ]
        skipped_count = len(tasks) - len(new_tasks)

        if skipped_count > 0:
            _log(
                f"Skipped {skipped_count} images that already exist locally.",
                level="info",
            )

        # The helper will log the unique file count, handle the
        # "no tasks" case, and execute the copy.
        self._parallel_copy_files(new_tasks, max_workers, "Image pull")

    def pull_outage_images(
        self,
        destination_folder,
        outage_csv_path="sensor_outages.csv",
        max_workers=None,
        only_daylight=True,
    ):
        """
        Pulls images for sensor outages. If only_daylight is True, it limits
        the query to daylight hours (sunrise to sunset) using the astral library.
        """
        if only_daylight:
            try:
                from astral.sun import sun
                from astral import LocationInfo
            except ImportError:
                _log(
                    "The 'astral' package is required for daylight calculations. Run: conda install -c conda-forge astral",
                    level="error",
                )
                return

        if not os.path.exists(outage_csv_path):
            _log(f"Outage CSV not found: {outage_csv_path}", level="info")
            return

        try:
            outages_df = pd.read_csv(outage_csv_path)
            outages_df["start_time_UTC"] = pd.to_datetime(
                outages_df["start_time_UTC"], utc=True
            )
            outages_df["end_time_UTC"] = pd.to_datetime(
                outages_df["end_time_UTC"], utc=True
            )
        except Exception as e:
            _log(f"Failed to load outage CSV: {e}", level="error")
            return

        if outages_df.empty:
            _log("No outages to process.", level="info")
            return

        _log(
            f"Gathering image tasks for {len(outages_df)} outages (Daylight Only: {only_daylight})...",
            level="info",
        )
        if not self._setup_destination_dir(destination_folder):
            return

        tasks = []
        eastern_tz = ZoneInfo("America/New_York")

        for _, row in outages_df.iterrows():
            sensor_id = row.get("sensor_ID")
            if not sensor_id:
                continue

            camera_id = f"CAM_{sensor_id}"
            start_time = row["start_time_UTC"]
            end_time = row["end_time_UTC"]

            if only_daylight:
                lat = row.get("latitude", 0.0)
                lon = row.get("longitude", 0.0)

                # Set up the location for Astral using Local Timezone
                loc = LocationInfo(
                    name=sensor_id,
                    region="NC",
                    timezone="America/New_York",
                    latitude=lat,
                    longitude=lon,
                )

                # Convert to local time to iterate over actual local calendar days
                local_start = start_time.astimezone(eastern_tz)
                local_end = end_time.astimezone(eastern_tz)

                current_date = local_start.date()
                end_date = local_end.date()

                while current_date <= end_date:
                    try:
                        s = sun(
                            loc.observer, date=current_date, tzinfo=eastern_tz
                        )
                        day_start = s["sunrise"].astimezone(timezone.utc)
                        day_end = s["sunset"].astimezone(timezone.utc)

                        query_start = max(start_time, day_start)
                        query_end = min(end_time, day_end)

                        if query_start < query_end:
                            utc_dates_to_check = set(
                                [query_start.date(), query_end.date()]
                            )

                            for u_date in utc_dates_to_check:
                                file_list = (
                                    self._list_files_in_archive_date_dir(
                                        u_date, camera_id
                                    )
                                )
                                filtered_files = (
                                    self._filter_files_by_start_and_end_time(
                                        file_list, query_start, query_end
                                    )
                                )
                                tasks.extend(filtered_files)

                    except Exception as e:
                        _log(
                            f"Failed solar calculation for {sensor_id} on {current_date}: {e}",
                            level="error",
                        )

                    current_date += timedelta(days=1)

            else:
                # Bypass daylight check, pull everything in the outage window
                current_date = start_time.date()
                end_date = end_time.date()

                while current_date <= end_date:
                    file_list = self._list_files_in_archive_date_dir(
                        current_date, camera_id
                    )
                    filtered_files = self._filter_files_by_start_and_end_time(
                        file_list, start_time, end_time
                    )
                    tasks.extend(filtered_files)
                    current_date += timedelta(days=1)

        tasks = sorted(list(set(tasks)))
        _log(f"Found {len(tasks)} outage images.", level="info")

        _log("Checking for existing outage images...", level="info")
        existing_files = set(os.listdir(destination_folder))
        new_tasks = [
            t for t in tasks if os.path.basename(t) not in existing_files
        ]
        skipped_count = len(tasks) - len(new_tasks)

        if skipped_count > 0:
            _log(
                f"Skipped {skipped_count} outage images that already exist locally.",
                level="info",
            )

        self._parallel_copy_files(new_tasks, max_workers, "Outage image pull")

    def copy_images_using_hour_window(
        self,
        image_dir,
        destination_folder,
        start_hour_east=6,
        end_hour_east=19,
        max_workers=None,
    ):
        """
        Copies images from a directory that fall within an Eastern
        Time hour window.

        Parameters
        ----------
        image_dir : str or os.PathLike
            The source directory containing images to filter.
        destination_folder : str or os.PathLike
            The path to the folder where all images will be copied.
            It will be created if it does not exist.
        start_hour_east : int, optional
            The start hour in Eastern Time (inclusive, 0-23).
            Defaults to 6.
        end_hour_east : int, optional
            The end hour in Eastern Time (inclusive, 0-23).
            Defaults to 19.
        max_workers : int, optional
            The maximum number of worker threads to use for copying
            files.
            If None, the default for ThreadPoolExecutor is used.

        """
        tasks = []

        _log(
            f"Starting copy of images matching provided hour window (Eastern). Destination: {destination_folder} Hours: {start_hour_east} to {end_hour_east}",
            level="info",
        )

        # Create destination folder if it does not already exist
        if not self._setup_destination_dir(destination_folder):
            return

        file_list = self._list_files_in_dir(image_dir)
        filtered_files = self._filter_files_eastern_time_window(
            file_list, start_hour_east, end_hour_east
        )

        # add the file paths to the task list to be copied
        tasks.extend(filtered_files)

        # The helper will log the file count, handle the
        # "no tasks" case, and execute the copy.
        self._parallel_copy_files(tasks, max_workers, "Windowed image pull")

    def generate_unlabeled_images_folder(
        self, image_dir, labels_dir, destination_folder, max_workers=None
    ):
        """
        Finds images in 'image_dir' that do NOT have a corresponding
        label in 'labels_dir' and copies them to 'destination_folder'.

        A match is determined by comparing the sensor ID
        (e.g., CAM_XX_00) and the timestamp (YYYYMMDDHHMMSS) extracted
        from the filenames.

        Parameters
        ----------
        image_dir : str or os.PathLike
            The directory containing all images to check.
        labels_dir : str or os.PathLike
            The directory containing labeled images.
        destination_folder : str or os.PathLike
            The path to the folder where all *unlabeled* images
            will be copied. It will be created if it does not exist.
        max_workers : int, optional
            The maximum number of worker threads to use for copying.
            If None, the default for ThreadPoolExecutor is used.
        """
        self.images_folder = Path(image_dir)

        _log(
            f"Starting unlabeled image generation. Destination: {destination_folder}",
            level="info",
        )

        # Setup destination folder
        if not self._setup_destination_dir(destination_folder):
            return

        # Get all files from both directories
        _log(f"Scanning label directory: {labels_dir}", level="info")
        label_files = self._list_files_in_dir(labels_dir)

        _log(f"Scanning image directory: {image_dir}", level="info")
        image_files = self._list_files_in_dir(image_dir)

        if not image_files:
            _log("No images found in the source directory.", level="info")
            return

        # Build the set of labeled keys (sensor_id, timestamp)
        _log("Building key set from label files...", level="info")
        labeled_keys = set()
        for f_path in label_files:
            key = self._get_image_key(os.path.basename(f_path))
            if key:
                labeled_keys.add(key)

        _log(f"Found {len(labeled_keys)} unique label keys.", level="info")

        # Find images that are NOT in the labeled set
        _log(
            "Finding unlabeled images... (This may take a moment)", level="info"
        )
        tasks = []  # This will be our list of files to copy

        total_images = len(image_files)
        report_interval = max(1, total_images // 10)

        # Use a list to store files we couldn't parse, to log at the end
        parse_failures = []

        for i, f_path in enumerate(image_files):
            key = self._get_image_key(
                os.path.basename(f_path), log_failure=False
            )

            if key:
                if key not in labeled_keys:
                    tasks.append(f_path)
            else:
                parse_failures.append(os.path.basename(f_path))

            if (
                i == 0
                or (i + 1) == total_images
                or (i + 1) % report_interval == 0
            ):
                _log(
                    f"  Scanning progress: {i + 1}/{total_images} images checked.",
                    level="info",
                )

        if parse_failures:
            _log(
                f"Could not parse a key for {len(parse_failures)} image files (e.g., '{parse_failures[0]}').",
                level="info",
            )

        # Copy the files
        # The helper will log the file count, handle the
        # "no tasks" case, and execute the copy.
        self._parallel_copy_files(tasks, max_workers, "Unlabeled image copy")

    def create_test_image_set(
        self,
        destination_folder,
        camera_id,
        num_images,
        start_year=2023,
        end_year=2024,
        hour_window=None,
        max_workers=None,
    ):
        """
        Creates a test set of random images for a camera.

        This method combines `_get_random_images` to find a list
        of files and then uses parallel copy logic to move
        them into the specified destination folder.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            The path to the folder where all images will be copied.
            It will be created if it does not exist.
        camera_id : str
            The ID of the camera (e.g., "CAM_XX_00").
        num_images : int
            The total number of images to try and retrieve.
        start_year : int, optional
            The starting year (inclusive) to search. Defaults to 2023.
        end_year : int, optional
            The ending year (inclusive) to search. Defaults to 2024.
        hour_window : tuple(int, int), optional
            A (start_hour, end_hour) tuple in Eastern Time.
            Defaults to None (all hours).
        max_workers : int, optional
            The maximum number of worker threads to use for copying.
            If None, the default for ThreadPoolExecutor is used.
        """

        _log(
            f"Starting test image set creation. Destination: {destination_folder}",
            level="info",
        )

        # Setup destination folder
        if not self._setup_destination_dir(destination_folder):
            return

        # Get the list of random images
        _log(
            f"Selecting {num_images} random images for {camera_id}...",
            level="info",
        )
        tasks = self._get_random_images(
            camera_id,
            start_year,
            end_year,
            num_images=num_images,
            hour_window=hour_window,
        )

        # Copy files in parallel
        # The helper will log the file count, handle the
        # "no tasks" case, and execute the copy.
        self._parallel_copy_files(tasks, max_workers, "Test set creation")

    def _format_abbreviated_events_for_pull(self, flood_event_csv_path):
        """
        Loads and formats flood event data from a CSV file.

        Reads the specified CSV, keeps only necessary columns, converts
        time strings to timezone-aware datetime objects, and creates
        a 'camera_ID' column from the 'sensor_ID'.

        Parameters
        ----------
        flood_event_csv_path : str or os.PathLike
            The file path to the abbreviated flood events CSV.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ['camera_ID', 'start_time_UTC',
            'end_time_UTC'].

        """
        columns_to_read = ["sensor_ID", "start_time_UTC", "end_time_UTC"]
        final_columns = ["camera_ID", "start_time_UTC", "end_time_UTC"]
        date_format = "%Y-%m-%d %H:%M:%S%z"

        flood_events = (
            # Read in necessary columns from abbr flood events csv
            pd.read_csv(flood_event_csv_path, usecols=columns_to_read).assign(
                # Convert times from strings to datetimes
                start_time_UTC=lambda x: pd.to_datetime(
                    x["start_time_UTC"], format=date_format
                ),
                end_time_UTC=lambda x: pd.to_datetime(
                    x["end_time_UTC"], format=date_format
                ),
                # Create camera ID column because sensor images are
                # stored with camera name
                camera_ID=lambda x: "CAM_" + x["sensor_ID"],
            )
        )

        # Drop unnecessary sensor_id column from dataframe
        return flood_events[final_columns]

    def _list_files_in_archive_date_dir(self, date, camera_id):
        """
        Lists all files in the archive directory for a specific date and
        camera.

        Constructs a directory path based on the archive structure
        (e.g., .../<YYYY> Archive/<camera_id>/<YYYY-MM-DD>/)
        and returns a list of all files within it.

        Parameters
        ----------
        date : datetime.date or datetime.datetime
            The date for which to list files.
        camera_id : str
            The ID of the camera.

        Returns
        -------
        list[str]
            A list of absolute file paths. Returns an empty list if the
            directory does not exist or an error occurs.

        """
        # Set directory to match provided date
        directory = (
            Path(self.drive)
            / f"{date.year} Archive"
            / camera_id
            / date.strftime("%Y-%m-%d")
        )

        if not os.path.isdir(directory):
            # This is an informational message, not a critical failure
            _log(
                f"Directory not found (this may be expected): {directory}",
                level="info",
            )
            return []

        try:
            # Re-check with Path object method
            if not directory.is_dir():
                return []

            return [os.fspath(p) for p in directory.iterdir() if p.is_file()]

        except OSError as e:
            # e.g., if you lack read permissions on 'directory'
            _log(
                f"Error reading files from directory '{directory}'. Reason: {e}",
                level="error",
            )
            return []  # Return an empty list on failure

    def _list_files_in_dir(self, directory_path):
        """
        Lists all files in a given directory.

        Parameters
        ----------
        directory_path : str or os.PathLike
            The directory to scan.

        Returns
        -------
        list[str]
            A list of absolute file paths. Returns an empty list
            if the directory is invalid or an error occurs.
        """
        directory = Path(directory_path)
        if not directory.is_dir():
            _log(f"Directory not found: {directory_path}", level="info")
            return []

        try:
            return [os.fspath(p) for p in directory.iterdir() if p.is_file()]
        except OSError as e:
            _log(
                f"Error reading files from '{directory_path}'. Reason: {e}",
                level="error",
            )
            return []

    def _filter_files_by_condition(self, file_list, condition_func):
        """
        Filters files based on a timestamp-checking function.

        Iterates a file list, parses the UTC timestamp from each
        filename, and applies the `condition_func` to it.

        Parameters
        ----------
        file_list : list[str]
            A list of file paths to filter.
        condition_func : callable
            A function that accepts one argument (a timezone-aware
            UTC datetime object) and returns True if the file
            should be included in the list, False otherwise.

        Returns
        -------
        list[str]
            The sub-list of file paths that satisfy the condition.
        """
        filtered_files = []

        # For each file in provided list
        for file in file_list:
            try:
                # strip timestamp from file name
                file_name = os.path.basename(file)
                file_timestamp_str = self._extract_timestamp(file_name)

                # and localize to time aware UTC
                file_timestamp_naive = datetime.strptime(
                    file_timestamp_str, "%Y%m%d%H%M%S"
                )
                file_timestamp_utc = file_timestamp_naive.replace(
                    tzinfo=timezone.utc
                )

                # Apply the provided condition function
                if condition_func(file_timestamp_utc):
                    filtered_files.append(file)
            except (IndexError, ValueError, TypeError):
                # Log the specific file that failed parsing, continue
                _log(
                    f"Skipping file: Could not parse timestamp from '{file_name}'.",
                    level="info",
                )
                continue

        return filtered_files

    def _filter_files_by_start_and_end_time(
        self, file_list, start_time, end_time
    ):
        """
        Filters a file list to those within a specified time range.

        Assumes filenames contain a UTC timestamp in the format
        '..._YYYYMMDDHHMMSS.ext'. Both start_time and end_time must be
        timezone-aware.

        Parameters
        ----------
        file_list : list[str]
            A list of file paths to filter.
        start_time : datetime.datetime
            The timezone-aware start of the time range (inclusive).
        end_time : datetime.datetime
            The timezone-aware end of the time range (inclusive).

        Returns
        -------
        list[str]
            The sub-list of file paths that fall within the time range.

        """
        # Comparing aware and naive datetimes raises a TypeError.
        # This check ensures the inputs are valid before starting the
        # loop.
        if start_time.tzinfo is None or end_time.tzinfo is None:
            _log(
                "Cannot filter files: start_time and end_time must be timezone-aware.",
                level="error",
            )
            return []

        # Define the condition and pass it to the helper
        condition = lambda ft_utc: start_time <= ft_utc <= end_time

        return self._filter_files_by_condition(file_list, condition)

    def _filter_files_eastern_time_window(
        self, file_list, start_hour_east, end_hour_east
    ):
        """
        Filters files to those between specific hours in Eastern Time.

        Parameters
        ----------
        file_list : list[str]
            A list of file paths to filter.
        start_hour_east : int
            The start hour in Eastern Time (inclusive), e.g., 6.
        end_hour_east : int
            The end hour in Eastern Time (inclusive), e.g., 18.

        Returns
        -------
        list[str]
            The sub-list of file paths that fall within the time range.
        """
        try:
            # Define the timezone
            eastern_tz = ZoneInfo("America/New_York")
        except Exception as e:
            _log(
                f"Could not load timezone 'America/New_York'. Error: {e}",
                level="error",
            )
            return []

        condition = lambda ft_utc: (
            start_hour_east
            <= ft_utc.astimezone(eastern_tz).hour
            <= end_hour_east
        )

        return self._filter_files_by_condition(file_list, condition)

    def _setup_destination_dir(self, destination_folder):
        """
        Ensures the destination folder exists and sets it as the
        copy target.

        This method creates the folder (including parent directories)
        if it does not exist. It also sets the
        `self.copy_destination_dir` attribute, which is required
        by the `_copy_file` method.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            The path to the folder to create.

        Returns
        -------
        bool
            True if the directory was created or already exists,
            False if an OSError occurred.
        """
        try:
            os.makedirs(destination_folder, exist_ok=True)
            self.copy_destination_dir = destination_folder
            _log(
                f"Ensured destination folder exists: {destination_folder}",
                level="info",
            )
            return True
        except OSError as e:
            _log(
                f"FATAL: Could not create destination folder '{destination_folder}'. Reason: {e}",
                level="error",
            )
            return False

    def _parallel_copy_files(self, tasks, max_workers, task_name="File copy"):
        """
        Copies a list of files in parallel to the destination
        directory.

        Uses a ThreadPoolExecutor to copy all files from the
        `tasks` list to the `self.copy_destination_dir`. It logs
        progress and a final summary.

        Parameters
        ----------
        tasks : list[str]
            A list of absolute file paths to be copied.
        max_workers : int or None
            The maximum number of worker threads to use.
        task_name : str, optional
            A human-readable name for the task, used in log
            messages. Defaults to "File copy".

        """
        total_files = len(tasks)
        if total_files == 0:
            _log(
                f"No files found to copy for task: '{task_name}'.", level="info"
            )
            return

        # Determine report interval for copying
        copy_report_interval = max(1, total_files // 10)

        _log(
            f"Starting parallel copy of {total_files} files (max_workers={max_workers or 'default'})...",
            level="info",
        )
        success_count = 0
        fail_count = 0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # map() submits all tasks and returns an iterator of results
            # in the order the tasks were submitted.
            results = executor.map(self._copy_file, tasks)

            # Iterate over results to count success/failure
            for i, result in enumerate(results):
                # _copy_file returns a truthy value on success
                # and a falsy (False) on failure.
                if result:
                    # result is truthy, so it's a success
                    success_count += 1
                else:
                    # result is falsy (False), so it's a failure
                    fail_count += 1

                if (
                    i == 0
                    or (i + 1) == total_files
                    or (i + 1) % copy_report_interval == 0
                ):
                    _log(
                        f"  Copy progress: {i + 1}/{total_files} files processed.",
                        level="info",
                    )

        _log(
            f"{task_name} complete. Copied: {success_count}, Failed: {fail_count}",
            level="info",
        )

    def _copy_file(self, file_path):
        """
        Copies a single file to the class's destination folder.

        Relies on `self.copy_destination_dir` being set by the
        calling method. Logs errors and returns a status.

        Parameters
        ----------
        file_path : str or os.PathLike
            The path to the source file to be copied.

        Returns
        -------
        bool
            True if the copy is successful, False otherwise.

        """
        # Check the destination folder has been set by the calling method
        if not hasattr(self, "copy_destination_dir"):
            _log(
                "Cannot copy: self.copy_destination_dir is not set.",
                level="error",
            )
            return False  # Return falsy on failure

        try:
            # shutil.copy will copy the file and its permissions.
            shutil.copy(file_path, self.copy_destination_dir)

            return True  # Return truthy on success

        except FileNotFoundError:
            # This is a common, expected error
            _log(f"File not found, could not copy: {file_path}", level="error")
            return False  # Return falsy on failure

        except shutil.Error as e:
            # This catches specific shutil errors (e.g., "Disk Full")
            _log(
                f"shutil error copying {file_path}. Reason: {e}", level="error"
            )
            return False  # Return falsy on failure

        except OSError as e:
            # This catches lower-level OS errors (e.g., "Permission Denied")
            _log(f"OS error copying {file_path}. Reason: {e}", level="error")
            return False  # Return falsy on failure

    def _get_random_images(
        self, camera_id, start_year, end_year, num_images=10, hour_window=None
    ):
        """
        Selects a random sample of images for a given camera.

        It searches within a year range, optionally filters for an
        Eastern Time hour window, and attempts to pull a specified
        number of images. The sampling is distributed by picking
        random dates and taking a small, random number of images
        from each date to ensure variety.

        Parameters
        ----------
        camera_id : str
            The ID of the camera (e.g., "CAM_XX_00").
        start_year : int
            The starting year (inclusive) to search for images.
        end_year : int
            The ending year (inclusive) to search for images.
        num_images : int, optional
            The total number of images to try and retrieve.
            Defaults to 10.
        hour_window : tuple(int, int), optional
            A (start_hour, end_hour) tuple in Eastern Time.
            If provided, only images within this window (inclusive)
            will be selected. Defaults to None (all hours).

        Returns
        -------
        list[str]
            A list of absolute file paths to the selected images.
            May contain fewer images than `num_images` if not
            enough were found.
        """
        _log(
            f"Starting random image selection for '{camera_id}' ({start_year}-{end_year})...",
            level="info",
        )

        try:
            date_list = self._generate_date_list(start_year, end_year)
            random.shuffle(date_list)
        except Exception as e:
            _log(
                f"FATAL: Failed to generate or shuffle date list. Reason: {e}",
                level="error",
            )
            return []

        selected_images = []

        while len(selected_images) < num_images and date_list:
            date = date_list.pop()

            # Construct the directory path
            date_dir = (
                Path(self.drive)
                / f"{date.year} Archive"
                / camera_id
                / date.strftime("%Y-%m-%d")
            )

            # _list_files_in_dir handles logging if dir not found
            images = self._list_files_in_dir(date_dir)

            if not images:
                continue  # Skip this date if dir was empty or missing

            # Apply hour window filter if provided
            if hour_window:
                try:
                    start_hour_east, end_hour_east = hour_window
                    filtered_images = self._filter_files_eastern_time_window(
                        images, start_hour_east, end_hour_east
                    )
                except Exception as e:
                    _log(
                        f"Error applying time filter for {date_dir}: {e}",
                        level="error",
                    )
                    continue  # Skip this day
            else:
                filtered_images = images  # Use all images

            if not filtered_images:
                continue  # Skip if filter produced no results

            random.shuffle(filtered_images)

            # This logic takes a random small chunk (1-15% of total
            # requested) from the day to distribute the sample.
            percentage = random.uniform(0.01, 0.15)
            num_to_select = max(1, int(num_images * percentage))

            # How many do we still need?
            needed = num_images - len(selected_images)

            # How many are available on this day?
            available = len(filtered_images)

            # Take the minimum of the number we want, need, and have
            num_to_add = min(num_to_select, needed, available)

            selected_images.extend(filtered_images[:num_to_add])

        if len(selected_images) < num_images:
            _log(
                f"Selection complete. Warning: Only found {len(selected_images)} "
                f"of {num_images} requested images.",
                level="info",
            )
        else:
            _log(
                f"Successfully selected {len(selected_images)} random images.",
                level="info",
            )

        return selected_images

    @staticmethod
    def _get_image_key(filename, log_failure=True):
        """
        Extracts the (sensor, timestamp) key from a filename.

        Parameters
        ----------
        filename : str
            The basename of the file.
        log_failure : bool, optional
            If True, logs a message if a key cannot be parsed.
            Defaults to True.

        Returns
        -------
        tuple (str, str) or None
            A tuple of (sensor_id, timestamp) or None if either
            part cannot be parsed.
        """
        sensor_id = ImageHandler._extract_sensor_name(filename)
        timestamp = ImageHandler._extract_timestamp(filename)

        if sensor_id and timestamp:
            return (sensor_id, timestamp)

        if log_failure:
            _log(
                f"Skipping file: Could not parse key from '{filename}'.",
                level="info",
            )
        return None

    @staticmethod
    def _extract_timestamp(filename):
        """
        Extracts UTC timestamp (YYYYMMDDHHMMSS) from a filename.

        Parameters
        ----------
        filename : str
            The basename of the file.

        Returns
        -------
        str or None
            The 14-digit timestamp string, or None if not found.
        """
        # Regular expression pattern to match the UTC timestamp
        pattern = r"\d{14}"

        match = re.search(pattern, filename)
        return match.group(0) if match else None

    @staticmethod
    def _extract_sensor_name(filename):
        """
        Extracts sensor ID (CAM_XX_00) from a filename.

        Parameters
        ----------
        filename : str
            The basename of the file.

        Returns
        -------
        str or None
            The sensor ID string, or None if not found.
        """
        # Regular expression pattern to match the sensor name
        pattern = r"CAM_[A-Z]{2}_[0-9]{2}"

        match = re.search(pattern, filename)
        return match.group(0) if match else None

    @staticmethod
    def _generate_date_list(start_year, end_year):
        """
        Generates a list of all dates within a given year range.

        Parameters
        ----------
        start_year : int
            The starting year (inclusive).
        end_year : int
            The ending year (inclusive).

        Returns
        -------
        list[datetime.date]
            A list of all dates from YYYY-01-01 to YYYY-12-31
            for the specified year range.

        Raises
        ------
        ValueError
            If `end_year` is before `start_year`.
        """
        if end_year < start_year:
            raise ValueError("end_year cannot be before start_year")

        _log(
            f"Generating date list from {start_year} to {end_year}...",
            level="info",
        )
        date_list = []
        start_date = datetime(start_year, 1, 1).date()
        end_date = datetime(end_year, 12, 31).date()

        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)

        _log(f"Generated {len(date_list)} dates.", level="info")
        return date_list
