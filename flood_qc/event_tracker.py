"""
Water level data retrieval and flood event detection.

This module defines :class:`EventTracker`, a client for the Sunny Day
Flooding Project (SDFP) water level API. It downloads water level data
for one or more sensors, detects flood events and sensor outages, writes
them to CSV files, and plots each flood event.

Classes
-------
EventTracker
    Download SDFP water level data and catalog flood events.
"""

import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytz
import requests

from .utils import log

# Fixed Eastern Standard Time (UTC-5 all year, no daylight saving time),
# used for every "_EST" column in the output CSVs
EST = pytz.timezone("EST")

# Road water level (ft above the road) that a reading must exceed to
# start a flood event
FLOOD_START_THRESHOLD_FT = 0.02

# Road water level (ft above the road) that a reading must drop below
# to end a flood event
FLOOD_END_THRESHOLD_FT = 0.0

# Divisor used to convert the maximum water level from feet to meters
FT_PER_M = 3.28

# Gap between consecutive readings that counts as a sensor outage
OUTAGE_GAP = pd.Timedelta(hours=1)

# Sensor IDs installed in each named region
SENSOR_REGIONS = {
    "carolina beach": ["CB_01", "CB_02", "CB_03"],
    "beaufort": ["BF_01"],
    "down east": ["DE_01", "DE_02", "DE_03", "DE_04"],
    "new bern": ["NB_01", "NB_02"],
    "north river": ["NR_01"],
}


class EventTracker:
    """
    Client for the SDFP water level API with flood event detection.

    Downloads water level data for a date range, then detects flood
    events and sensor outages, writes them to CSV files, and plots each
    flood event.

    Attributes
    ----------
    BASE_URL : str
        The SDFP API endpoint that returns water level data.
    authorization : tuple of (str, str)
        The (username, password) pair for the SDFP API.
    min_date : datetime.date or datetime.datetime
        First day of data to request (inclusive).
    max_date : datetime.date or datetime.datetime
        Last day of data to request (inclusive).
    max_workers : int
        Number of threads used to send API requests in parallel.
    timeout : int
        Timeout for each API request, in seconds.
    """

    # SDFP API endpoint that returns water level readings
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
        Initialize the tracker and validate its inputs.

        Parameters
        ----------
        authorization : tuple of (str, str)
            The (username, password) pair for the SDFP API.
        min_date : datetime.date or datetime.datetime
            First day of data to request (inclusive).
        max_date : datetime.date or datetime.datetime
            Last day of data to request (inclusive).
        max_workers : int, optional
            Number of threads used to send API requests in parallel.
            Default is 16.
        timeout : int, optional
            Timeout for each API request, in seconds. Default is 60.

        Raises
        ------
        ValueError
            If `authorization` is missing or incomplete, or if
            `min_date` or `max_date` is not provided.
        """
        # Require both a username and a password
        if not authorization or not all(authorization):
            raise ValueError(
                "Authorization (username, password) tuple is required."
            )
        # Require both ends of the date range
        if not min_date or not max_date:
            raise ValueError("min_date and max_date are required.")

        # Store the API credentials
        self.authorization = authorization
        # Store the date range to request
        self.min_date = min_date
        self.max_date = max_date
        # Store the request settings
        self.max_workers = max_workers
        self.timeout = timeout

        # Report the settings in the process log
        log(
            f"EventTracker initialized. Max workers: {self.max_workers}, "
            f"Timeout: {self.timeout}s"
        )
        log(f"Date Range Set: {self.min_date} to {self.max_date}")

    # ===== Public pipeline methods =====

    def get_data(
        self, sensor_ids: str | List[str], chunk_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Download water level data for the configured date range.

        The date range is split into chunks of `chunk_days` days and
        every (sensor, chunk) pair is requested in parallel. Any chunk
        that fails is retried one day at a time.

        Parameters
        ----------
        sensor_ids : str or list of str
            A region name (e.g. ``"Down East"``) or a list of sensor
            IDs (e.g. ``["DE_01", "DE_02"]``).
        chunk_days : int, optional
            Number of days requested per API call. Default is 30.

        Returns
        -------
        pandas.DataFrame or None
            Every reading downloaded, de-duplicated and sorted by date,
            or None if no data could be retrieved.
        """
        # Turn a region name into its sensor IDs, or keep a given list
        sensor_ids_list = self._resolve_sensor_ids(sensor_ids)
        # Stop if the region name was not recognized (already logged)
        if sensor_ids_list is None:
            return None

        # Split the full date range into chunks of chunk_days days
        date_chunks = self._create_date_chunks(chunk_days)
        # Stop if the date range produced no chunks
        if not date_chunks:
            log(
                "No date chunks to process (check min/max dates).",
                level="error",
            )
            return None

        # Count one request per sensor per chunk for the progress log
        total_tasks = len(sensor_ids_list) * len(date_chunks)
        log(
            f"Total date range split into {len(date_chunks)} chunks of "
            f"~{chunk_days} days each."
        )
        log(
            f"Starting chunked download for {len(sensor_ids_list)} "
            f"sensors... ({total_tasks} total requests)"
        )

        # Collect every DataFrame returned by the requests
        all_dataframes = []

        # Share one HTTP session across all requests
        with requests.Session() as session:
            # Attach the API username and password to the session
            session.auth = self.authorization

            # Run the chunk requests on a pool of worker threads
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit one request per (sensor, chunk) pair
                futures = self._submit_chunk_tasks(
                    executor, session, sensor_ids_list, date_chunks
                )
                # Gather the results and note which chunks failed
                primary_data, failed_tasks = self._process_chunk_results(
                    futures
                )
                # Keep the data from the chunks that succeeded
                all_dataframes.extend(primary_data)

            # Retry any failed chunks one day at a time
            if failed_tasks:
                fallback_data = self._handle_failed_chunks(
                    session, failed_tasks
                )
                # Keep the data recovered by the daily retries
                all_dataframes.extend(fallback_data)

        log("Data retrieval complete.")

        # Merge all chunks into one de-duplicated, sorted DataFrame
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
        Run the full pipeline: download, detect events, and plot.

        Downloads the water level data, writes the detailed and
        abbreviated flood event CSVs, writes the sensor outage CSV,
        flags outages that start during flood events, and saves a plot
        of each flood event.

        Parameters
        ----------
        location : str or list of str
            A region name (e.g. ``"Down East"``) or a list of sensor
            IDs.
        chunk_days : int, optional
            Number of days requested per API call. Default is 30.
        output_dir : str, optional
            Folder that all outputs are written to. Default is
            ``"data"``.
        full_csv_name : str, optional
            File name of the detailed (one row per reading) flood event
            CSV. Default is ``"flood_events.csv"``.
        abbr_csv_name : str, optional
            File name of the abbreviated (one row per event) flood
            event CSV. Default is ``"abbr_flood_events.csv"``.
        outage_csv_name : str, optional
            File name of the sensor outage CSV. Default is
            ``"sensor_outages.csv"``.
        plot_folder : str, optional
            Name of the subfolder of `output_dir` that the flood plots
            are saved to. Default is ``"flood_plots"``.

        Returns
        -------
        pandas.DataFrame or None
            The downloaded water level data, or None if the download
            failed or returned no data.
        """
        # Make sure the output folder exists
        os.makedirs(output_dir, exist_ok=True)
        # Build the full path of every output inside the output folder
        full_csv_name = os.path.join(output_dir, full_csv_name)
        abbr_csv_name = os.path.join(output_dir, abbr_csv_name)
        outage_csv_name = os.path.join(output_dir, outage_csv_name)
        plot_folder = os.path.join(output_dir, plot_folder)

        # Download the water level data for the location
        log(f"Starting data pull for location: '{location}'")
        download_data = self.get_data(location, chunk_days)

        # Stop if nothing was downloaded
        if download_data is None or download_data.empty:
            log(
                "Data download failed or returned no data. "
                "Stopping processing.",
                level="error",
            )
            return None

        log(f"Data pull successful. Shape: {download_data.shape}")

        # Write the detailed and abbreviated flood event CSVs
        log("Generating CSVs...")
        try:
            self._gen_flood_tracker(download_data, full_csv_name)
            self._gen_abbr_flood_event_csv(download_data, abbr_csv_name)
            log("CSV files created successfully.")
        # Log the error and continue so the remaining steps still run
        except Exception as e:
            log(f"An error occurred during CSV generation: {e}", "error")

        # Write the outage CSV and cross-flag outages and flood events
        log("Checking for sensor outages...")
        try:
            self._find_outages(download_data, outage_csv_name)
            self._check_for_outage_during_flood(outage_csv_name, abbr_csv_name)
            log("Sensor outage logs generated and CSVs appended.")
        # Log the error and continue so plotting still runs
        except Exception as e:
            log(f"An error occurred during outage check: {e}", "error")

        # Save a water level plot for every flood event
        log("Plotting flood events...")
        try:
            self._plot_and_save_flood_plots(
                download_data, abbr_csv_name, plot_folder
            )
            log("Plotting completed.")
        # Log the error; the CSVs written above are still usable
        except Exception as e:
            log(f"An error occurred during plotting: {e}", level="error")

        # Return the raw data so callers can reuse it
        return download_data

    def pull_qced_time_series_bulk(
        self,
        qc_csv_path: str,
        output_csv_name: str = "qced_detailed_events.csv",
        chunk_days: int = 30,
        time_buffer_hours: float = 0.0,
    ) -> Optional[pd.DataFrame]:
        """
        Download the full time series for every event in a QC CSV.

        Pulls the water level data for every sensor in the QC CSV over
        the tracker's date range, then keeps only the readings inside
        each QC'd event window (plus an optional buffer).

        Parameters
        ----------
        qc_csv_path : str
            Path to a QC results CSV with ``sensor_ID``,
            ``start_time_UTC``, ``end_time_UTC``, and ``flood_event``
            columns.
        output_csv_name : str, optional
            Path of the detailed CSV to write. Default is
            ``"qced_detailed_events.csv"``.
        chunk_days : int, optional
            Number of days requested per API call. Default is 30.
        time_buffer_hours : float, optional
            Hours of data to keep before and after each event. Default
            is 0.0.

        Returns
        -------
        pandas.DataFrame or None
            One row per reading with ``flood_event``, ``sensor_ID``,
            ``time_UTC``, ``time_EST``, ``road_water_level``, and
            ``sensor_water_level`` columns, or None on failure.
        """
        # Load the QC CSV and parse its event times as UTC
        try:
            qc_df = pd.read_csv(qc_csv_path)
            qc_df["start_time_UTC"] = pd.to_datetime(
                qc_df["start_time_UTC"], utc=True
            )
            qc_df["end_time_UTC"] = pd.to_datetime(
                qc_df["end_time_UTC"], utc=True
            )
        # Stop if the file is missing or its columns are malformed
        except Exception as e:
            log(f"Error loading QC CSV: {e}", level="error")
            return None

        # Stop if the QC CSV has no events
        if qc_df.empty:
            log("QC CSV is empty.", level="error")
            return None

        # List the sensors that need data
        unique_sensors = qc_df["sensor_ID"].unique().tolist()
        log(
            f"Extracting detailed events for {len(qc_df)} events across "
            f"sensors: {unique_sensors}"
        )

        # Download all readings for those sensors over the date range
        bulk_df = self.get_data(
            sensor_ids=unique_sensors, chunk_days=chunk_days
        )

        # Stop if the download failed
        if bulk_df is None or bulk_df.empty:
            log("Bulk data retrieval failed or returned empty.", "error")
            return None

        # Keep only the readings inside each QC'd event window
        log("Cross-referencing bulk data with QC timestamps...")
        detailed_rows = []

        for _, row in qc_df.iterrows():
            # Read this event's sensor, window, and number
            sensor_id = row["sensor_ID"]
            start_time = row["start_time_UTC"]
            end_time = row["end_time_UTC"]
            event_num = row.get("flood_event", pd.NA)

            # Widen the window by the buffer on each side
            mask_start = start_time - timedelta(hours=time_buffer_hours)
            mask_end = end_time + timedelta(hours=time_buffer_hours)

            # Select this sensor's readings inside the window
            event_df = bulk_df[
                (bulk_df["sensor_ID"] == sensor_id)
                & (bulk_df["date"] >= mask_start)
                & (bulk_df["date"] <= mask_end)
            ].copy()

            # Label the readings with the event number and keep them
            if not event_df.empty:
                event_df["flood_event"] = event_num
                detailed_rows.append(event_df)
            # No readings means the event was likely found in an outage
            else:
                log(
                    f"No bulk data matched Event {event_num} "
                    f"({sensor_id}). Likely an outage."
                )

        # Stop if no event had any readings
        if not detailed_rows:
            log(
                "No detailed time series data matched any QCed events.",
                level="error",
            )
            return None

        # Stack all events and order them by event, then time
        combined_df = pd.concat(detailed_rows, ignore_index=True)
        combined_df.sort_values(by=["flood_event", "date"], inplace=True)

        # Add UTC and EST time columns
        combined_df["time_UTC"] = combined_df["date"]
        combined_df["time_EST"] = combined_df["time_UTC"].dt.tz_convert(EST)

        # Keep only the standard output columns, renamed to match
        # flood_events.csv (water levels are empty if not returned)
        final_cols = {
            "flood_event": combined_df["flood_event"],
            "sensor_ID": combined_df["sensor_ID"],
            "time_UTC": combined_df["time_UTC"],
            "time_EST": combined_df["time_EST"],
            "road_water_level": combined_df.get(
                "road_water_level_adj", pd.Series(dtype=float)
            ),
            "sensor_water_level": combined_df.get(
                "sensor_water_level_adj", pd.Series(dtype=float)
            ),
        }

        # Build the output table and save it
        final_df = pd.DataFrame(final_cols)
        final_df.to_csv(output_csv_name, index=False)

        log(f"Successfully wrote detailed time series to {output_csv_name}")
        return final_df

    # ===== Sensor and date helpers =====

    def _sensor_list_generator(self, location_name: str) -> List[str]:
        """
        Look up the sensor IDs installed in a named region.

        Parameters
        ----------
        location_name : str
            Region name, not case-sensitive (e.g. ``"Carolina Beach"``).
            See ``SENSOR_REGIONS`` for the supported names.

        Returns
        -------
        list of str
            The sensor IDs in the region.

        Raises
        ------
        ValueError
            If `location_name` is not a known region.
        """
        # Look the region up by its lower-case name
        sensor_ids = SENSOR_REGIONS.get(location_name.lower())
        # Fail clearly if the region is unknown
        if sensor_ids is None:
            raise ValueError(
                f"Location name '{location_name}' not recognized."
            )
        # Return a copy so callers cannot change the lookup table
        return list(sensor_ids)

    def _resolve_sensor_ids(
        self, sensor_ids_input: str | List[str]
    ) -> Optional[List[str]]:
        """
        Turn a region name or a list of sensor IDs into a list.

        Parameters
        ----------
        sensor_ids_input : str or list of str
            A region name, or a list of sensor IDs. A list is returned
            unchanged.

        Returns
        -------
        list of str or None
            The sensor IDs, or None if a region name was not
            recognized.
        """
        # A string is treated as a region name
        if isinstance(sensor_ids_input, str):
            try:
                log(f"Resolving location name: '{sensor_ids_input}'")
                return self._sensor_list_generator(sensor_ids_input)
            # Log unknown region names and signal failure with None
            except ValueError as e:
                log(f"Error: {e}", level="error")
                log("Stopping data retrieval.", level="error")
                return None

        # Anything else is treated as a list of sensor IDs
        return sensor_ids_input

    def _create_date_chunks(self, chunk_days: int) -> List[Tuple[date, date]]:
        """
        Split the date range into consecutive chunks.

        Parameters
        ----------
        chunk_days : int
            Maximum number of days in each chunk.

        Returns
        -------
        list of tuple of (datetime.date, datetime.date)
            (first day, last day) of each chunk, both inclusive. The
            last chunk ends on `max_date` and may be shorter.
        """
        date_chunks = []
        # Start the first chunk on the first day of the range
        current_start_date = self.min_date

        # Keep adding chunks until the whole range is covered
        while current_start_date <= self.max_date:
            # The chunk ends chunk_days - 1 days after it starts
            current_end_date = current_start_date + timedelta(
                days=chunk_days - 1
            )
            # Do not run past the end of the range
            if current_end_date > self.max_date:
                current_end_date = self.max_date
            # Record the chunk
            date_chunks.append((current_start_date, current_end_date))
            # The next chunk starts the day after this one ends
            current_start_date += timedelta(days=chunk_days)
        return date_chunks

    # ===== API request helpers =====

    def _fetch_data(
        self,
        session: requests.Session,
        start_date: date | datetime,
        end_date: date | datetime,
        sensor_id: str,
    ) -> Optional[Dict[str, Any] | List[Any]]:
        """
        Send one request to the SDFP water level endpoint.

        Parameters
        ----------
        session : requests.Session
            Authenticated session used to send the request.
        start_date : datetime.date or datetime.datetime
            First day to request (inclusive).
        end_date : datetime.date or datetime.datetime
            Day after the last day to request (exclusive).
        sensor_id : str
            The sensor to request.

        Returns
        -------
        dict or list or None
            The parsed JSON response, or None if the request failed.
        """
        # Build the query string the API expects
        query_params = {
            "min_date": start_date.strftime("%Y-%m-%d"),
            "max_date": end_date.strftime("%Y-%m-%d"),
            "sensor_ID": sensor_id,
        }

        try:
            # Send the request
            response = session.get(
                self.BASE_URL, params=query_params, timeout=self.timeout
            )
            # Raise an HTTPError for 4xx and 5xx responses
            response.raise_for_status()
            # Parse and return the JSON body
            return response.json()

        # Log HTTP errors such as 401 (bad credentials) or 500
        except requests.exceptions.HTTPError as http_err:
            log(
                f"HTTP error for sensor {sensor_id} "
                f"({start_date} to {end_date}): {http_err}",
                level="error",
            )
        # Log responses that are not valid JSON
        except requests.exceptions.JSONDecodeError:
            log(
                "Error: Server returned OK but failed to decode JSON "
                f"for {sensor_id}.",
                level="error",
            )
        # Log connection problems and timeouts
        except requests.exceptions.RequestException as req_err:
            log(
                f"Request error for sensor {sensor_id} "
                f"({start_date} to {end_date}): {req_err}",
                level="error",
            )

        # Signal failure to the caller
        return None

    def _fetch_chunk(
        self,
        session: requests.Session,
        sensor_id: str,
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse one sensor's data for a date chunk.

        Parameters
        ----------
        session : requests.Session
            Authenticated session used to send the request.
        sensor_id : str
            The sensor to request.
        start_date : datetime.date or datetime.datetime
            First day of the chunk (inclusive).
        end_date : datetime.date or datetime.datetime
            Last day of the chunk (inclusive).

        Returns
        -------
        pandas.DataFrame or None
            The chunk's readings with UTC ``date`` and
            ``date_surveyed`` columns, or None if the request failed or
            returned no data.
        """
        # The API's max_date is exclusive, so request through the next
        # day to include end_date
        exclusive_end_date = end_date + timedelta(days=1)

        # Send the request
        json_data = self._fetch_data(
            session=session,
            start_date=start_date,
            end_date=exclusive_end_date,
            sensor_id=sensor_id,
        )

        # Stop if the request failed or returned nothing
        if not json_data:
            return None

        try:
            # Convert the JSON records to a table
            df = pd.DataFrame(json_data)
            # Treat an empty table as no data
            if df.empty:
                return None

            # Parse the reading and survey timestamps as UTC
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["date_surveyed"] = pd.to_datetime(df["date_surveyed"], utc=True)
            return df

        # Log responses that lack an expected column
        except KeyError as e:
            log(
                f"Error processing DataFrame for {sensor_id}: Missing "
                f"expected column {e}.",
                level="error",
            )
        # Log any other parsing problem
        except Exception as e:
            log(
                "An unexpected error occurred processing data for "
                f"{sensor_id}: {e}",
                level="error",
            )

        # Signal failure so the chunk is retried day by day
        return None

    def _fetch_day(
        self,
        session: requests.Session,
        sensor_id: str,
        target_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse one sensor's data for a single day.

        Parameters
        ----------
        session : requests.Session
            Authenticated session used to send the request.
        sensor_id : str
            The sensor to request.
        target_date : datetime.date
            The day to request.

        Returns
        -------
        pandas.DataFrame or None
            The day's readings with UTC ``date`` and ``date_surveyed``
            columns, or None if the request failed or returned no data.
        """
        # The API's max_date is exclusive, so request through the next
        # day to get exactly one day
        exclusive_end_date = target_date + timedelta(days=1)

        # Send the request
        json_data = self._fetch_data(
            session=session,
            start_date=target_date,
            end_date=exclusive_end_date,
            sensor_id=sensor_id,
        )

        # Stop if the request failed or returned nothing
        if not json_data:
            return None

        try:
            # Convert the JSON records to a table
            df = pd.DataFrame(json_data)
            # Treat an empty table as no data
            if df.empty:
                return None

            # Parse the reading and survey timestamps as UTC
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["date_surveyed"] = pd.to_datetime(df["date_surveyed"], utc=True)
            return df

        # Log responses that lack an expected column
        except KeyError as e:
            log(
                f"Error processing DataFrame for {sensor_id} on "
                f"{target_date}: Missing column {e}.",
                level="error",
            )
        # Log any other parsing problem
        except Exception as e:
            log(
                f"Unexpected error processing data for {sensor_id} on "
                f"{target_date}: {e}",
                level="error",
            )

        # Signal failure to the caller
        return None

    def _submit_chunk_tasks(
        self,
        executor: ThreadPoolExecutor,
        session: requests.Session,
        sensor_ids: List[str],
        date_chunks: List[Tuple[date, date]],
    ) -> Dict[Future, Tuple[str, date, date]]:
        """
        Submit one download task per (sensor, chunk) pair.

        Parameters
        ----------
        executor : concurrent.futures.ThreadPoolExecutor
            The thread pool that runs the tasks.
        session : requests.Session
            Authenticated session passed to every task.
        sensor_ids : list of str
            Sensors to download.
        date_chunks : list of tuple of (datetime.date, datetime.date)
            Date chunks to download, from :meth:`_create_date_chunks`.

        Returns
        -------
        dict
            Maps each submitted Future to its (sensor ID, chunk start,
            chunk end) so failures can be retried.
        """
        futures = {}
        # Loop over every sensor and every chunk
        for sensor_id in sensor_ids:
            for start_chunk, end_chunk in date_chunks:
                # Start the download on a worker thread
                task = executor.submit(
                    self._fetch_chunk,
                    session,
                    sensor_id,
                    start_chunk,
                    end_chunk,
                )
                # Remember which sensor and chunk this task covers
                futures[task] = (sensor_id, start_chunk, end_chunk)
        return futures

    def _process_chunk_results(
        self, futures: Dict[Future, Tuple[str, date, date]]
    ) -> Tuple[List[pd.DataFrame], Set[Tuple[str, date, date]]]:
        """
        Collect finished chunk downloads and note the failures.

        Parameters
        ----------
        futures : dict
            Futures from :meth:`_submit_chunk_tasks`, mapped to their
            (sensor ID, chunk start, chunk end).

        Returns
        -------
        all_dataframes : list of pandas.DataFrame
            The data from every chunk that succeeded.
        failed_tasks : set of tuple of (str, datetime.date, datetime.date)
            (sensor ID, chunk start, chunk end) of every chunk that
            failed and needs a daily retry.
        """
        all_dataframes = []
        failed_tasks = set()

        # Handle each task as soon as it finishes
        for future in as_completed(futures):
            # Look up which sensor and chunk this task covered
            task_info = futures[future]
            task_id = f"{task_info[0]} ({task_info[1]} to {task_info[2]})"
            try:
                # Get the task's DataFrame (or None)
                df = future.result()
                # Keep the data if the task returned some
                if df is not None:
                    all_dataframes.append(df)
                # None means the request failed or was empty: retry it
                else:
                    log(f"Task failed, queueing for 1-day fallback: {task_id}")
                    failed_tasks.add(task_info)
            # An exception inside the task also queues a retry
            except Exception as e:
                log(
                    f"Task error, queueing for 1-day fallback: "
                    f"{task_id}: {e}",
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
        Retry failed chunks by requesting each day separately.

        Parameters
        ----------
        session : requests.Session
            Authenticated session used to send the requests.
        failed_tasks : set of tuple of (str, datetime.date, datetime.date)
            (sensor ID, chunk start, chunk end) of each failed chunk.

        Returns
        -------
        list of pandas.DataFrame
            The data from every day that succeeded.
        """
        # Nothing to retry
        if not failed_tasks:
            return []

        log(
            f"Handling {len(failed_tasks)} failed chunks by fetching "
            "1-day increments..."
        )
        all_fallback_data = []

        # Expand every failed chunk into one task per day
        daily_tasks_to_run = []
        for sensor_id, start_date, end_date in failed_tasks:
            log(
                f"Queueing 1-day fallback for {sensor_id} from "
                f"{start_date} to {end_date}"
            )
            # Count the days in the chunk (end date inclusive)
            total_days = (end_date - start_date).days
            for i in range(total_days + 1):
                current_date = start_date + timedelta(days=i)
                daily_tasks_to_run.append((sensor_id, current_date))

        # Nothing to retry after expansion
        if not daily_tasks_to_run:
            return []

        log(f"Total fallback 1-day requests: {len(daily_tasks_to_run)}")

        # Run the daily requests on a pool of worker threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit one task per day, labelled for the log
            futures = {
                executor.submit(
                    self._fetch_day, session, sensor_id, target_date
                ): f"{sensor_id} on {target_date}"
                for (sensor_id, target_date) in daily_tasks_to_run
            }

            # Keep the data from each day as it finishes
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        all_fallback_data.append(df)
                # Log days that raised an error; they are skipped
                except Exception as e:
                    log(f"Error in fallback task {task_id}: {e}", "error")

        log("Fallback data retrieval complete.")
        return all_fallback_data

    def _process_and_combine_data(
        self, all_dataframes: List[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        Combine downloaded chunks into one clean DataFrame.

        Parameters
        ----------
        all_dataframes : list of pandas.DataFrame
            Every DataFrame downloaded (chunks and daily retries).

        Returns
        -------
        pandas.DataFrame or None
            The readings with duplicate (date, sensor) rows removed,
            sorted by date, or None if there was no data.
        """
        # Stop if every request failed
        if not all_dataframes:
            log(
                "No data was successfully retrieved for any sensor.",
                level="error",
            )
            return None

        try:
            # Stack all chunks into one table
            log(f"Combining {len(all_dataframes)} DataFrame chunks...")
            combined_data = pd.concat(all_dataframes, ignore_index=True)

            # Drop readings downloaded twice (e.g. overlapping retries)
            log("Dropping duplicates and sorting by date...")
            combined_data.drop_duplicates(
                subset=["date", "sensor_ID"], inplace=True
            )
            # Put the readings in time order
            combined_data.sort_values(by=["date"], inplace=True)
            # Renumber the rows 0..n-1
            combined_data.reset_index(drop=True, inplace=True)

            log(f"Final DataFrame shape: {combined_data.shape}")
            return combined_data
        # Log any failure to combine the chunks
        except Exception as e:
            log(f"Failed to concatenate DataFrames: {e}", level="error")
            return None

    # ===== Flood event detection =====

    def _reassign_abbr_flood_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign event numbers in the abbreviated flood event table.

        Events are numbered from 1 in order of start time. Events at
        different sensors that overlap in time share a number.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per flood event, with ``start_time_UTC`` and
            ``end_time_UTC`` columns.

        Returns
        -------
        pandas.DataFrame
            A copy of `df`, sorted by start time, with a renumbered
            integer ``flood_event`` column.
        """
        # Work on a copy so the caller's table is not changed
        df = df.copy()

        # Parse the event times as UTC (CSV values are strings)
        df["start_time_UTC"] = pd.to_datetime(df["start_time_UTC"], utc=True)
        df["end_time_UTC"] = pd.to_datetime(df["end_time_UTC"], utc=True)
        # Put the events in start-time order and renumber the rows
        df = df.sort_values(by=["start_time_UTC"]).reset_index(drop=True)

        # Nothing to number
        if df.empty:
            return df

        # Make sure the event number column exists
        if "flood_event" not in df.columns:
            df["flood_event"] = 0

        # Pull out the time columns for fast positional access
        start_times = df["start_time_UTC"]
        end_times = df["end_time_UTC"]

        # The first event is number 1 and sets the current end time
        current_event_number = 1
        current_end_time = end_times.iloc[0]
        df.loc[0, "flood_event"] = current_event_number

        # Walk through the remaining events in start-time order
        for i in range(1, len(df)):
            start_time = start_times.iloc[i]
            end_time = end_times.iloc[i]

            # Starts before the current group ends: same event number
            if start_time < current_end_time:
                df.loc[i, "flood_event"] = current_event_number
                # Extend the group if this event ends later
                current_end_time = max(current_end_time, end_time)
            # Starts after the current group ends: new event number
            else:
                current_event_number += 1
                df.loc[i, "flood_event"] = current_event_number
                # This event now sets the group's end time
                current_end_time = end_time

        # Store the event numbers as integers
        df["flood_event"] = df["flood_event"].astype(int)
        return df

    def _gen_abbr_flood_event_csv(
        self,
        dataframe: pd.DataFrame,
        csv_filename: str = "abbr_flood_events.csv",
    ):
        """
        Write the abbreviated (one row per event) flood event CSV.

        A flood event starts when the road water level rises above
        ``FLOOD_START_THRESHOLD_FT`` and ends at the first reading below
        ``FLOOD_END_THRESHOLD_FT``. If the CSV already exists, the new
        events are added to it, duplicates are dropped, and all events
        are renumbered.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Water level readings from the SDFP API.
        csv_filename : str, optional
            Path of the CSV to write. Default is
            ``"abbr_flood_events.csv"``.

        Returns
        -------
        None
        """
        try:
            # Load the existing CSV so new events can be added to it
            existing_data = pd.read_csv(csv_filename)
            # Continue numbering after the existing events (these
            # numbers are temporary; all events are renumbered below)
            last_event_number = (
                existing_data["flood_event"].max() + 1
                if "flood_event" in existing_data.columns
                and not existing_data.empty
                else 0
            )
        # No CSV yet: start from an empty table with the output columns
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

        # No flood is in progress at the start
        flood_start_time = None
        max_water_level = 0
        flood_events = []

        # Put the readings in time order and list the sensors
        sorted_dataframe = dataframe.sort_values(by=["date"])
        sensors = sorted_dataframe["sensor_ID"].unique().tolist()

        # Scan each sensor's readings separately
        for sensor in sensors:
            filtered_dataframe = sorted_dataframe[
                sorted_dataframe["sensor_ID"] == sensor
            ]

            for _, row in filtered_dataframe.iterrows():
                # Read the road water level (ft) and its timestamp
                water_level = row["road_water_level_adj"]
                timestamp = row["date"]

                # Rising above the start threshold begins a flood
                if (
                    water_level > FLOOD_START_THRESHOLD_FT
                    and flood_start_time is None
                ):
                    flood_start_time = timestamp
                    max_water_level = water_level

                # During a flood, track the highest water level
                if (
                    flood_start_time is not None
                    and water_level > max_water_level
                ):
                    max_water_level = water_level

                # Dropping below the end threshold ends the flood
                if (
                    water_level < FLOOD_END_THRESHOLD_FT
                    and flood_start_time is not None
                ):
                    flood_end_time = timestamp
                    # Convert both ends of the flood to EST
                    start_time_est = flood_start_time.astimezone(EST)
                    end_time_est = flood_end_time.astimezone(EST)

                    # Record one row summarizing the flood
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
                                max_water_level / FT_PER_M, 3
                            ),
                        }
                    )

                    # Reset the state to look for the next flood
                    last_event_number += 1
                    flood_start_time = None
                    max_water_level = 0

        # Nothing to write if no floods were found
        if not flood_events:
            log("No new flood events found.")
            return

        # Turn the flood summaries into a table
        flood_event_df = pd.DataFrame(flood_events)

        # First run: number the new events
        if existing_data.empty:
            updated_data = self._reassign_abbr_flood_numbers(flood_event_df)
        # Existing CSV: add the new events, drop repeats, renumber all
        else:
            merged_df = pd.concat(
                [existing_data, flood_event_df], ignore_index=True
            )
            # Compare start times as strings, since the loaded CSV
            # values are strings and the new values are Timestamps
            merged_df["start_time_UTC"] = merged_df["start_time_UTC"].astype(
                str
            )
            # Keep the first copy of each (sensor, start time) event
            filtered_df = merged_df.drop_duplicates(
                subset=["sensor_ID", "start_time_UTC"], keep="first"
            )
            updated_data = self._reassign_abbr_flood_numbers(filtered_df)

        # Save the table
        updated_data.to_csv(csv_filename, index=False)

    def _reassign_flood_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign event numbers in the detailed flood event table.

        Applies the same numbering as
        :meth:`_reassign_abbr_flood_numbers` to the detailed table, where
        each event spans many rows and only its last row has the event
        summary (start, end, duration, maximum).

        Parameters
        ----------
        df : pandas.DataFrame
            One row per reading inside a flood event, with
            ``time_UTC``, ``start_time_UTC``, ``end_time_UTC``, and
            ``duration_(hours)`` columns.

        Returns
        -------
        pandas.DataFrame
            A copy of `df` sorted by time and sensor, with renumbered
            ``flood_event`` values.
        """
        # Work on a copy so the caller's table is not changed
        df = df.copy()

        # Parse the time columns as UTC (CSV values are strings)
        df["start_time_UTC"] = pd.to_datetime(df["start_time_UTC"], utc=True)
        df["end_time_UTC"] = pd.to_datetime(df["end_time_UTC"], utc=True)
        df["time_UTC"] = pd.to_datetime(df["time_UTC"], utc=True)

        # Map each sensor ID to a number (in order of first appearance)
        # so rows can be sorted by sensor as a tie-breaker
        sensor_id_list = df["sensor_ID"].unique()
        sensor_id_mapping = {
            sensor_id: i for i, sensor_id in enumerate(sensor_id_list)
        }
        df["Sensor_ID_numeric"] = df["sensor_ID"].map(sensor_id_mapping)

        # Sort by time, then sensor, renumber the rows, and drop the
        # helper column
        df = df.sort_values(by=["time_UTC", "Sensor_ID_numeric"])
        df = df.reset_index(drop=True)
        df = df.drop(columns=["Sensor_ID_numeric"])

        # The last row of each event is the only one with a duration
        complete_rows = df[df["duration_(hours)"].notnull()]

        # Nothing to number if no event has ended
        if complete_rows.empty:
            log("No complete flood events to re-number.")
            return df

        # Start numbering at 1, using the first event's end time
        flood_event_number = 1
        # Row index of the last end row numbered for each sensor
        last_assigned_event = {}
        current_end_time = complete_rows.iloc[0]["end_time_UTC"]

        # Walk through the event end rows in time order
        for index, row in complete_rows.iterrows():
            # Starts before the current group ends: same event number
            if row["start_time_UTC"] < current_end_time:
                # First event for this sensor: number all its rows up
                # to and including this end row
                if row["sensor_ID"] not in last_assigned_event:
                    df.loc[
                        (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index
                # Later event for this sensor: number its rows after the
                # previous end row up to and including this end row
                elif row["sensor_ID"] in last_assigned_event:
                    df.loc[
                        (df.index > last_assigned_event[row["sensor_ID"]])
                        & (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index

                # Extend the group if this event ends later
                current_end_time = max(current_end_time, row["end_time_UTC"])
            # Starts after the current group ends: new event number
            else:
                flood_event_number += 1

                # Number this sensor's rows for the event (as above)
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

                # This event now sets the group's end time
                current_end_time = row["end_time_UTC"]

        return df

    def _gen_flood_tracker(
        self, dataframe: pd.DataFrame, csv_filename: str = "flood_events.csv"
    ):
        """
        Write the detailed (one row per reading) flood event CSV.

        Every reading from the start of a flood up to and including the
        reading that ends it gets a row. The ending row also holds the
        event summary (start, end, duration, maximum water level). If
        the CSV already exists, the new rows are added to it, duplicates
        are dropped, and all events are renumbered.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Water level readings from the SDFP API.
        csv_filename : str, optional
            Path of the CSV to write. Default is ``"flood_events.csv"``.

        Returns
        -------
        None
        """
        try:
            # Load the existing CSV so new rows can be added to it
            existing_data = pd.read_csv(csv_filename)
            # Continue numbering after the existing events (these
            # numbers are temporary; all events are renumbered below)
            last_event_number = (
                existing_data["flood_event"].max() + 1
                if "flood_event" in existing_data.columns
                and not existing_data.empty
                else 0
            )
        # No CSV yet: start from an empty table with the output columns
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

        # No flood is in progress at the start
        flood_start_time = None
        flood_end_time = None
        max_water_level = 0
        flood_events = []

        # Put the readings in time order and list the sensors
        sorted_dataframe = dataframe.sort_values(by=["date"])
        sensors = sorted_dataframe["sensor_ID"].unique().tolist()

        # Scan each sensor's readings separately
        for sensor in sensors:
            filtered_dataframe = sorted_dataframe[
                sorted_dataframe["sensor_ID"] == sensor
            ]

            for _, row in filtered_dataframe.iterrows():
                # Read the road and sensor water levels and the time
                water_level = row["road_water_level_adj"]
                sensor_wtr_level = row["sensor_water_level_adj"]
                timestamp = row["date"]

                # Rising above the start threshold begins a flood
                if (
                    water_level > FLOOD_START_THRESHOLD_FT
                    and flood_start_time is None
                ):
                    flood_start_time = timestamp
                    max_water_level = water_level

                # During a flood, track the highest water level
                if (
                    flood_start_time is not None
                    and water_level > max_water_level
                ):
                    max_water_level = water_level

                # Dropping below the end threshold ends the flood
                if (
                    water_level < FLOOD_END_THRESHOLD_FT
                    and flood_start_time is not None
                ):
                    flood_end_time = timestamp

                # Flood in progress: record the reading on its own
                if flood_start_time is not None and flood_end_time is None:
                    flood_events.append(
                        {
                            "flood_event": last_event_number,
                            "sensor_ID": sensor,
                            "time_UTC": timestamp,
                            "time_EST": timestamp.astimezone(EST),
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
                # Flood just ended: record the reading with the summary
                elif (
                    flood_start_time is not None and flood_end_time is not None
                ):
                    # Convert both ends of the flood to EST
                    start_time_est = flood_start_time.astimezone(EST)
                    end_time_est = flood_end_time.astimezone(EST)

                    flood_events.append(
                        {
                            "flood_event": last_event_number,
                            "sensor_ID": sensor,
                            "time_UTC": timestamp,
                            "time_EST": timestamp.astimezone(EST),
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
                                max_water_level / FT_PER_M, 3
                            ),
                        }
                    )

                    # Reset the state to look for the next flood
                    last_event_number += 1
                    flood_start_time = None
                    flood_end_time = None
                    max_water_level = 0

        # Nothing to write if no floods were found
        if not flood_events:
            log("No new detailed flood data to track.")
            return

        # Turn the recorded rows into a table
        flood_event_df = pd.DataFrame(flood_events)

        # First run: number the new events
        if existing_data.empty:
            updated_data = self._reassign_flood_numbers(flood_event_df)
        # Existing CSV: add the new rows, drop repeats, renumber all
        else:
            merged_df = pd.concat(
                [existing_data, flood_event_df], ignore_index=True
            )
            # Compare times as strings, since the loaded CSV values are
            # strings and the new values are Timestamps
            merged_df["time_UTC"] = merged_df["time_UTC"].astype(str)
            # Keep the first copy of each (sensor, time) reading
            filtered_df = merged_df.drop_duplicates(
                subset=["sensor_ID", "time_UTC"], keep="first"
            )
            updated_data = self._reassign_flood_numbers(filtered_df)

        # Save the table
        updated_data.to_csv(csv_filename, index=False)

    # ===== Sensor outages =====

    def _find_outages(
        self, dataframe: pd.DataFrame, csv_filename: str = "sensor_outages.csv"
    ) -> Optional[pd.DataFrame]:
        """
        Find gaps in each sensor's readings and write them to a CSV.

        An outage is any gap longer than ``OUTAGE_GAP`` between two
        consecutive readings from the same sensor. It runs from the last
        reading before the gap to the first reading after it.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Water level readings from the SDFP API.
        csv_filename : str, optional
            Path of the CSV to write. Default is
            ``"sensor_outages.csv"``.

        Returns
        -------
        pandas.DataFrame or None
            One row per outage, numbered from 1 in start-time order, or
            None if no outages were found.
        """
        # List the sensors in the data
        sensor_id_list = dataframe["sensor_ID"].unique().tolist()
        outages = []

        for sensor_id in sensor_id_list:
            # Select this sensor's readings in time order
            filtered_dataframe = dataframe[dataframe["sensor_ID"] == sensor_id]
            sorted_dataframe = filtered_dataframe.sort_values(
                by="date"
            ).reset_index(drop=True)

            # Record the sensor's location, used later to find daylight
            # hours when pulling outage images (0.0 if not returned)
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

            # Time between each reading and the one before it
            time_differences = sorted_dataframe["date"].diff()

            # Row positions where the gap before the reading is too long
            gaps = time_differences > OUTAGE_GAP
            indices_of_gaps = gaps[gaps].index

            for i in indices_of_gaps:
                # The outage spans from the reading before the gap to
                # the reading after it
                outage_start_time = sorted_dataframe.loc[i - 1, "date"]
                outage_end_time = sorted_dataframe.loc[i, "date"]
                # Convert both ends to EST
                start_time_est = outage_start_time.astimezone(EST)
                end_time_est = outage_end_time.astimezone(EST)

                # Record the outage (numbered after sorting below)
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
                        "latitude": lat,
                        "longitude": lon,
                    }
                )

        # Nothing to write if no outages were found
        if not outages:
            log("No sensor outages found.")
            return None

        # Put the outages in start-time order
        outage_dataframe = pd.DataFrame(outages).sort_values(
            by="start_time_UTC"
        )

        # Number the outages 1, 2, 3, ... in that order
        outage_dataframe["outage_number"] = range(1, len(outage_dataframe) + 1)

        # Save the table (this replaces any existing outage CSV)
        outage_dataframe.to_csv(csv_filename, index=False)
        return outage_dataframe

    def _check_for_outage_during_flood(
        self,
        outage_csv: str = "sensor_outages.csv",
        abbr_flood_csv: str = "abbr_flood_events.csv",
    ):
        """
        Flag outages that start during a flood event, and vice versa.

        Adds ``during_flood_event`` (``"Yes"``/``"No"``) and
        ``flood_event_number`` columns to the outage CSV, and an
        ``outage`` (``"Yes"``/``"No"``) column to the abbreviated flood
        event CSV. Both files are overwritten.

        Parameters
        ----------
        outage_csv : str, optional
            Path of the outage CSV. Default is ``"sensor_outages.csv"``.
        abbr_flood_csv : str, optional
            Path of the abbreviated flood event CSV. Default is
            ``"abbr_flood_events.csv"``.

        Returns
        -------
        None
        """
        try:
            # Load both tables
            abbr_floods = pd.read_csv(abbr_flood_csv)
            outage_dataframe = pd.read_csv(outage_csv)
        # Stop if either file does not exist
        except FileNotFoundError as e:
            log(
                "Error: Cannot check for outages, file not found: "
                f"{e.filename}",
                level="error",
            )
            return

        # Nothing to compare if either table is empty
        if abbr_floods.empty or outage_dataframe.empty:
            log("No flood or outage data to compare.")
            return

        # Parse the times needed for the comparison
        abbr_floods["start_time_UTC"] = pd.to_datetime(
            abbr_floods["start_time_UTC"]
        )
        abbr_floods["end_time_UTC"] = pd.to_datetime(
            abbr_floods["end_time_UTC"]
        )
        outage_dataframe["start_time_UTC"] = pd.to_datetime(
            outage_dataframe["start_time_UTC"]
        )

        # Start with every row flagged "No"
        outage_dataframe["during_flood_event"] = "No"
        outage_dataframe["flood_event_number"] = pd.NA
        abbr_floods["outage"] = "No"

        # Pair every outage with every flood at the same sensor
        merged = pd.merge(
            outage_dataframe,
            abbr_floods,
            on="sensor_ID",
            suffixes=("_outage", "_flood"),
        )

        # Keep the pairs where the outage starts inside the flood
        mask = (
            merged["start_time_UTC_outage"] >= merged["start_time_UTC_flood"]
        ) & (merged["start_time_UTC_outage"] <= merged["end_time_UTC_flood"])

        # Flood event numbers that contain an outage start
        flood_event_nums_to_flag = merged.loc[mask, "flood_event"].unique()

        # Map each matching outage number to its flood event number
        matching_outages = merged.loc[
            mask, ["outage_number", "flood_event"]
        ].set_index("outage_number")

        # Mark each matching outage and record its flood event number
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

        # Mark the flood events that contain an outage
        abbr_floods.loc[
            abbr_floods["flood_event"].isin(flood_event_nums_to_flag), "outage"
        ] = "Yes"

        # Save both tables
        abbr_floods.to_csv(abbr_flood_csv, index=False)
        outage_dataframe.to_csv(outage_csv, index=False)

    # ===== Plotting =====

    def _plot_and_save_flood_plots(
        self,
        sunnyd_data: pd.DataFrame,
        csv_filename: str = "abbr_flood_events.csv",
        plot_folder: str = "flood_plots",
    ):
        """
        Save a water level plot for each event in the abbreviated CSV.

        Each plot shows the road water level from one day before to one
        day after the event, with the event's readings marked. Plots
        that already exist are skipped.

        Parameters
        ----------
        sunnyd_data : pandas.DataFrame
            Water level readings from the SDFP API.
        csv_filename : str, optional
            Path of the abbreviated flood event CSV. Default is
            ``"abbr_flood_events.csv"``.
        plot_folder : str, optional
            Folder to save the plots in, as
            ``flood_event_<number>_<sensor_ID>.png``. Default is
            ``"flood_plots"``.

        Returns
        -------
        None
        """
        # Create the plot folder if needed
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
            log(f"Folder {plot_folder} created successfully.")

        try:
            # Load the list of events to plot
            abbr_dataframe = pd.read_csv(csv_filename)
        # Stop if the event CSV does not exist
        except FileNotFoundError:
            log(
                f"Error: Cannot create plots, file not found: {csv_filename}",
                level="error",
            )
            return

        # Nothing to plot
        if abbr_dataframe.empty:
            log("No flood events in CSV to plot.")
            return

        for _, row in abbr_dataframe.iterrows():
            # Read the event's sensor and number
            sensor_id = row["sensor_ID"]
            event_number = row["flood_event"]

            # Skip events that already have a plot
            plot_filename = os.path.join(
                plot_folder, f"flood_event_{event_number}_{sensor_id}.png"
            )
            if os.path.exists(plot_filename):
                log(f"Plot already exists: {plot_filename}")
                continue

            # Parse the event's start and end as UTC
            flood_start_time = pd.to_datetime(row["start_time_UTC"], utc=True)
            flood_end_time = pd.to_datetime(row["end_time_UTC"], utc=True)

            # Show one day of context on each side of the event
            plot_start_time = flood_start_time - timedelta(days=1)
            plot_end_time = flood_end_time + timedelta(days=1)

            # Select this sensor's readings inside the plot window
            filtered_df = sunnyd_data[
                (sunnyd_data["date"] >= plot_start_time)
                & (sunnyd_data["date"] <= plot_end_time)
                & (sunnyd_data["sensor_ID"] == sensor_id)
            ]

            # Skip events with no readings in the window
            if filtered_df.empty:
                log(
                    f"No raw data found for event {event_number} "
                    f"({sensor_id}). Skipping plot.",
                    level="error",
                )
                continue

            # Start a new figure
            plt.figure(figsize=(12, 10))

            # Draw the road water level over time
            plt.plot(
                filtered_df["date"],
                filtered_df["road_water_level_adj"],
                label="Water Level",
                linestyle="-",
                color="#427e93",
            )

            # Draw the road surface as a line at 0 ft
            plt.plot(
                filtered_df["date"],
                np.zeros_like(filtered_df["date"]),
                label="Roadway",
                linestyle="-",
                linewidth=2,
                color="black",
            )

            # Find the readings that fall inside the flood event
            flood_indices = (filtered_df["date"] >= flood_start_time) & (
                filtered_df["date"] <= flood_end_time
            )

            # Mark the flood event readings with red x's
            plt.plot(
                filtered_df.loc[flood_indices, "date"],
                filtered_df.loc[flood_indices, "road_water_level_adj"],
                marker="x",
                markersize=5,
                linestyle="None",
                color="#cc0000",
                label="Flood Event Data Points",
            )

            # Fix the y-axis from -2.5 to 2 ft with 0.25 ft ticks
            plt.ylim(-2.5, 2)
            plt.xticks(rotation=45)
            plt.yticks(np.arange(-2.5, 2.25, 0.25))
            plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            # Draw horizontal grid lines at every tick
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
            # Label the plot
            plt.xlabel("Date")
            plt.ylabel("Water Level (ft above road)")
            plt.title(f"Flood Event {event_number} for {sensor_id}")
            plt.legend(loc="upper right")
            plt.tight_layout()

            # Save the plot and free its memory
            plt.savefig(plot_filename)
            plt.close()

    # ===== Static analysis helpers =====

    @staticmethod
    def num_of_flood_days_by_start(csv_name="abbr_flood_events.csv"):
        """
        Count the unique UTC dates on which flood events started.

        Parameters
        ----------
        csv_name : str, optional
            Path of an abbreviated flood event CSV. Default is
            ``"abbr_flood_events.csv"``.

        Returns
        -------
        int or None
            The number of unique start dates, or None if the file was
            not found.
        """
        try:
            # Load the flood events
            read_df = pd.read_csv(csv_name)
        # Stop if the file does not exist
        except FileNotFoundError:
            log(f"Error: File '{csv_name}' not found.", level="error")
            return

        # Parse the start times
        read_df["start_time_UTC"] = pd.to_datetime(read_df["start_time_UTC"])

        # Keep only the date part of each start time
        read_df["date"] = read_df["start_time_UTC"].dt.date

        # Count the distinct dates
        num_days = read_df["date"].nunique()

        log(f"Number of days of flooding: {num_days}")
        return num_days

    @staticmethod
    def num_of_flood_days(timezone="EST", csv_name="abbr_flood_events.csv"):
        """
        Count the unique calendar days touched by any flood event.

        Every day from an event's start date through its end date is
        counted, so multi-day events count every day they span.

        Parameters
        ----------
        timezone : {"EST", "UTC"}, optional
            Time zone whose calendar days are counted. Default is
            ``"EST"``.
        csv_name : str, optional
            Path of an abbreviated flood event CSV. Default is
            ``"abbr_flood_events.csv"``.

        Returns
        -------
        int or None
            The number of unique flood days, or None if the file was not
            found or `timezone` is not recognized.
        """
        try:
            # Load the flood events
            read_df = pd.read_csv(csv_name)
        # Stop if the file does not exist
        except FileNotFoundError:
            log(f"Error: File '{csv_name}' not found.", level="error")
            return

        # Use the UTC columns for UTC calendar days
        if timezone == "UTC":
            read_df["start_time"] = pd.to_datetime(read_df["start_time_UTC"])
            read_df["end_time"] = pd.to_datetime(read_df["end_time_UTC"])

        # Use the EST columns for EST calendar days
        elif timezone == "EST":
            read_df["start_time"] = pd.to_datetime(read_df["start_time_EST"])
            read_df["end_time"] = pd.to_datetime(read_df["end_time_EST"])

        # Reject any other time zone
        else:
            log(
                f"Error: Timezone '{timezone}' not recognized. "
                "Use 'EST' or 'UTC'.",
                level="error",
            )
            return

        # Keep only the date part of each start and end time
        read_df["start_date"] = read_df["start_time"].dt.date
        read_df["end_date"] = read_df["end_time"].dt.date

        # List every day each event spans (start and end inclusive)
        all_dates = []
        for _, row in read_df.iterrows():
            date_range = pd.date_range(
                start=row["start_date"], end=row["end_date"]
            )
            all_dates.extend(date_range)

        # Count the distinct days across all events
        all_dates = pd.Series(all_dates, name="combined_date")
        num_days = all_dates.nunique()

        log(f"Number of days of flooding: {num_days}")
        return num_days

    @staticmethod
    def avg_errors(
        error_csv, save_file_as=None, csv_filename="abbr_flood_events.csv"
    ):
        """
        Add 7-day average sensor error metrics to each flood event.

        For each event, averages the ``Mean``, ``5th Pct``, and
        ``95th Pct`` columns of the error CSV over the 7 days before the
        event's end time, and adds them as ``Mean of Mean``,
        ``Mean of 5th Pct``, and ``Mean of 95th Pct`` columns.

        Parameters
        ----------
        error_csv : str
            Path of the sensor's error CSV, with ``Date``, ``Mean``,
            ``5th Pct``, and ``95th Pct`` columns.
        save_file_as : str, optional
            Path to save the result to. If None (the default),
            `csv_filename` is overwritten.
        csv_filename : str, optional
            Path of the abbreviated flood event CSV to read. Default is
            ``"abbr_flood_events.csv"``.

        Returns
        -------
        bool or None
            True if the file was saved, or None if an input file was
            not found.
        """
        try:
            # Load the flood events and the error data
            flood_df = pd.read_csv(csv_filename)
            other_df = pd.read_csv(error_csv)
        # Stop if either file does not exist
        except FileNotFoundError as e:
            log(f"Error: File not found. {e}", level="error")
            return

        # Parse the event end times as UTC
        flood_df["end_time_UTC"] = pd.to_datetime(
            flood_df["end_time_UTC"], utc=True
        )

        # Parse the error dates as UTC, allowing mixed formats (e.g.
        # month abbreviations)
        other_df["Date"] = pd.to_datetime(
            other_df["Date"], format="mixed", utc=True
        )

        # Average the error metrics for each event
        averages = []
        for _, row in flood_df.iterrows():
            # Use the 7 days up to the event's end time
            start_time = row["end_time_UTC"] - pd.Timedelta(days=7)
            end_time = row["end_time_UTC"]

            # Select the error rows in that window
            selected_rows = other_df[
                (other_df["Date"] >= start_time)
                & (other_df["Date"] <= end_time)
            ]

            # Average each metric, rounded to 4 decimal places
            avg_values = (
                selected_rows[["Mean", "5th Pct", "95th Pct"]].mean().round(4)
            )
            averages.append(avg_values.values)

        # Turn the averages into a table with the output column names
        averages_df = pd.DataFrame(
            averages,
            columns=["Mean of Mean", "Mean of 5th Pct", "Mean of 95th Pct"],
        )

        # Add the averages as new columns beside the events
        flood_df_with_averages = pd.concat([flood_df, averages_df], axis=1)

        # Overwrite the input CSV unless another path was given
        if save_file_as is None:
            save_file = csv_filename
        else:
            save_file = save_file_as

        # Save the result
        flood_df_with_averages.to_csv(save_file, index=False)

        log(f"File with averages saved successfully to '{save_file}'.")
        return True
