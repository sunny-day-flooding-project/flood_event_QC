"""
Flood Event QC Dashboard (Streamlit app).

Pulls water level data and camera images for SDFP sensors, then steps
the reviewer through each detected flood event and sensor outage to
record QC decisions. Decisions are saved to ``data/qc_results.csv`` as
they are made.

Run from the repository root::

    streamlit run app.py

Functions
---------
load_qc_state
    Join the detected flood events with any saved QC decisions.
get_images_for_event
    List the local camera images taken during a time window.
save_qc_record
    Save (or replace) the QC decision for one event.
build_fast_carousel
    Build the HTML/JavaScript image carousel.
generate_calendar_plot
    Draw a calendar PNG of the approved flood days in a year.
run_data_pull
    Download data, detect events, and copy images.
render_sidebar
    Draw the sidebar (mode, data pull form, calendar button).
render_event_review
    Draw the Standard QC or Review Flagged Events workflow.
save_outage_review
    Save the reviewer's decision for one outage.
render_outage_review
    Draw the Review Outages workflow.
render_calendar_preview
    Show the most recently generated flood calendar.
main
    Draw the whole dashboard.
"""

import calendar
import glob
import os
import sys
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import pytz
import streamlit as st
import streamlit.components.v1 as components

from flood_qc import EventTracker, ImageHandler

# ===== Paths =====

# Folder that contains this app (the repository root)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Working folder for the pulled data and QC results
WORK_DIR = os.path.join(APP_DIR, "data")
# Flood event plots and the flood calendar
PLOTS_DIR = os.path.join(WORK_DIR, "flood_plots")
# One row per detected flood event
ABBR_CSV = os.path.join(WORK_DIR, "abbr_flood_events.csv")
# The reviewer's QC decisions
QC_CSV = os.path.join(WORK_DIR, "qc_results.csv")
# One row per detected sensor outage
OUTAGE_CSV = os.path.join(WORK_DIR, "sensor_outages.csv")
# Outage numbers the reviewer has finished
REVIEWED_OUTAGES_CSV = os.path.join(WORK_DIR, "outages_reviewed.csv")
# Sidebar logo
LOGO_PATH = os.path.join(APP_DIR, "assets", "logo.png")

# Streamlit serves files in the "static" folder next to the app at
# /app/static/ (enabled in .streamlit/config.toml)
STATIC_DIR = os.path.join(APP_DIR, "static")
# Local copies of the camera images shown in the carousel
IMAGES_DIR = os.path.join(STATIC_DIR, "event_images")
# URL that the browser uses to load those images
IMAGES_URL = "/app/static/event_images/"

# ===== QC options =====

# Choices for the event status radio buttons
STATUS_OPTIONS = ["Approved (Flooded)", "Rejected (Not Flooded)"]
# Choices for the visual confirmation radio buttons
VISUAL_OPTIONS = ["Yes", "No"]
# QC columns saved in qc_results.csv and joined onto the events
QC_COLUMNS = [
    "qc_status",
    "visual_confirmation",
    "flagged",
    "king_tide",
    "notes",
]

# Fixed Eastern Standard Time (UTC-5), matching the flood event CSVs
EST = pytz.timezone("EST")

# Format of the UTC timestamp embedded in image file names
IMAGE_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"


# ===== Helpers =====


class StreamlitLogWrapper:
    """
    File-like object that shows printed text in a Streamlit box.

    Assigned to ``sys.stdout`` and ``sys.stderr`` during a data pull so
    the library's log messages appear live in the app.

    Attributes
    ----------
    placeholder : streamlit.delta_generator.DeltaGenerator
        The empty Streamlit element the log is drawn in.
    buffer : list of str
        Every line written so far.
    """

    def __init__(self, placeholder):
        """
        Initialize the wrapper.

        Parameters
        ----------
        placeholder : streamlit.delta_generator.DeltaGenerator
            The empty Streamlit element to draw the log in.
        """
        # Element to draw in, and the lines written so far
        self.placeholder = placeholder
        self.buffer = []

    def write(self, data):
        """
        Add text to the log and redraw the last 25 lines.

        Parameters
        ----------
        data : str
            Text written by ``print``.

        Returns
        -------
        None
        """
        # print() writes the newline separately; ignore it
        if data != "\n":
            # Store the line without surrounding whitespace
            self.buffer.append(data.strip())
            # Show only the most recent 25 lines
            display_text = "\n".join(self.buffer[-25:])
            self.placeholder.code(display_text, language="bash")

    def flush(self):
        """
        Do nothing; required for file-like objects.

        Returns
        -------
        None
        """
        pass


def load_qc_state(events_csv, qc_csv):
    """
    Join the detected flood events with any saved QC decisions.

    Parameters
    ----------
    events_csv : str
        Path of the abbreviated flood event CSV.
    qc_csv : str
        Path of the QC results CSV.

    Returns
    -------
    pandas.DataFrame or None
        One row per event with the ``QC_COLUMNS`` added (empty for
        events not reviewed yet), or None if `events_csv` does not
        exist.
    """
    # No events yet: the data has not been pulled
    if not os.path.exists(events_csv):
        return None

    # Load the detected events
    events_df = pd.read_csv(events_csv)

    # Join on the saved decisions, matched by event number and sensor
    if os.path.exists(qc_csv):
        qc_df = pd.read_csv(qc_csv)
        qc_columns = qc_df[["flood_event", "sensor_ID"] + QC_COLUMNS]
        merged = pd.merge(
            events_df, qc_columns, on=["flood_event", "sensor_ID"], how="left"
        )
    # No decisions yet: add empty QC columns
    else:
        merged = events_df.copy()
        merged["qc_status"] = None
        merged["visual_confirmation"] = None
        merged["flagged"] = False
        merged["king_tide"] = False
        merged["notes"] = ""

    return merged


def get_images_for_event(sensor_id, start_utc, end_utc, images_dir):
    """
    List the local camera images taken during a time window.

    Parameters
    ----------
    sensor_id : str
        The sensor ID (e.g. ``"CB_01"``).
    start_utc : str or pandas.Timestamp
        Start of the window, in UTC.
    end_utc : str or pandas.Timestamp
        End of the window, in UTC.
    images_dir : str
        Folder of local camera images.

    Returns
    -------
    list of str
        Paths of the camera's images taken inside the window
        (inclusive), sorted by name (and so by time).
    """
    # Images are named by camera ID, e.g. CAM_CB_01_...
    camera_id = "CAM_" + sensor_id
    all_images = glob.glob(os.path.join(images_dir, f"{camera_id}_*.jpg"))

    event_images = []
    # Parse the window ends
    start_dt = pd.to_datetime(start_utc)
    end_dt = pd.to_datetime(end_utc)

    for img_path in all_images:
        # Read the UTC timestamp from the file name
        filename = os.path.basename(img_path)
        timestamp_str = ImageHandler._extract_timestamp(filename)
        if timestamp_str:
            img_dt = pd.to_datetime(
                timestamp_str, format=IMAGE_TIMESTAMP_FORMAT, utc=True
            )
            # Keep the image if it falls inside the window
            if start_dt <= img_dt <= end_dt:
                event_images.append(img_path)

    return sorted(event_images)


def save_qc_record(
    event_series, status, vis_conf, is_flagged, is_king_tide, notes
):
    """
    Save (or replace) the QC decision for one event.

    Parameters
    ----------
    event_series : pandas.Series
        The event's row, including ``flood_event`` and ``sensor_ID``.
    status : str
        One of ``STATUS_OPTIONS``.
    vis_conf : str
        One of ``VISUAL_OPTIONS``.
    is_flagged : bool
        Whether the event is flagged for secondary review.
    is_king_tide : bool
        Whether the event happened during a king tide.
    notes : str
        Free-text reviewer notes.

    Returns
    -------
    None
    """
    # Start from the event's own columns
    record = event_series.to_dict()

    # Add the decision and the time it was made
    record["qc_status"] = status
    record["visual_confirmation"] = vis_conf
    record["flagged"] = is_flagged
    record["king_tide"] = is_king_tide
    record["notes"] = notes
    record["qc_timestamp"] = datetime.now().isoformat()

    new_record = pd.DataFrame([record])

    if os.path.exists(QC_CSV):
        # Drop any earlier decision for the same event and sensor
        qc_df = pd.read_csv(QC_CSV)
        event_num = record["flood_event"]
        sensor_id = record["sensor_ID"]
        qc_df = qc_df[
            ~(
                (qc_df["flood_event"] == event_num)
                & (qc_df["sensor_ID"] == sensor_id)
            )
        ]
        # Add the new decision
        qc_df = pd.concat([qc_df, new_record], ignore_index=True)
    # First decision: start the file
    else:
        qc_df = new_record

    # Save the decisions
    qc_df.to_csv(QC_CSV, index=False)


def build_fast_carousel(image_paths):
    """
    Build the HTML/JavaScript image carousel.

    The carousel switches images in the browser without rerunning the
    app, and has a full-screen button.

    Parameters
    ----------
    image_paths : list of str
        Paths of images in ``IMAGES_DIR`` to show, in order.

    Returns
    -------
    str
        HTML to pass to ``streamlit.components.v1.html``.
    """
    # Message shown when there is nothing to display
    if not image_paths:
        return (
            "<p style='font-family: sans-serif;'>"
            "No images found for this event.</p>"
        )

    # The browser loads images by file name from IMAGES_URL
    file_names = [os.path.basename(path) for path in image_paths]

    # Buttons, counter, caption, image, and full-screen button, followed
    # by the script that switches images (doubled braces are literal
    # braces in this f-string)
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center;
                font-family: sans-serif; width: 100%;">
        <div style="margin-bottom: 10px; display: flex; gap: 20px;
                    align-items: center; justify-content: center;
                    width: 100%;">
            <button onclick="prevImg()" style="padding: 0.5rem 1rem;
                border: 1px solid #ccc; border-radius: 0.5rem;
                background: white; cursor: pointer; flex: 1;">
                &#8592; Previous</button>
            <span id="counter" style="font-weight: bold; color: #1C2833;
                white-space: nowrap;">1 of {len(file_names)}</span>
            <button onclick="nextImg()" style="padding: 0.5rem 1rem;
                border: 1px solid #ccc; border-radius: 0.5rem;
                background: white; cursor: pointer; flex: 1;">
                Next &#8594;</button>
        </div>
        <div id="caption" style="margin-bottom: 15px; font-size: 0.9rem;
            color: gray;">{file_names[0]}</div>
        <div style="position: relative; display: inline-block; width: 100%;
                    text-align: center;">
            <img id="carousel-img" src="{IMAGES_URL}{file_names[0]}"
                style="max-height: 750px; max-width: 100%;
                border-radius: 0.5rem; object-fit: contain;">
            <button onclick="toggleFullScreen()" title="Full Screen"
                style="position: absolute; top: 10px; right: 10px;
                background: rgba(255, 255, 255, 0.7); border: 1px solid #ccc;
                border-radius: 4px; padding: 6px 10px; cursor: pointer;
                font-size: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: background 0.2s;">
                &#9974;
            </button>
        </div>
    </div>

    <script>
        // Image file names and the URL folder they are served from
        const names = {str(file_names)};
        const basePath = "{IMAGES_URL}";
        let currentIndex = 0;

        const imgElement = document.getElementById("carousel-img");
        const counterElement = document.getElementById("counter");
        const captionElement = document.getElementById("caption");

        // Show the image at currentIndex and update the labels
        function updateImage() {{
            imgElement.src = basePath + names[currentIndex];
            counterElement.innerText =
                (currentIndex + 1) + " of " + names.length;
            captionElement.innerText = names[currentIndex];
        }}

        // Step back one image (stops at the first)
        function prevImg() {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateImage();
            }}
        }}

        // Step forward one image (stops at the last)
        function nextImg() {{
            if (currentIndex < names.length - 1) {{
                currentIndex++;
                updateImage();
            }}
        }}

        // Enter or leave full-screen view of the image
        function toggleFullScreen() {{
            if (!document.fullscreenElement) {{
                if (imgElement.requestFullscreen) {{
                    imgElement.requestFullscreen();
                }} else if (imgElement.webkitRequestFullscreen) {{
                    imgElement.webkitRequestFullscreen();
                }} else if (imgElement.msRequestFullscreen) {{
                    imgElement.msRequestFullscreen();
                }}
            }} else {{
                if (document.exitFullscreen) {{
                    document.exitFullscreen();
                }}
            }}
        }}
    </script>
    """
    return html_code


def generate_calendar_plot(qc_csv_path, year, output_dir):
    """
    Draw a calendar PNG of the approved flood days in a year.

    Every EST calendar day from the start date through the end date of
    each ``Approved (Flooded)`` event is highlighted.

    Parameters
    ----------
    qc_csv_path : str
        Path of the QC results CSV.
    year : int
        Calendar year to draw.
    output_dir : str
        Folder to save ``flood_calendar_<year>.png`` in.

    Returns
    -------
    str
        Path of the saved PNG.
    """
    # Dates to highlight
    highlight_dates = set()

    if os.path.exists(qc_csv_path):
        df = pd.read_csv(qc_csv_path)
        if "qc_status" in df.columns:
            # Keep only the approved events
            df["qc_status"] = df["qc_status"].astype(str).str.strip()
            df = df[df["qc_status"] == "Approved (Flooded)"]

            if "start_time_EST" in df.columns and "end_time_EST" in df.columns:
                # Take the EST date of each event's start and end
                df["start_date"] = pd.to_datetime(df["start_time_EST"]).dt.date
                df["end_date"] = pd.to_datetime(df["end_time_EST"]).dt.date

                for _, row in df.iterrows():
                    start = row["start_date"]
                    end = row["end_date"]

                    # Highlight every day from start through end
                    if pd.notna(start) and pd.notna(end):
                        delta = end - start
                        for i in range(delta.days + 1):
                            highlight_dates.add(start + timedelta(days=i))

    # One panel per month in a 3 x 4 grid
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    days_of_week = ["S", "M", "T", "W", "T", "F", "S"]
    # Weeks start on Sunday (6) to match the headers
    sunday_first_calendar = calendar.Calendar(firstweekday=6)

    for i, ax in enumerate(axes.flatten()):
        month = i + 1
        # Day numbers laid out by week (0 = no day)
        cal_matrix = sunday_first_calendar.monthdayscalendar(year, month)

        # Hide the ticks and set a fixed drawing area
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.10, 1.35)

        # Black bar with the month name
        ax.axhspan(1.15, 1.35, facecolor="black", zorder=0)
        ax.text(
            0.5,
            1.25,
            calendar.month_name[month].upper(),
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="white",
            zorder=1,
        )

        # Grey bar with the day-of-week letters
        ax.axhspan(0.95, 1.15, facecolor="#E0E0E0", zorder=0)
        for col_idx, day_name in enumerate(days_of_week):
            ax.text(
                col_idx / 6,
                1.05,
                day_name,
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="black",
                zorder=1,
            )

        # Day numbers, one week per row
        for row_idx, week in enumerate(cal_matrix):
            for col_idx, day_num in enumerate(week):
                # Skip the blank cells before the 1st and after the end
                if day_num == 0:
                    continue

                # Center of this day's cell
                x = col_idx / 6
                y = 0.82 - (row_idx * 0.16)
                current_date = date(year, month, day_num)

                if current_date in highlight_dates:
                    # Purple box behind flood days
                    box_width = (1 / 6) + 0.01
                    box_height = 0.16 + 0.01

                    box = plt.Rectangle(
                        (x - (box_width / 2), y - (box_height / 2)),
                        box_width,
                        box_height,
                        fill=True,
                        facecolor="#E6D8F8",
                        alpha=1.0,
                        edgecolor="none",
                        zorder=1,
                    )
                    ax.add_patch(box)
                    # Bold day number on top of the box
                    ax.text(
                        x,
                        y,
                        str(day_num),
                        ha="center",
                        va="center",
                        fontsize=11,
                        weight="bold",
                        color="black",
                        zorder=2,
                    )
                else:
                    # Plain grey day number
                    ax.text(
                        x,
                        y,
                        str(day_num),
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="#444444",
                    )

    # Tighten the spacing between months
    plt.subplots_adjust(wspace=0.05, hspace=0.10)

    # Save the calendar and free its memory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"flood_calendar_{year}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# ===== Sidebar =====


def run_data_pull(
    log_container,
    username,
    password,
    final_location,
    start_date,
    end_date,
    img_drive_path,
    error_csv_path,
    filter_outage_daylight,
):
    """
    Download data, detect events, and copy images.

    Log messages are shown live in `log_container` while this runs.

    Parameters
    ----------
    log_container : streamlit.delta_generator.DeltaGenerator
        Empty Streamlit element to show the process log in.
    username : str
        SDFP API username.
    password : str
        SDFP API password.
    final_location : str or list of str
        Region name or list of sensor IDs.
    start_date : datetime.date
        First day to pull (inclusive).
    end_date : datetime.date
        Last day to pull (inclusive).
    img_drive_path : str
        Root folder of the camera image archive.
    error_csv_path : str
        Path of the sensor's error CSV, or an empty string to skip the
        error metrics.
    filter_outage_daylight : bool
        If True, only copy outage images taken in daylight.

    Returns
    -------
    None
    """
    log_container.markdown("### Process Log")

    # Send printed output to the log box while the pull runs
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    st_logger = StreamlitLogWrapper(log_container)
    sys.stdout = st_logger
    sys.stderr = st_logger

    try:
        # Download the data and write the event, outage, and plot files
        tracker = EventTracker(
            authorization=(username, password),
            min_date=start_date,
            max_date=end_date,
        )
        tracker.pull_data_gen_csvs_and_plots(
            location=final_location,
            output_dir=WORK_DIR,
            full_csv_name="flood_events.csv",
            abbr_csv_name="abbr_flood_events.csv",
            outage_csv_name="sensor_outages.csv",
            plot_folder="flood_plots",
        )

        # Add the 7-day error metrics if an error CSV was given
        if error_csv_path:
            if os.path.exists(error_csv_path):
                print(f"Calculating average errors using {error_csv_path}...")
                EventTracker.avg_errors(
                    error_csv=error_csv_path, csv_filename=ABBR_CSV
                )
            else:
                print(
                    f"Warning: Error CSV not found at {error_csv_path}. "
                    "Skipping metrics."
                )

        # Copy the images for each flood event (with a 1-hour buffer)
        handler = ImageHandler(sdfp_image_drive=img_drive_path)
        handler.pull_flood_event_images(
            destination_folder=IMAGES_DIR,
            flood_event_csv_path=ABBR_CSV,
            time_buffer_hours=1,
        )

        # Copy the images for each outage
        handler.pull_outage_images(
            destination_folder=IMAGES_DIR,
            outage_csv_path=OUTAGE_CSV,
            only_daylight=filter_outage_daylight,
        )

    # Always restore normal printing, even if the pull fails
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def render_sidebar(log_container):
    """
    Draw the sidebar (mode, data pull form, calendar button).

    Parameters
    ----------
    log_container : streamlit.delta_generator.DeltaGenerator
        Empty Streamlit element in the main area for the pull log.

    Returns
    -------
    workflow_mode : str
        The selected workflow mode.
    cal_year : int
        The selected calendar year.
    """
    with st.sidebar:
        # Show the logo at the top
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width="stretch")

        st.divider()

        # ----- Workflow mode -----
        st.header("Workflow Mode")
        workflow_mode = st.radio(
            "Select Mode:",
            ["Standard QC", "Review Flagged Events", "Review Outages"],
        )

        st.divider()

        # ----- Data pull form -----
        st.header("Pull New Data")
        input_type = st.radio(
            "Query by:", ["Region Name", "Specific Sensor IDs"]
        )

        # Group the fields so the app only reruns on submit
        with st.form("pull_form"):
            username = st.text_input("API Username")
            password = st.text_input("API Password", type="password")

            # Ask for a region or for sensor IDs, depending on the choice
            if input_type == "Region Name":
                location_input = st.text_input(
                    "Region Name (e.g., Carolina Beach)"
                )
            else:
                location_input = st.text_input(
                    "Sensor IDs (comma-separated, e.g., CB_01, CB_02)"
                )

            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            img_drive_path = st.text_input("Source Image Drive Path")
            error_csv_path = st.text_input(
                "Error CSV Path (Optional, for error metrics)"
            )

            filter_outage_daylight = st.checkbox(
                "Filter outage images to daylight hours only", value=True
            )

            pull_submitted = st.form_submit_button(
                "Fetch Data & Images", width="stretch"
            )

        if pull_submitted:
            # Require credentials, a location, and the image path
            if (
                not username
                or not password
                or not img_drive_path
                or not location_input
            ):
                st.error(
                    "Please provide credentials, a location/sensor, and "
                    "paths."
                )
            else:
                # A region name is passed as text, sensor IDs as a list
                if input_type == "Region Name":
                    final_location = location_input.strip()
                else:
                    final_location = [
                        s.strip()
                        for s in location_input.split(",")
                        if s.strip()
                    ]

                # Run the pull, then rerun the app to show the results
                run_data_pull(
                    log_container,
                    username,
                    password,
                    final_location,
                    start_date,
                    end_date,
                    img_drive_path,
                    error_csv_path,
                    filter_outage_daylight,
                )
                st.success("Data pull complete! Refreshing...")
                st.rerun()

        st.divider()

        # ----- Flood calendar -----
        st.header("Generate Calendar")
        cal_year = st.number_input(
            "Calendar Year",
            min_value=2000,
            max_value=2100,
            value=datetime.now().year,
        )

        # Draw the calendar and remember where it was saved
        if st.button("Generate Flood Calendar", width="stretch"):
            with st.spinner("Drawing calendar..."):
                cal_path = generate_calendar_plot(QC_CSV, cal_year, PLOTS_DIR)
                st.session_state.calendar_path = cal_path

    return workflow_mode, cal_year


# ===== Event review =====


def render_event_review(workflow_mode):
    """
    Draw the Standard QC or Review Flagged Events workflow.

    Shows the first event still to review: its details, plot, error
    metrics, camera images, and the QC decision form.

    Parameters
    ----------
    workflow_mode : {"Standard QC", "Review Flagged Events"}
        Standard QC shows events with no decision yet; Review Flagged
        Events shows events flagged for secondary review.

    Returns
    -------
    None
    """
    # Load the events joined with any saved decisions
    merged_df = load_qc_state(ABBR_CSV, QC_CSV)

    # No events until data has been pulled
    if merged_df is None:
        st.info("No flood events found. Please pull data using the sidebar.")
        return

    # Pick the events still to review for this mode
    if workflow_mode == "Standard QC":
        active_events = merged_df[merged_df["qc_status"].isna()].copy()
        success_message = "🎉 All events have been QC'd!"
    else:
        active_events = merged_df[merged_df["flagged"].eq(True)].copy()
        success_message = "✅ No flagged events require review!"

    total_events = len(merged_df)

    # Show progress (Standard QC) or the remaining count (flagged)
    if workflow_mode == "Standard QC":
        completed = total_events - len(active_events)
        st.progress(completed / total_events if total_events > 0 else 0)
        st.write(
            f"**Progress:** {completed} / {total_events} events reviewed."
        )
    else:
        st.warning(
            f"**Flagged Events:** {len(active_events)} remaining to review."
        )

    st.divider()

    # Everything reviewed: celebrate once per time the queue empties
    if active_events.empty:
        st.success(success_message)
        if workflow_mode == "Standard QC":
            if not st.session_state.qc_balloons_played:
                st.balloons()
                st.session_state.qc_balloons_played = True
        return

    # Events remain, so the balloons can play again when it empties
    st.session_state.qc_balloons_played = False

    # Review the first remaining event
    current_event = active_events.iloc[0]
    event_num = current_event["flood_event"]
    sensor_id = current_event["sensor_ID"]
    # Unique form key per event so the form resets between events
    unique_event_id = f"{event_num}_{sensor_id}"

    # Event summary
    st.subheader(f"Reviewing: Event {event_num} at {sensor_id}")
    st.write(
        f"**Start (EST):** {current_event['start_time_EST']} | "
        f"**End (EST):** {current_event['end_time_EST']}"
    )
    st.write(
        f"**Max Depth:** {current_event['max_road_water_level_(ft)']} ft | "
        f"**Duration:** {current_event['duration_(hours)']} hrs"
    )

    # Evidence on the left (wider), decision form on the right
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        # Water level plot
        expected_plot_path = os.path.join(
            PLOTS_DIR, f"flood_event_{event_num}_{sensor_id}.png"
        )
        if os.path.exists(expected_plot_path):
            st.image(
                expected_plot_path,
                caption="Water Level Over Time",
                width="stretch",
            )
        else:
            st.warning("Plot not found for this event.")

        # Error metrics, if they were added during the pull
        if "Mean of Mean" in current_event and not pd.isna(
            current_event.get("Mean of Mean")
        ):
            st.markdown("##### Error Metrics (7-Day Rolling Avg)")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Mean of Mean", f"{current_event['Mean of Mean']}")
            ec2.metric(
                "Mean of 5th Pct", f"{current_event['Mean of 5th Pct']}"
            )
            ec3.metric(
                "Mean of 95th Pct", f"{current_event['Mean of 95th Pct']}"
            )

        st.divider()

        # Camera images taken during the event
        st.markdown("#### Sensor Photos")
        event_images = get_images_for_event(
            sensor_id,
            current_event["start_time_UTC"],
            current_event["end_time_UTC"],
            IMAGES_DIR,
        )

        if not event_images:
            st.info("No images found for this event window.")
        else:
            carousel_html = build_fast_carousel(event_images)
            components.html(carousel_html, height=850, scrolling=True)

    with col2:
        st.markdown("#### Record QC Decision")

        def get_val(col, default):
            """
            Get a saved QC value for the event, or a default.

            Parameters
            ----------
            col : str
                Column name.
            default : object
                Value to use if the column is missing or empty.

            Returns
            -------
            object
                The saved value, or `default`.
            """
            val = current_event.get(col, default)
            return default if pd.isna(val) else val

        # Preselect any saved status (needed for flagged reviews)
        prev_status = get_val("qc_status", STATUS_OPTIONS[0])
        idx = (
            STATUS_OPTIONS.index(prev_status)
            if prev_status in STATUS_OPTIONS
            else 0
        )

        # Preselect any saved visual confirmation
        prev_vis_conf = get_val("visual_confirmation", VISUAL_OPTIONS[0])
        vis_idx = (
            VISUAL_OPTIONS.index(prev_vis_conf)
            if prev_vis_conf in VISUAL_OPTIONS
            else 0
        )

        with st.form(key=f"qc_form_{unique_event_id}"):
            # Main decision
            status = st.radio(
                "Event Status:", options=STATUS_OPTIONS, index=idx
            )
            visual_confirmation = st.radio(
                "Visual confirmation:",
                options=VISUAL_OPTIONS,
                index=vis_idx,
                horizontal=True,
            )

            # Extra flags
            st.markdown("##### Additional Criteria")
            king_tide = st.checkbox(
                "During King Tide?",
                value=bool(get_val("king_tide", False)),
            )
            flagged = st.checkbox(
                "Flag for secondary review",
                value=bool(get_val("flagged", False)),
            )

            notes = st.text_area(
                "Notes:",
                value=str(get_val("notes", "")),
                placeholder="Add context here...",
            )

            submit_qc = st.form_submit_button(
                "Save & Next", type="primary", width="stretch"
            )

            # Save the decision and rerun to show the next event
            if submit_qc:
                save_qc_record(
                    current_event,
                    status,
                    visual_confirmation,
                    flagged,
                    king_tide,
                    notes,
                )
                st.rerun()


# ===== Outage review =====


def save_outage_review(
    current_outage,
    event_images,
    flood_detected,
    start_img,
    end_img,
    flagged,
    king_tide,
    notes,
    mark_reviewed,
):
    """
    Save the reviewer's decision for one outage.

    If a flood was seen in the images, a new approved flood event is
    added to the QC results using the chosen start and end images. If
    `mark_reviewed` is True, the outage is logged as reviewed.

    Parameters
    ----------
    current_outage : pandas.Series
        The outage's row, including ``outage_number`` and
        ``sensor_ID``.
    event_images : list of str
        Paths of the outage's images (empty if there are none).
    flood_detected : {"Yes", "No"}
        Whether the reviewer saw flooding.
    start_img : str
        File name of the image where the flooding starts.
    end_img : str
        File name of the image where the flooding ends.
    flagged : bool
        Whether to flag the new event for secondary review.
    king_tide : bool
        Whether the new event happened during a king tide.
    notes : str
        Free-text reviewer notes for the new event.
    mark_reviewed : bool
        Whether to log the outage as fully reviewed.

    Returns
    -------
    None
    """
    outage_num = current_outage["outage_number"]
    sensor_id = current_outage["sensor_ID"]

    # Record a new flood event if one was seen in the images
    if flood_detected == "Yes" and event_images:
        # Read the UTC times from the chosen image file names
        start_dt_utc = pd.to_datetime(
            ImageHandler._extract_timestamp(start_img),
            format=IMAGE_TIMESTAMP_FORMAT,
            utc=True,
        )
        end_dt_utc = pd.to_datetime(
            ImageHandler._extract_timestamp(end_img),
            format=IMAGE_TIMESTAMP_FORMAT,
            utc=True,
        )
        # Length of the flood in hours
        duration_hrs = (end_dt_utc - start_dt_utc).total_seconds() / 3600.0

        # Give the event the next unused event number
        abbr_df = (
            pd.read_csv(ABBR_CSV)
            if os.path.exists(ABBR_CSV)
            else pd.DataFrame(columns=["flood_event"])
        )
        qc_df = (
            pd.read_csv(QC_CSV)
            if os.path.exists(QC_CSV)
            else pd.DataFrame(columns=["flood_event"])
        )
        max_abbr = abbr_df["flood_event"].max() if not abbr_df.empty else 0
        max_qc = qc_df["flood_event"].max() if not qc_df.empty else 0
        new_event_id = int(max(max_abbr, max_qc) + 1)

        # Build the event row (no depth: the sensor was not recording)
        new_event_series = pd.Series(
            {
                "flood_event": new_event_id,
                "sensor_ID": sensor_id,
                "start_time_UTC": start_dt_utc,
                "end_time_UTC": end_dt_utc,
                "start_time_EST": start_dt_utc.astimezone(EST),
                "end_time_EST": end_dt_utc.astimezone(EST),
                "duration_(hours)": round(duration_hrs, 3),
                "max_road_water_level_(ft)": None,
                "max_road_water_level_(m)": None,
            }
        )

        # Save it as an approved, visually confirmed flood
        save_qc_record(
            new_event_series,
            "Approved (Flooded)",
            "Yes",
            flagged,
            king_tide,
            notes,
        )

    # Log the outage as reviewed so it is not shown again
    if mark_reviewed:
        rev_df = (
            pd.read_csv(REVIEWED_OUTAGES_CSV)
            if os.path.exists(REVIEWED_OUTAGES_CSV)
            else pd.DataFrame(columns=["outage_number"])
        )
        rev_df = pd.concat(
            [rev_df, pd.DataFrame([{"outage_number": outage_num}])],
            ignore_index=True,
        )
        rev_df.to_csv(REVIEWED_OUTAGES_CSV, index=False)


def render_outage_review():
    """
    Draw the Review Outages workflow.

    Shows the first outage not yet reviewed: its details, its camera
    images, and the form for recording a flood seen in the images.

    Returns
    -------
    None
    """
    # No outages until data has been pulled
    if not os.path.exists(OUTAGE_CSV):
        st.info("No outages found. Please pull data using the sidebar.")
        return

    outages_df = pd.read_csv(OUTAGE_CSV)

    # Outage numbers already reviewed
    if os.path.exists(REVIEWED_OUTAGES_CSV):
        reviewed_outages = pd.read_csv(REVIEWED_OUTAGES_CSV)[
            "outage_number"
        ].tolist()
    else:
        reviewed_outages = []

    # Outages still to review
    pending_outages = outages_df[
        ~outages_df["outage_number"].isin(reviewed_outages)
    ].copy()

    # Show progress
    total_outages = len(outages_df)
    completed = total_outages - len(pending_outages)
    st.progress(completed / total_outages if total_outages > 0 else 0)
    st.write(f"**Progress:** {completed} / {total_outages} outages reviewed.")

    st.divider()

    # Everything reviewed: celebrate once per time the queue empties
    if pending_outages.empty:
        st.success("✅ All sensor outages have been reviewed!")
        if not st.session_state.outage_balloons_played:
            st.balloons()
            st.session_state.outage_balloons_played = True
        return

    # Outages remain, so the balloons can play again when it empties
    st.session_state.outage_balloons_played = False

    # Review the first remaining outage
    current_outage = pending_outages.iloc[0]
    outage_num = current_outage["outage_number"]
    sensor_id = current_outage["sensor_ID"]

    # Outage summary
    st.subheader(f"Reviewing Outage: {outage_num} at {sensor_id}")
    st.write(
        f"**Start (EST):** {current_outage['start_time_EST']} | "
        f"**End (EST):** {current_outage['end_time_EST']}"
    )
    st.write(f"**Duration:** {current_outage['duration_(hours)']} hrs")

    # Camera images taken during the outage
    event_images = get_images_for_event(
        sensor_id,
        current_outage["start_time_UTC"],
        current_outage["end_time_UTC"],
        IMAGES_DIR,
    )

    # Images on the left (wider), form on the right
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("#### Daylight Sensor Photos")
        if not event_images:
            st.info("No daylight images were found for this outage window.")
        else:
            carousel_html = build_fast_carousel(event_images)
            components.html(carousel_html, height=850, scrolling=True)

    with col2:
        st.markdown("#### Record Flood Discovery")

        with st.form(key=f"outage_form_{outage_num}_{sensor_id}"):
            flood_detected = st.radio(
                "Was a flood detected during this outage?",
                ["No", "Yes"],
                index=0,
            )

            # Choose the first and last images showing flooding
            st.markdown("##### Flood Times (If Yes)")
            image_names = (
                [os.path.basename(img) for img in event_images]
                if event_images
                else ["No images available"]
            )
            start_img = st.selectbox("Select Start Image:", image_names)
            end_img = st.selectbox("Select End Image:", image_names)

            # Extra flags for the new event
            st.markdown("##### Additional Criteria")
            king_tide = st.checkbox("During King Tide?", value=False)
            flagged = st.checkbox("Flag for secondary review", value=False)
            notes = st.text_area("Notes:", placeholder="Add context here...")

            # Uncheck to record more than one flood in this outage
            mark_reviewed = st.checkbox(
                "Mark entire outage as fully reviewed", value=True
            )

            submit_outage = st.form_submit_button(
                "Save & Next", type="primary", width="stretch"
            )

            # Save the decision and rerun to show the next outage
            if submit_outage:
                save_outage_review(
                    current_outage,
                    event_images,
                    flood_detected,
                    start_img,
                    end_img,
                    flagged,
                    king_tide,
                    notes,
                    mark_reviewed,
                )
                st.rerun()


# ===== Calendar preview =====


def render_calendar_preview(cal_year):
    """
    Show the most recently generated flood calendar.

    Parameters
    ----------
    cal_year : int
        Year shown in the heading.

    Returns
    -------
    None
    """
    # Only show a calendar that has been generated and still exists
    if "calendar_path" not in st.session_state or not os.path.exists(
        st.session_state.calendar_path
    ):
        return

    st.divider()
    st.markdown(f"### 📅 {cal_year} Approved Flood Calendar")

    # Calendar image on the left, download button on the right
    col_img, col_btn = st.columns([4, 1], gap="large")
    with col_img:
        st.image(st.session_state.calendar_path, width="stretch")
    with col_btn:
        with open(st.session_state.calendar_path, "rb") as file:
            st.download_button(
                label="📥 Download PNG",
                data=file,
                file_name=os.path.basename(st.session_state.calendar_path),
                mime="image/png",
                width="stretch",
            )


# ===== Page =====


def main():
    """
    Draw the whole dashboard.

    Returns
    -------
    None
    """
    # Use the full browser width
    st.set_page_config(page_title="Flood Event QC", layout="wide")

    # Make sure the working folders exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Values remembered between reruns for this browser session
    if "img_index" not in st.session_state:
        st.session_state.img_index = 0
    if "current_event_id" not in st.session_state:
        st.session_state.current_event_id = None
    if "qc_balloons_played" not in st.session_state:
        st.session_state.qc_balloons_played = False
    if "outage_balloons_played" not in st.session_state:
        st.session_state.outage_balloons_played = False

    # Page title, and an empty box for the data pull log
    st.title("Flood Event QC Dashboard")
    main_log_container = st.empty()

    # Sidebar controls
    workflow_mode, cal_year = render_sidebar(main_log_container)

    # Main area: the selected workflow
    if workflow_mode in ["Standard QC", "Review Flagged Events"]:
        render_event_review(workflow_mode)
    elif workflow_mode == "Review Outages":
        render_outage_review()

    # Calendar preview at the bottom
    render_calendar_preview(cal_year)


# Streamlit runs this file as "__main__", so this draws the page; the
# check keeps the page from drawing if the file is imported elsewhere
if __name__ == "__main__":
    main()
