import streamlit as st
import pandas as pd
import os
import glob
import sys
import base64
import streamlit.components.v1 as components
import pytz
import calendar
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

# Import your custom classes
from flood_tools import EventTracker, ImageHandler

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Flood Event QC", layout="wide")

WORK_DIR = "data"
PLOTS_DIR = os.path.join(WORK_DIR, "flood_plots")
ABBR_CSV = os.path.join(WORK_DIR, "abbr_flood_events.csv")
QC_CSV = os.path.join(WORK_DIR, "qc_results.csv")
OUTAGE_CSV = os.path.join(WORK_DIR, "sensor_outages.csv")
REVIEWED_OUTAGES_CSV = os.path.join(WORK_DIR, "outages_reviewed.csv")
LOGO_PATH = "logo.png"

# Point the images directory directly to the static folder
STATIC_DIR = "static"
IMAGES_DIR = os.path.join(STATIC_DIR, "event_images")

# Ensure all directories exist
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Initialize Session State
if "img_index" not in st.session_state:
    st.session_state.img_index = 0
if "current_event_id" not in st.session_state:
    st.session_state.current_event_id = None
if "qc_balloons_played" not in st.session_state:
    st.session_state.qc_balloons_played = False
if "outage_balloons_played" not in st.session_state:
    st.session_state.outage_balloons_played = False


# --- HELPER CLASSES & FUNCTIONS ---
class StreamlitLogWrapper:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.buffer = []

    def write(self, data):
        if data != "\n":
            self.buffer.append(data.strip())
            display_text = "\n".join(self.buffer[-25:])
            self.placeholder.code(display_text, language="bash")

    def flush(self):
        pass


def load_qc_state(events_csv, qc_csv):
    if not os.path.exists(events_csv):
        return None

    events_df = pd.read_csv(events_csv)

    if os.path.exists(qc_csv):
        qc_df = pd.read_csv(qc_csv)
        qc_columns = qc_df[
            [
                "flood_event",
                "sensor_ID",
                "qc_status",
                "visual_confirmation",
                "flagged",
                "king_tide",
                "notes",
            ]
        ]
        merged = pd.merge(
            events_df, qc_columns, on=["flood_event", "sensor_ID"], how="left"
        )
    else:
        merged = events_df.copy()
        merged["qc_status"] = None
        merged["visual_confirmation"] = None
        merged["flagged"] = False
        merged["king_tide"] = False
        merged["notes"] = ""

    return merged


def get_images_for_event(sensor_id, start_utc, end_utc, images_dir):
    camera_id = "CAM_" + sensor_id
    all_images = glob.glob(os.path.join(images_dir, f"{camera_id}_*.jpg"))

    event_images = []
    start_dt = pd.to_datetime(start_utc)
    end_dt = pd.to_datetime(end_utc)

    for img_path in all_images:
        filename = os.path.basename(img_path)
        timestamp_str = ImageHandler._extract_timestamp(filename)
        if timestamp_str:
            img_dt = pd.to_datetime(
                timestamp_str, format="%Y%m%d%H%M%S", utc=True
            )
            if start_dt <= img_dt <= end_dt:
                event_images.append(img_path)

    return sorted(event_images)


def save_qc_record(
    event_series, status, vis_conf, is_flagged, is_king_tide, notes
):
    record = event_series.to_dict()

    record["qc_status"] = status
    record["visual_confirmation"] = vis_conf
    record["flagged"] = is_flagged
    record["king_tide"] = is_king_tide
    record["notes"] = notes
    record["qc_timestamp"] = datetime.now().isoformat()

    new_record = pd.DataFrame([record])

    if os.path.exists(QC_CSV):
        qc_df = pd.read_csv(QC_CSV)
        event_num = record["flood_event"]
        sensor_id = record["sensor_ID"]
        qc_df = qc_df[
            ~(
                (qc_df["flood_event"] == event_num)
                & (qc_df["sensor_ID"] == sensor_id)
            )
        ]
        qc_df = pd.concat([qc_df, new_record], ignore_index=True)
    else:
        qc_df = new_record

    qc_df.to_csv(QC_CSV, index=False)


def build_fast_carousel(image_paths):
    if not image_paths:
        return "<p style='font-family: sans-serif;'>No images found for this event.</p>"

    file_names = [os.path.basename(path) for path in image_paths]

    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; font-family: sans-serif; width: 100%;">
        <div style="margin-bottom: 10px; display: flex; gap: 20px; align-items: center; justify-content: center; width: 100%;">
            <button onclick="prevImg()" style="padding: 0.5rem 1rem; border: 1px solid #ccc; border-radius: 0.5rem; background: white; cursor: pointer; flex: 1;">&#8592; Previous</button>
            <span id="counter" style="font-weight: bold; color: #1C2833; white-space: nowrap;">1 of {len(file_names)}</span>
            <button onclick="nextImg()" style="padding: 0.5rem 1rem; border: 1px solid #ccc; border-radius: 0.5rem; background: white; cursor: pointer; flex: 1;">Next &#8594;</button>
        </div>
        <div id="caption" style="margin-bottom: 15px; font-size: 0.9rem; color: gray;">{file_names[0]}</div>
        <div style="position: relative; display: inline-block; width: 100%; text-align: center;">
            <img id="carousel-img" src="/app/static/event_images/{file_names[0]}" style="max-height: 750px; max-width: 100%; border-radius: 0.5rem; object-fit: contain;">
            <button onclick="toggleFullScreen()" title="Full Screen" style="position: absolute; top: 10px; right: 10px; background: rgba(255, 255, 255, 0.7); border: 1px solid #ccc; border-radius: 4px; padding: 6px 10px; cursor: pointer; font-size: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: background 0.2s;">
                &#9974;
            </button>
        </div>
    </div>
    
    <script>
        const names = {str(file_names)};
        const basePath = "/app/static/event_images/";
        let currentIndex = 0;
        
        const imgElement = document.getElementById("carousel-img");
        const counterElement = document.getElementById("counter");
        const captionElement = document.getElementById("caption");
        
        function updateImage() {{
            imgElement.src = basePath + names[currentIndex];
            counterElement.innerText = (currentIndex + 1) + " of " + names.length;
            captionElement.innerText = names[currentIndex];
        }}
        
        function prevImg() {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateImage();
            }}
        }}
        
        function nextImg() {{
            if (currentIndex < names.length - 1) {{
                currentIndex++;
                updateImage();
            }}
        }}
        
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
    """Generates and saves a calendar PNG of approved flood days."""
    highlight_dates = set()

    if os.path.exists(qc_csv_path):
        df = pd.read_csv(qc_csv_path)
        if "qc_status" in df.columns:
            df["qc_status"] = df["qc_status"].astype(str).str.strip()
            df = df[df["qc_status"] == "Approved (Flooded)"]

            if "start_time_EST" in df.columns and "end_time_EST" in df.columns:
                df["start_date"] = pd.to_datetime(df["start_time_EST"]).dt.date
                df["end_date"] = pd.to_datetime(df["end_time_EST"]).dt.date

                for _, row in df.iterrows():
                    start = row["start_date"]
                    end = row["end_date"]

                    if pd.notna(start) and pd.notna(end):
                        delta = end - start
                        for i in range(delta.days + 1):
                            highlight_dates.add(start + timedelta(days=i))

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    days_of_week = ["S", "M", "T", "W", "T", "F", "S"]

    for i, ax in enumerate(axes.flatten()):
        month = i + 1
        cal_matrix = calendar.Calendar(firstweekday=6).monthdayscalendar(
            year, month
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.10, 1.35)

        # Draw Headers
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

        # Draw Dates
        for row_idx, week in enumerate(cal_matrix):
            for col_idx, day_num in enumerate(week):
                if day_num == 0:
                    continue

                x = col_idx / 6
                y = 0.82 - (row_idx * 0.16)
                current_date = date(year, month, day_num)

                if current_date in highlight_dates:
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
                    ax.text(
                        x,
                        y,
                        str(day_num),
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="#444444",
                    )

    plt.subplots_adjust(wspace=0.05, hspace=0.10)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"flood_calendar_{year}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# --- MAIN UI STRUCTURE ---
st.title("Flood Event QC Dashboard")
main_log_container = st.empty()

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width="stretch")

    st.divider()

    st.header("Workflow Mode")
    workflow_mode = st.radio(
        "Select Mode:",
        ["Standard QC", "Review Flagged Events", "Review Outages"],
    )

    st.divider()

    st.header("Pull New Data")
    input_type = st.radio("Query by:", ["Region Name", "Specific Sensor IDs"])

    with st.form("pull_form"):
        username = st.text_input("API Username")
        password = st.text_input("API Password", type="password")

        if input_type == "Region Name":
            location_input = st.text_input("Region Name (e.g., Carolina Beach)")
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
        if (
            not username
            or not password
            or not img_drive_path
            or not location_input
        ):
            st.error(
                "Please provide credentials, a location/sensor, and paths."
            )
        else:
            if input_type == "Region Name":
                final_location = location_input.strip()
            else:
                final_location = [
                    s.strip() for s in location_input.split(",") if s.strip()
                ]

            main_log_container.markdown("### Process Log")

            original_stdout = sys.stdout
            original_stderr = sys.stderr
            st_logger = StreamlitLogWrapper(main_log_container)
            sys.stdout = st_logger
            sys.stderr = st_logger

            try:
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

                if error_csv_path:
                    if os.path.exists(error_csv_path):
                        print(
                            f"Calculating average errors using {error_csv_path}..."
                        )
                        EventTracker.avg_errors(
                            error_csv=error_csv_path, csv_filename=ABBR_CSV
                        )
                    else:
                        print(
                            f"Warning: Error CSV not found at {error_csv_path}. Skipping metrics."
                        )

                handler = ImageHandler(sdfp_image_drive=img_drive_path)
                handler.pull_flood_event_images(
                    destination_folder=IMAGES_DIR,
                    flood_event_csv_path=ABBR_CSV,
                    time_buffer_hours=1,
                )

                handler.pull_outage_images(
                    destination_folder=IMAGES_DIR,
                    outage_csv_path=OUTAGE_CSV,
                    only_daylight=filter_outage_daylight,
                )

            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            st.success("Data pull complete! Refreshing...")
            st.rerun()

    st.divider()

    st.header("Generate Calendar")
    cal_year = st.number_input(
        "Calendar Year",
        min_value=2000,
        max_value=2100,
        value=datetime.now().year,
    )

    if st.button("Generate Flood Calendar", width="stretch"):
        with st.spinner("Drawing calendar..."):
            cal_path = generate_calendar_plot(QC_CSV, cal_year, PLOTS_DIR)
            st.session_state.calendar_path = cal_path

# --- MAIN AREA: QC WORKFLOW ---

if workflow_mode in ["Standard QC", "Review Flagged Events"]:
    merged_df = load_qc_state(ABBR_CSV, QC_CSV)

    if merged_df is None:
        st.info("No flood events found. Please pull data using the sidebar.")
    else:
        if workflow_mode == "Standard QC":
            active_events = merged_df[merged_df["qc_status"].isna()].copy()
            success_message = "🎉 All events have been QC'd!"
        else:
            active_events = merged_df[merged_df["flagged"] == True].copy()
            success_message = "✅ No flagged events require review!"

        total_events = len(merged_df)

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

        if active_events.empty:
            st.success(success_message)
            if workflow_mode == "Standard QC":
                if not st.session_state.qc_balloons_played:
                    st.balloons()
                    st.session_state.qc_balloons_played = True
        else:
            # Reset balloons if there are events to review so it fires again next time it hits zero
            st.session_state.qc_balloons_played = False

            current_event = active_events.iloc[0]
            event_num = current_event["flood_event"]
            sensor_id = current_event["sensor_ID"]

            unique_event_id = f"{event_num}_{sensor_id}"

            st.subheader(f"Reviewing: Event {event_num} at {sensor_id}")
            st.write(
                f"**Start (EST):** {current_event['start_time_EST']} | **End (EST):** {current_event['end_time_EST']}"
            )
            st.write(
                f"**Max Depth:** {current_event['max_road_water_level_(ft)']} ft | **Duration:** {current_event['duration_(hours)']} hrs"
            )

            col1, col2 = st.columns([2, 1], gap="large")

            with col1:
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

                if "Mean of Mean" in current_event and not pd.isna(
                    current_event.get("Mean of Mean")
                ):
                    st.markdown("##### Error Metrics (7-Day Rolling Avg)")
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.metric(
                        "Mean of Mean", f"{current_event['Mean of Mean']}"
                    )
                    ec2.metric(
                        "Mean of 5th Pct", f"{current_event['Mean of 5th Pct']}"
                    )
                    ec3.metric(
                        "Mean of 95th Pct",
                        f"{current_event['Mean of 95th Pct']}",
                    )

                st.divider()

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
                    val = current_event.get(col, default)
                    return default if pd.isna(val) else val

                prev_status = get_val("qc_status", "Approved (Flooded)")
                opts = ["Approved (Flooded)", "Rejected (Not Flooded)"]
                idx = opts.index(prev_status) if prev_status in opts else 0

                prev_vis_conf = get_val("visual_confirmation", "Yes")
                vis_opts = ["Yes", "No"]
                vis_idx = (
                    vis_opts.index(prev_vis_conf)
                    if prev_vis_conf in vis_opts
                    else 0
                )

                with st.form(key=f"qc_form_{unique_event_id}"):
                    status = st.radio("Event Status:", options=opts, index=idx)
                    visual_confirmation = st.radio(
                        "Visual confirmation:",
                        options=vis_opts,
                        index=vis_idx,
                        horizontal=True,
                    )

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

elif workflow_mode == "Review Outages":
    if not os.path.exists(OUTAGE_CSV):
        st.info("No outages found. Please pull data using the sidebar.")
    else:
        outages_df = pd.read_csv(OUTAGE_CSV)

        if os.path.exists(REVIEWED_OUTAGES_CSV):
            reviewed_outages = pd.read_csv(REVIEWED_OUTAGES_CSV)[
                "outage_number"
            ].tolist()
        else:
            reviewed_outages = []

        pending_outages = outages_df[
            ~outages_df["outage_number"].isin(reviewed_outages)
        ].copy()

        total_outages = len(outages_df)
        completed = total_outages - len(pending_outages)
        st.progress(completed / total_outages if total_outages > 0 else 0)
        st.write(
            f"**Progress:** {completed} / {total_outages} outages reviewed."
        )

        st.divider()

        if pending_outages.empty:
            st.success("✅ All sensor outages have been reviewed!")
            if not st.session_state.outage_balloons_played:
                st.balloons()
                st.session_state.outage_balloons_played = True
        else:
            # Reset balloons if there are outages to review so it fires again next time it hits zero
            st.session_state.outage_balloons_played = False

            current_outage = pending_outages.iloc[0]
            outage_num = current_outage["outage_number"]
            sensor_id = current_outage["sensor_ID"]

            st.subheader(f"Reviewing Outage: {outage_num} at {sensor_id}")
            st.write(
                f"**Start (EST):** {current_outage['start_time_EST']} | **End (EST):** {current_outage['end_time_EST']}"
            )
            st.write(f"**Duration:** {current_outage['duration_(hours)']} hrs")

            event_images = get_images_for_event(
                sensor_id,
                current_outage["start_time_UTC"],
                current_outage["end_time_UTC"],
                IMAGES_DIR,
            )

            col1, col2 = st.columns([2, 1], gap="large")

            with col1:
                st.markdown("#### Daylight Sensor Photos")
                if not event_images:
                    st.info(
                        "No daylight images were found for this outage window."
                    )
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

                    st.markdown("##### Flood Times (If Yes)")
                    image_names = (
                        [os.path.basename(img) for img in event_images]
                        if event_images
                        else ["No images available"]
                    )
                    start_img = st.selectbox("Select Start Image:", image_names)
                    end_img = st.selectbox("Select End Image:", image_names)

                    st.markdown("##### Additional Criteria")
                    king_tide = st.checkbox("During King Tide?", value=False)
                    flagged = st.checkbox(
                        "Flag for secondary review", value=False
                    )
                    notes = st.text_area(
                        "Notes:", placeholder="Add context here..."
                    )

                    mark_reviewed = st.checkbox(
                        "Mark entire outage as fully reviewed", value=True
                    )

                    submit_outage = st.form_submit_button(
                        "Save & Next", type="primary", width="stretch"
                    )

                    if submit_outage:
                        if flood_detected == "Yes" and event_images:
                            start_dt_utc = pd.to_datetime(
                                ImageHandler._extract_timestamp(start_img),
                                format="%Y%m%d%H%M%S",
                                utc=True,
                            )
                            end_dt_utc = pd.to_datetime(
                                ImageHandler._extract_timestamp(end_img),
                                format="%Y%m%d%H%M%S",
                                utc=True,
                            )
                            duration_hrs = (
                                end_dt_utc - start_dt_utc
                            ).total_seconds() / 3600.0
                            eastern = pytz.timezone("EST")

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
                            max_abbr = (
                                abbr_df["flood_event"].max()
                                if not abbr_df.empty
                                else 0
                            )
                            max_qc = (
                                qc_df["flood_event"].max()
                                if not qc_df.empty
                                else 0
                            )
                            new_event_id = int(max(max_abbr, max_qc) + 1)

                            new_event_series = pd.Series(
                                {
                                    "flood_event": new_event_id,
                                    "sensor_ID": sensor_id,
                                    "start_time_UTC": start_dt_utc,
                                    "end_time_UTC": end_dt_utc,
                                    "start_time_EST": start_dt_utc.astimezone(
                                        eastern
                                    ),
                                    "end_time_EST": end_dt_utc.astimezone(
                                        eastern
                                    ),
                                    "duration_(hours)": round(duration_hrs, 3),
                                    "max_road_water_level_(ft)": None,
                                    "max_road_water_level_(m)": None,
                                }
                            )

                            save_qc_record(
                                new_event_series,
                                "Approved (Flooded)",
                                "Yes",
                                flagged,
                                king_tide,
                                notes,
                            )

                        if mark_reviewed:
                            rev_df = (
                                pd.read_csv(REVIEWED_OUTAGES_CSV)
                                if os.path.exists(REVIEWED_OUTAGES_CSV)
                                else pd.DataFrame(columns=["outage_number"])
                            )
                            rev_df = pd.concat(
                                [
                                    rev_df,
                                    pd.DataFrame(
                                        [{"outage_number": outage_num}]
                                    ),
                                ],
                                ignore_index=True,
                            )
                            rev_df.to_csv(REVIEWED_OUTAGES_CSV, index=False)

                        st.rerun()

# --- CALENDAR PREVIEW AT BOTTOM OF MAIN WORKFLOW ---
if "calendar_path" in st.session_state and os.path.exists(
    st.session_state.calendar_path
):
    st.divider()
    st.markdown(f"### 📅 {cal_year} Approved Flood Calendar")

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