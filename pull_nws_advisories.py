"""
NWS Extreme Event Fetcher and Calendar Generator.

Fetches National Weather Service (NWS) watch/warn/advisory data from the IEM API,
filters for coastal and tropical phenomena, outputs a CSV, and generates a visual
meteorological year calendar.
"""

import os
import argparse
import calendar
from datetime import date, timedelta
from collections import defaultdict
import io

import pandas as pd
import requests
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont


def fetch_extreme_events(station: str, start_year: int) -> pd.DataFrame:
    """
    Fetches extreme weather events for a specific meteorological year.

    A meteorological year in this context is defined as May 1st of the
    start_year to April 30th of the following year.

    Parameters
    ----------
    station : str
        The 3-letter NWS Weather Forecast Office (WFO) ID (e.g., 'ILM', 'MHX').
    start_year : int
        The starting year of the meteorological cycle (e.g., 2025).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the filtered storm events.
    """
    end_year = start_year + 1
    base_url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
    )

    params = {
        "accept": "excel",
        "wfo": station,
        "sts": f"{start_year}-05-01T00:00Z",
        "ets": f"{end_year}-04-30T23:59Z",
    }

    print(
        f"Fetching NWS events for {station} (May {start_year} - Apr {end_year})..."
    )

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    try:
        df = pd.read_excel(io.BytesIO(response.content), engine="openpyxl")
    except ValueError:
        print("Error: Failed to parse Excel data from the IEM API.")
        return pd.DataFrame()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    col_map = {
        "utc_issue": "issue",
        "utc_expire": "expire",
        "phenomena": "phenom",
        "significance": "sig",
    }
    df = df.rename(columns=col_map)

    if df.empty or "issue" not in df.columns:
        print("Notice: No valid data returned for this time period.")
        return pd.DataFrame()

    # Filter for targeted coastal and extreme weather phenomena
    target_phenomena = ["CF", "HU", "TR", "SS", "GL", "HF"]
    target_sigs = ["W", "Y", "A"]

    df_filtered = df[
        (df["phenom"].isin(target_phenomena)) & (df["sig"].isin(target_sigs))
    ].copy()

    if df_filtered.empty:
        print("Notice: Data retrieved, but no Coastal/Tropical events found.")
        return df_filtered

    # Convert timestamps to timezone-aware UTC objects
    df_filtered["issue"] = pd.to_datetime(df_filtered["issue"], utc=True)
    df_filtered["expire"] = pd.to_datetime(df_filtered["expire"], utc=True)

    # Generate permanent links to the NWS text product
    def generate_vtec_url(row: pd.Series) -> str:
        if pd.isna(row["issue"]):
            return ""
        year = row["issue"].year
        wfo = f"K{row['wfo']}" if len(str(row["wfo"])) == 3 else row["wfo"]
        etn = str(int(row["eventid"])).zfill(4)
        return f"https://mesonet.agron.iastate.edu/vtec/#{year}-O-NEW-{wfo}-{row['phenom']}-{row['sig']}-{etn}"

    df_filtered["text_permalink"] = df_filtered.apply(generate_vtec_url, axis=1)
    df_filtered = df_filtered.sort_values("issue").reset_index(drop=True)

    return df_filtered


def generate_pillow_calendar(
    df: pd.DataFrame, station: str, start_year: int, output_dir: str
) -> str:
    """
    Generates a visual 12-month calendar rendering of the advisory dates.

    Parameters
    ----------
    df : pd.DataFrame
        The processed dataframe containing 'issue', 'expire', 'phenom', and 'sig'.
    station : str
        The NWS WFO ID.
    start_year : int
        The starting year of the meteorological cycle.
    output_dir : str
        Directory to save the resulting PNG image.

    Returns
    -------
    str
        The file path of the generated PNG image.
    """
    end_year = start_year + 1
    months_sequence = [(m, start_year) for m in range(5, 13)] + [
        (m, end_year) for m in range(1, 5)
    ]

    daily_events = defaultdict(set)

    # Process dataframe dates into local eastern time for the visual calendar
    if not df.empty:
        df_local = df.copy()
        df_local["issue"] = df_local["issue"].dt.tz_convert("America/New_York")
        df_local["expire"] = df_local["expire"].dt.tz_convert(
            "America/New_York"
        )

        for _, row in df_local.iterrows():
            if pd.notna(row["issue"]) and pd.notna(row["expire"]):
                start_date = row["issue"].date()
                end_date = row["expire"].date()
                event_str = f"{row['phenom']}-{row['sig']}"

                delta = end_date - start_date
                for i in range(delta.days + 1):
                    current_date = start_date + timedelta(days=i)

                    # Store event if it falls within our meteorological year
                    if (
                        current_date.year == start_year
                        and current_date.month >= 5
                    ) or (
                        current_date.year == end_year
                        and current_date.month <= 4
                    ):
                        daily_events[current_date].add(event_str)

    # Canvas dimensions and padding
    day_w, day_h = 200, 160
    month_w = day_w * 7
    month_h = 80 + 50 + day_h * 6

    margin_x, margin_y = 100, 100
    gap_x, gap_y = 75, 60
    title_height, legend_height = 150, 120

    total_w = margin_x * 2 + month_w * 4 + gap_x * 3
    total_h = (
        title_height + legend_height + month_h * 3 + gap_y * 2 + margin_y * 2
    )

    # Styling definitions
    c_bg, c_text = (248, 249, 250), (30, 30, 30)
    c_month_bg, c_month_text = (0, 0, 0), (255, 255, 255)
    c_day_hdr, c_day_text = (224, 224, 224), (20, 20, 20)
    c_border = (100, 100, 100)
    c_empty, c_warn, c_adv, c_watch = (
        (255, 255, 255),
        (242, 176, 208),
        (255, 201, 168),
        (255, 230, 172),
    )

    # Font configuration
    try:
        font_path_bold = fm.findfont(
            fm.FontProperties(family="DejaVu Sans", weight="bold")
        )
        f_title = ImageFont.truetype(font_path_bold, 80)
        f_month = ImageFont.truetype(font_path_bold, 50)
        f_day_hdr = ImageFont.truetype(font_path_bold, 30)
        f_date = ImageFont.truetype(font_path_bold, 32)
        f_event = ImageFont.truetype(font_path_bold, 24)
        f_legend = ImageFont.truetype(font_path_bold, 36)
    except Exception:
        print(
            "Warning: Standard bold fonts not found. Falling back to default."
        )
        f_title = f_month = f_day_hdr = f_date = f_event = f_legend = (
            ImageFont.load_default()
        )

    img = Image.new("RGB", (total_w, total_h), c_bg)
    draw = ImageDraw.Draw(img)

    # Helper to draw centered text
    def draw_centered_text(x, y, w, h, text, font, fill):
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (
                x + (w - (bbox[2] - bbox[0])) / 2,
                y + (h - (bbox[3] - bbox[1])) / 2 - 5,
            ),
            text,
            font=font,
            fill=fill,
        )

    # Draw Title
    draw_centered_text(
        0,
        margin_y,
        total_w,
        title_height,
        f"National Weather Service Coastal Advisories - {station.upper()}",
        f_title,
        c_text,
    )

    # Draw Legend
    leg_y = margin_y + title_height
    leg_box_w, leg_box_h, spacing = 50, 50, 400
    start_x = (total_w - (3 * leg_box_w + 3 * 250 + 2 * spacing)) / 2 + 100

    for i, (label, color) in enumerate(
        [
            ("Watch (A)", c_watch),
            ("Advisory (Y)", c_adv),
            ("Warning (W)", c_warn),
        ]
    ):
        lx = start_x + i * (leg_box_w + 250 + spacing)
        draw.rectangle(
            [lx, leg_y, lx + leg_box_w, leg_y + leg_box_h],
            fill=color,
            outline=(0, 0, 0),
            width=3,
        )
        draw.text(
            (lx + leg_box_w + 20, leg_y + 5), label, font=f_legend, fill=c_text
        )

    # Draw Grid
    days_of_week = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    for m_idx, (month, year_val) in enumerate(months_sequence):
        row, col = m_idx // 4, m_idx % 4
        x_off = margin_x + col * (month_w + gap_x)
        y_off = (
            margin_y + title_height + legend_height + row * (month_h + gap_y)
        )

        # Month Header
        draw.rectangle(
            [x_off, y_off, x_off + month_w, y_off + 80],
            fill=c_month_bg,
            outline=c_border,
            width=2,
        )
        draw_centered_text(
            x_off,
            y_off,
            month_w,
            80,
            f"{calendar.month_name[month].upper()} {year_val}",
            f_month,
            c_month_text,
        )

        # Day Headers
        for d_idx, d_name in enumerate(days_of_week):
            dx = x_off + d_idx * day_w
            dy = y_off + 80
            draw.rectangle(
                [dx, dy, dx + day_w, dy + 50],
                fill=c_day_hdr,
                outline=c_border,
                width=2,
            )
            draw_centered_text(dx, dy, day_w, 50, d_name, f_day_hdr, c_day_text)

        # Day Cells
        cal_matrix = calendar.monthcalendar(year_val, month)
        for w_idx, week in enumerate(cal_matrix):
            for d_idx, day_num in enumerate(week):
                dx, dy = x_off + d_idx * day_w, y_off + 130 + w_idx * day_h

                # Empty trailing/leading days
                if day_num == 0:
                    draw.rectangle(
                        [dx, dy, dx + day_w, dy + day_h],
                        fill=(235, 235, 235),
                        outline=c_border,
                        width=2,
                    )
                    continue

                current_date = date(year_val, month, day_num)
                events_list = sorted(list(daily_events[current_date]))

                # Determine cell color hierarchy
                fill_c = c_empty
                if events_list:
                    sigs = [e[-1] for e in events_list]
                    if "W" in sigs:
                        fill_c = c_warn
                    elif "Y" in sigs:
                        fill_c = c_adv
                    else:
                        fill_c = c_watch

                draw.rectangle(
                    [dx, dy, dx + day_w, dy + day_h],
                    fill=fill_c,
                    outline=c_border,
                    width=2,
                )

                # Draw Date Number top-right
                date_str = str(day_num)
                date_bbox = draw.textbbox((0, 0), date_str, font=f_date)
                draw.text(
                    (dx + day_w - (date_bbox[2] - date_bbox[0]) - 15, dy + 10),
                    date_str,
                    font=f_date,
                    fill=(50, 50, 50),
                )
                date_h = date_bbox[3] - date_bbox[1] + 10

                # Draw Event Codes
                if events_list:
                    lines = [
                        ", ".join(events_list[i : i + 2])
                        for i in range(0, len(events_list), 2)
                    ]
                    start_y, line_spacing = dy + date_h + 10, 30
                    for l_idx, line in enumerate(lines):
                        draw_centered_text(
                            dx,
                            start_y + l_idx * line_spacing,
                            day_w,
                            line_spacing,
                            line,
                            f_event,
                            (10, 10, 10),
                        )

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(
        output_dir,
        f"nws_met_calendar_{station.lower()}_{start_year}-{end_year}.png",
    )
    img.save(out_png, format="PNG")
    print(f"Calendar successfully generated: {out_png}")

    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch NWS extreme weather events and generate a calendar."
    )
    parser.add_argument(
        "station", type=str, help="3-letter WFO station ID (e.g., ILM, MHX)."
    )
    parser.add_argument(
        "met_year",
        type=int,
        help="Starting year for the Meteorological cycle (e.g., 2025 for May 2025 - Apr 2026).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to save the CSV and Calendar.",
    )

    args = parser.parse_args()

    # 1. Fetch the data
    events_df = fetch_extreme_events(args.station.upper(), args.met_year)

    if not events_df.empty:
        print(f"Retrieved {len(events_df)} extreme events.")

        # 2. Save the raw CSV data
        os.makedirs(args.outdir, exist_ok=True)
        csv_path = os.path.join(
            args.outdir, f"{args.station.lower()}_events_{args.met_year}.csv"
        )
        events_df.to_csv(csv_path, index=False)
        print(f"CSV data saved to: {csv_path}")

        # 3. Pass the in-memory dataframe directly to the calendar generator
        generate_pillow_calendar(
            events_df, args.station.upper(), args.met_year, args.outdir
        )
