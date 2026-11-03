"""
NWS coastal and tropical advisory fetcher and calendar.

Downloads National Weather Service (NWS) watches, warnings, and
advisories for one Weather Forecast Office (WFO) from the Iowa
Environmental Mesonet (IEM), keeps the coastal and tropical products,
and draws a calendar of them for a meteorological year (May 1 through
April 30 of the next year).

Functions
---------
fetch_nws_advisories
    Download a WFO's coastal and tropical advisories for a met year.
plot_advisory_calendar
    Draw a 12-month calendar PNG of the advisory days.
main
    Command-line entry point.

Examples
--------
Run from the repository root::

    python -m flood_qc.nws_advisories MHX 2025 --outdir ./nws_output
"""

import argparse
import calendar
import io
import os
from collections import defaultdict
from datetime import date, timedelta

import matplotlib.font_manager as fm
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

from .utils import log

# IEM service that returns NWS watch/warning/advisory records
IEM_WATCHWARN_URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
)

# NWS phenomena codes kept: Coastal Flood, Hurricane, Tropical Storm,
# Storm Surge, Gale, and Hurricane Force Wind
TARGET_PHENOMENA = ["CF", "HU", "TR", "SS", "GL", "HF"]

# NWS significance codes kept: Warning, Advisory, and Watch
TARGET_SIGNIFICANCE = ["W", "Y", "A"]


def fetch_nws_advisories(station: str, start_year: int) -> pd.DataFrame:
    """
    Download a WFO's coastal and tropical advisories for a met year.

    The meteorological year runs from May 1 of `start_year` through
    April 30 of the following year.

    Parameters
    ----------
    station : str
        Three-letter NWS WFO ID (e.g. ``"ILM"``, ``"MHX"``).
    start_year : int
        First year of the meteorological year (e.g. 2025 for May 2025
        through April 2026).

    Returns
    -------
    pandas.DataFrame
        One row per advisory, sorted by issue time, with UTC ``issue``
        and ``expire`` times, ``phenom`` and ``sig`` codes, and a
        ``text_permalink`` to the NWS product. Empty if nothing was
        found or the response could not be read.

    Raises
    ------
    requests.HTTPError
        If the IEM service returns an error status.
    """
    # The meteorological year ends in the next calendar year
    end_year = start_year + 1

    # Ask for an Excel file of the WFO's products for May 1 - Apr 30
    params = {
        "accept": "excel",
        "wfo": station,
        "sts": f"{start_year}-05-01T00:00Z",
        "ets": f"{end_year}-04-30T23:59Z",
    }

    log(
        f"Fetching NWS events for {station} "
        f"(May {start_year} - Apr {end_year})..."
    )

    # Download the file and stop on an HTTP error
    response = requests.get(IEM_WATCHWARN_URL, params=params)
    response.raise_for_status()

    try:
        # Read the Excel file from memory
        df = pd.read_excel(io.BytesIO(response.content), engine="openpyxl")
    # Return an empty table if the response is not a valid Excel file
    except ValueError:
        log("Error: Failed to parse Excel data from the IEM API.", "error")
        return pd.DataFrame()

    # Make the column names lower case with no extra spaces
    df.columns = df.columns.str.lower().str.strip()

    # Rename the IEM columns to the short names used below
    col_map = {
        "utc_issue": "issue",
        "utc_expire": "expire",
        "phenomena": "phenom",
        "significance": "sig",
    }
    df = df.rename(columns=col_map)

    # Return an empty table if there were no products
    if df.empty or "issue" not in df.columns:
        log("Notice: No valid data returned for this time period.")
        return pd.DataFrame()

    # Keep only the coastal and tropical watches/warnings/advisories
    df_filtered = df[
        (df["phenom"].isin(TARGET_PHENOMENA))
        & (df["sig"].isin(TARGET_SIGNIFICANCE))
    ].copy()

    # Stop if none of the products are coastal or tropical
    if df_filtered.empty:
        log("Notice: Data retrieved, but no Coastal/Tropical events found.")
        return df_filtered

    # Parse the issue and expire times as UTC
    df_filtered["issue"] = pd.to_datetime(df_filtered["issue"], utc=True)
    df_filtered["expire"] = pd.to_datetime(df_filtered["expire"], utc=True)

    def generate_vtec_url(row: pd.Series) -> str:
        """
        Build the IEM permalink for one advisory's NWS text product.

        Parameters
        ----------
        row : pandas.Series
            One advisory, with ``issue``, ``wfo``, ``phenom``, ``sig``,
            and ``eventid`` fields.

        Returns
        -------
        str
            The permalink URL, or an empty string if the issue time is
            missing.
        """
        # No link without an issue time
        if pd.isna(row["issue"]):
            return ""
        # The link uses the year the product was issued
        year = row["issue"].year
        # Three-letter office IDs get the "K" prefix used in VTEC codes
        wfo = f"K{row['wfo']}" if len(str(row["wfo"])) == 3 else row["wfo"]
        # Event tracking numbers are padded to four digits
        etn = str(int(row["eventid"])).zfill(4)
        return (
            "https://mesonet.agron.iastate.edu/vtec/#"
            f"{year}-O-NEW-{wfo}-{row['phenom']}-{row['sig']}-{etn}"
        )

    # Add a link to each advisory's text product
    df_filtered["text_permalink"] = df_filtered.apply(
        generate_vtec_url, axis=1
    )
    # Sort by issue time and renumber the rows
    df_filtered = df_filtered.sort_values("issue").reset_index(drop=True)

    return df_filtered


def plot_advisory_calendar(
    df: pd.DataFrame, station: str, start_year: int, output_dir: str
) -> str:
    """
    Draw a 12-month calendar PNG of the advisory days.

    Each day is colored by the most severe product active that day
    (warning, then advisory, then watch) and lists the product codes.

    Parameters
    ----------
    df : pandas.DataFrame
        Advisories from :func:`fetch_nws_advisories`, with ``issue``,
        ``expire``, ``phenom``, and ``sig`` columns.
    station : str
        The NWS WFO ID, shown in the title and file name.
    start_year : int
        First year of the meteorological year.
    output_dir : str
        Folder to save the PNG in.

    Returns
    -------
    str
        Path of the saved PNG
        (``nws_met_calendar_<station>_<start>-<end>.png``).
    """
    # The calendar runs May of start_year through April of end_year
    end_year = start_year + 1
    months_sequence = [(m, start_year) for m in range(5, 13)] + [
        (m, end_year) for m in range(1, 5)
    ]

    # Map each date to the set of product codes active on it
    daily_events = defaultdict(set)

    if not df.empty:
        # Convert times to US Eastern so days match local calendar days
        df_local = df.copy()
        df_local["issue"] = df_local["issue"].dt.tz_convert("America/New_York")
        df_local["expire"] = df_local["expire"].dt.tz_convert(
            "America/New_York"
        )

        for _, row in df_local.iterrows():
            # Skip products without both times
            if pd.notna(row["issue"]) and pd.notna(row["expire"]):
                start_date = row["issue"].date()
                end_date = row["expire"].date()
                # Code such as "CF-Y" (Coastal Flood Advisory)
                event_str = f"{row['phenom']}-{row['sig']}"

                # Add the code to every day the product was active
                delta = end_date - start_date
                for i in range(delta.days + 1):
                    current_date = start_date + timedelta(days=i)

                    # Only keep days inside the meteorological year
                    if (
                        current_date.year == start_year
                        and current_date.month >= 5
                    ) or (
                        current_date.year == end_year
                        and current_date.month <= 4
                    ):
                        daily_events[current_date].add(event_str)

    # Size of one day cell and one month block, in pixels
    day_w, day_h = 200, 160
    month_w = day_w * 7
    month_h = 80 + 50 + day_h * 6

    # Page margins, gaps between months, and title/legend heights
    margin_x, margin_y = 100, 100
    gap_x, gap_y = 75, 60
    title_height, legend_height = 150, 120

    # Total image size for a 4 x 3 grid of months
    total_w = margin_x * 2 + month_w * 4 + gap_x * 3
    total_h = (
        title_height + legend_height + month_h * 3 + gap_y * 2 + margin_y * 2
    )

    # Colors (RGB) for the background, headers, borders, and cells
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

    try:
        # Use bold DejaVu Sans (shipped with matplotlib) in several sizes
        font_path_bold = fm.findfont(
            fm.FontProperties(family="DejaVu Sans", weight="bold")
        )
        f_title = ImageFont.truetype(font_path_bold, 80)
        f_month = ImageFont.truetype(font_path_bold, 50)
        f_day_hdr = ImageFont.truetype(font_path_bold, 30)
        f_date = ImageFont.truetype(font_path_bold, 32)
        f_event = ImageFont.truetype(font_path_bold, 24)
        f_legend = ImageFont.truetype(font_path_bold, 36)
    # Fall back to Pillow's built-in font if the font is unavailable
    except Exception:
        log("Warning: Standard bold fonts not found. Falling back to default.")
        f_title = f_month = f_day_hdr = f_date = f_event = f_legend = (
            ImageFont.load_default()
        )

    # Create the blank image and a drawing tool for it
    img = Image.new("RGB", (total_w, total_h), c_bg)
    draw = ImageDraw.Draw(img)

    def draw_centered_text(x, y, w, h, text, font, fill):
        """
        Draw text centered inside a box.

        Parameters
        ----------
        x, y : float
            Top-left corner of the box, in pixels.
        w, h : float
            Width and height of the box, in pixels.
        text : str
            The text to draw.
        font : PIL.ImageFont.ImageFont
            The font to draw with.
        fill : tuple of int
            The RGB text color.

        Returns
        -------
        None
        """
        # Measure the text so it can be centered
        bbox = draw.textbbox((0, 0), text, font=font)
        # Draw it centered (nudged up 5 px to look balanced)
        draw.text(
            (
                x + (w - (bbox[2] - bbox[0])) / 2,
                y + (h - (bbox[3] - bbox[1])) / 2 - 5,
            ),
            text,
            font=font,
            fill=fill,
        )

    # Draw the title across the top
    draw_centered_text(
        0,
        margin_y,
        total_w,
        title_height,
        f"National Weather Service Coastal Advisories - {station.upper()}",
        f_title,
        c_text,
    )

    # Draw the legend below the title
    leg_y = margin_y + title_height
    leg_box_w, leg_box_h, spacing = 50, 50, 400
    # Center the three legend entries horizontally
    start_x = (total_w - (3 * leg_box_w + 3 * 250 + 2 * spacing)) / 2 + 100

    for i, (label, color) in enumerate(
        [
            ("Watch (A)", c_watch),
            ("Advisory (Y)", c_adv),
            ("Warning (W)", c_warn),
        ]
    ):
        # Draw the color swatch and its label
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

    # Day-of-week column headers, with weeks starting on Sunday
    days_of_week = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    # Calendar whose weeks also start on Sunday (6), so the day cells
    # line up with the headers above
    sunday_first_calendar = calendar.Calendar(firstweekday=6)

    # Draw each month in a 4-column grid
    for m_idx, (month, year_val) in enumerate(months_sequence):
        # Grid row and column of this month
        row, col = m_idx // 4, m_idx % 4
        # Top-left corner of this month block
        x_off = margin_x + col * (month_w + gap_x)
        y_off = (
            margin_y + title_height + legend_height + row * (month_h + gap_y)
        )

        # Draw the month name bar
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

        # Draw the day-of-week header row
        for d_idx, d_name in enumerate(days_of_week):
            dx = x_off + d_idx * day_w
            dy = y_off + 80
            draw.rectangle(
                [dx, dy, dx + day_w, dy + 50],
                fill=c_day_hdr,
                outline=c_border,
                width=2,
            )
            draw_centered_text(
                dx, dy, day_w, 50, d_name, f_day_hdr, c_day_text
            )

        # Draw the day cells, one week per row (0 = no day)
        cal_matrix = sunday_first_calendar.monthdayscalendar(year_val, month)
        for w_idx, week in enumerate(cal_matrix):
            for d_idx, day_num in enumerate(week):
                # Top-left corner of this cell
                dx, dy = x_off + d_idx * day_w, y_off + 130 + w_idx * day_h

                # Grey out cells before the 1st and after the last day
                if day_num == 0:
                    draw.rectangle(
                        [dx, dy, dx + day_w, dy + day_h],
                        fill=(235, 235, 235),
                        outline=c_border,
                        width=2,
                    )
                    continue

                # Product codes active on this day, in sorted order
                current_date = date(year_val, month, day_num)
                events_list = sorted(list(daily_events[current_date]))

                # Color by the most severe significance that day
                fill_c = c_empty
                if events_list:
                    # The last character of each code is its significance
                    sigs = [e[-1] for e in events_list]
                    if "W" in sigs:
                        fill_c = c_warn
                    elif "Y" in sigs:
                        fill_c = c_adv
                    else:
                        fill_c = c_watch

                # Draw the cell
                draw.rectangle(
                    [dx, dy, dx + day_w, dy + day_h],
                    fill=fill_c,
                    outline=c_border,
                    width=2,
                )

                # Draw the day number in the top-right corner
                date_str = str(day_num)
                date_bbox = draw.textbbox((0, 0), date_str, font=f_date)
                draw.text(
                    (dx + day_w - (date_bbox[2] - date_bbox[0]) - 15, dy + 10),
                    date_str,
                    font=f_date,
                    fill=(50, 50, 50),
                )
                date_h = date_bbox[3] - date_bbox[1] + 10

                # List the product codes, two per line
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

    # Save the image
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(
        output_dir,
        f"nws_met_calendar_{station.lower()}_{start_year}-{end_year}.png",
    )
    img.save(out_png, format="PNG")
    log(f"Calendar successfully generated: {out_png}")

    return out_png


def main():
    """
    Fetch one WFO's advisories, save them to CSV, and draw a calendar.

    Reads the WFO ID, meteorological year, and output folder from the
    command line.

    Returns
    -------
    None
    """
    # Define the command-line arguments
    parser = argparse.ArgumentParser(
        description="Fetch NWS coastal/tropical advisories for one office "
        "and meteorological year, and draw a calendar of them."
    )
    parser.add_argument(
        "station", type=str, help="3-letter WFO station ID (e.g., ILM, MHX)."
    )
    parser.add_argument(
        "met_year",
        type=int,
        help="Starting year for the meteorological cycle (e.g., 2025 for "
        "May 2025 - Apr 2026).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to save the CSV and calendar (default: current "
        "directory).",
    )
    args = parser.parse_args()

    # Download the advisories
    events_df = fetch_nws_advisories(args.station.upper(), args.met_year)

    # Nothing else to do if there were none
    if events_df.empty:
        return

    log(f"Retrieved {len(events_df)} extreme events.")

    # Save the advisories to CSV
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(
        args.outdir, f"{args.station.lower()}_events_{args.met_year}.csv"
    )
    events_df.to_csv(csv_path, index=False)
    log(f"CSV data saved to: {csv_path}")

    # Draw the calendar from the same table
    plot_advisory_calendar(
        events_df, args.station.upper(), args.met_year, args.outdir
    )


# Run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
