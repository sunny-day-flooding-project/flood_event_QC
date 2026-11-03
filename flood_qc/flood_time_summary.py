"""
Temporal summary plots of flood events.

Makes a 2 x 2 figure showing when flood events occur:

- day vs. night, based on each event's start time and the local sunrise
  and sunset at the sensor,
- hour of day,
- day of week,
- season.

Each panel counts both flood events and unique flood days. Times are
shown in US Eastern local time (with daylight saving time).

Functions
---------
load_and_prepare_data
    Load a flood event CSV and add Eastern-time columns.
get_season
    Map a month number to its meteorological season.
calculate_temporal_metrics
    Count events and days by day/night, hour, weekday, and season.
plot_summary_figure
    Draw and save the 2 x 2 summary figure.
main
    Run the full summary for one CSV.

Examples
--------
Run from the repository root::

    python -m flood_qc.flood_time_summary events.csv --lat 34.87 --lon -76.39
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from astral import LocationInfo
from astral.sun import sun

# US Eastern local time zone (observes daylight saving time)
EASTERN_TZ = pytz.timezone("US/Eastern")

# Okabe-Ito colorblind-safe palette colors used in the figure
C_ORANGE = "#E69F00"
C_SKY_BLUE = "#56B4E9"
C_PINK = "#CC79A7"
C_BLUE = "#0072B2"
C_YELLOW = "#F0E442"

# Season names in plotting order
SEASONS = ["Winter", "Spring", "Summer", "Fall"]


def load_and_prepare_data(filepath):
    """
    Load a flood event CSV and add Eastern-time columns.

    Parameters
    ----------
    filepath : str
        Path of a CSV with ``start_time_UTC`` and ``end_time_UTC``
        columns.

    Returns
    -------
    pandas.DataFrame
        The events with added ``start_time_Eastern`` and
        ``end_time_Eastern`` columns (US Eastern local time).
    """
    # Load the events
    df = pd.read_csv(filepath)
    # Parse the UTC start and end times
    start_utc = pd.to_datetime(df["start_time_UTC"])
    end_utc = pd.to_datetime(df["end_time_UTC"])

    # Convert them to US Eastern local time
    df["start_time_Eastern"] = start_utc.dt.tz_convert(EASTERN_TZ)
    df["end_time_Eastern"] = end_utc.dt.tz_convert(EASTERN_TZ)

    return df


def get_season(month):
    """
    Map a month number to its meteorological season.

    Parameters
    ----------
    month : int
        Calendar month (1-12).

    Returns
    -------
    str
        ``"Winter"`` (Dec-Feb), ``"Spring"`` (Mar-May), ``"Summer"``
        (Jun-Aug), or ``"Fall"`` (Sep-Nov).
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def calculate_temporal_metrics(df, lat, lon):
    """
    Count events and days by day/night, hour, weekday, and season.

    Each event is classified as daytime or nighttime by its start time
    alone. For the hour, weekday, and season counts, an event counts
    once in every category it spans (checked minute by minute), and a
    day counts once in each category it had flooding in.

    Parameters
    ----------
    df : pandas.DataFrame
        Events from :func:`load_and_prepare_data`.
    lat : float
        Sensor latitude in decimal degrees, used for sunrise and sunset.
    lon : float
        Sensor longitude in decimal degrees, used for sunrise and
        sunset.

    Returns
    -------
    day_night_stats : dict
        ``event_day_count`` and ``event_night_count`` (events starting
        in daylight or darkness), and ``day_flood_days_count`` and
        ``night_flood_days_count`` (unique days spanned by those
        events).
    hourly_stats : dict
        ``{"events": {hour: count}, "days": {hour: count}}`` for hours
        0-23.
    weekday_stats : dict
        ``{"events": {weekday: count}, "days": {weekday: count}}`` for
        weekdays 0 (Monday) to 6 (Sunday).
    season_stats : dict
        ``{"events": {season: count}, "days": {season: count}}``.
    """
    # Describe the sensor's location for the sunrise/sunset calculation
    loc = LocationInfo(latitude=lat, longitude=lon)

    # Unique days spanned by daytime-start and nighttime-start events
    day_flood_days = set()
    night_flood_days = set()
    # Number of events starting in daylight and in darkness
    event_day_count = 0
    event_night_count = 0

    # Number of events touching each hour, weekday, and season
    hourly_events = {h: 0 for h in range(24)}
    weekday_events = {w: 0 for w in range(7)}
    season_events = {s: 0 for s in SEASONS}

    # Unique dates with flooding in each hour, weekday, and season
    hourly_days = {h: set() for h in range(24)}
    weekday_days = {w: set() for w in range(7)}
    season_days = {s: set() for s in SEASONS}

    # Sunrise and sunset per date, so each date is computed only once
    sun_cache = {}

    for _, row in df.iterrows():
        start_time = row["start_time_Eastern"]
        end_time = row["end_time_Eastern"]

        # ----- Day/night classification (start time only) -----
        start_date = start_time.date()

        # Compute and cache sunrise and sunset for the start date
        if start_date not in sun_cache:
            try:
                s = sun(loc.observer, date=start_date, tzinfo=EASTERN_TZ)
                sun_cache[start_date] = (s["sunrise"], s["sunset"])
            # Some dates/locations have no sunrise or sunset
            except Exception:
                sun_cache[start_date] = (None, None)

        sunrise, sunset = sun_cache[start_date]

        # Daytime if the event starts between sunrise and sunset
        is_day_event = False
        if sunrise and sunset and (sunrise <= start_time < sunset):
            is_day_event = True

        # Count the event as daytime or nighttime
        if is_day_event:
            event_day_count += 1
        else:
            event_night_count += 1

        # ----- Hour, weekday, and season spread -----
        # List every minute the event covers
        time_range = pd.date_range(start=start_time, end=end_time, freq="1min")
        # Use just the start time if the event is shorter than a minute
        if len(time_range) == 0:
            time_range = pd.DatetimeIndex([start_time])

        # Count the event once in each hour, weekday, and season it
        # touches
        for h in set(t.hour for t in time_range):
            hourly_events[h] += 1

        for w in set(t.dayofweek for t in time_range):
            weekday_events[w] += 1

        for s in set(get_season(t.month) for t in time_range):
            season_events[s] += 1

        # Record the date of every minute in each category it falls in
        for t in time_range:
            d = t.date()

            hourly_days[t.hour].add(d)
            weekday_days[t.dayofweek].add(d)
            season_days[get_season(t.month)].add(d)

            # Every day the event spans counts as a daytime or
            # nighttime flood day, based on the event's start
            if is_day_event:
                day_flood_days.add(d)
            else:
                night_flood_days.add(d)

    # Package the day/night counts
    day_night_stats = {
        "event_day_count": event_day_count,
        "event_night_count": event_night_count,
        "day_flood_days_count": len(day_flood_days),
        "night_flood_days_count": len(night_flood_days),
    }

    # Turn each set of dates into its size (number of unique days)
    hourly_stats = {
        "events": hourly_events,
        "days": {h: len(days) for h, days in hourly_days.items()},
    }
    weekday_stats = {
        "events": weekday_events,
        "days": {w: len(days) for w, days in weekday_days.items()},
    }
    season_stats = {
        "events": season_events,
        "days": {s: len(days) for s, days in season_days.items()},
    }

    return day_night_stats, hourly_stats, weekday_stats, season_stats


def _add_dual_axis_plot(
    ax,
    x_positions,
    x_labels,
    events_data,
    days_data,
    title,
    xlabel=None,
    color_events="#009E73",
    color_days=C_PINK,
    width=0.35,
):
    """
    Draw side-by-side event and day bars with separate y-axes.

    Events use the left y-axis and days use the right y-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on (used for the event bars).
    x_positions : array_like
        x positions of the bar groups.
    x_labels : list
        Tick labels for the bar groups.
    events_data : list of int
        Event counts, one per group.
    days_data : list of int
        Day counts, one per group.
    title : str
        Axes title.
    xlabel : str, optional
        x-axis label. If None (the default), no label is drawn.
    color_events : str, optional
        Color of the event bars. Default is Okabe-Ito green.
    color_days : str, optional
        Color of the day bars. Default is Okabe-Ito pink.
    width : float, optional
        Width of each bar. Default is 0.35.

    Returns
    -------
    None
    """
    # Make a second y-axis that shares the x-axis
    ax2 = ax.twinx()

    # Draw the event bars left of each position and the day bars right
    ax.bar(
        x_positions - width / 2,
        events_data,
        width,
        label="Events",
        color=color_events,
    )
    ax2.bar(
        x_positions + width / 2,
        days_data,
        width,
        label="Days",
        color=color_days,
    )

    # Label the axes and title
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Events")
    ax2.set_ylabel("Number of Days")
    ax.set_title(title)

    # Put a labeled tick under each bar group
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)

    # Combine both axes' legend entries into one legend
    lines_labels = [
        ax.get_legend_handles_labels(),
        ax2.get_legend_handles_labels(),
    ]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax.legend(lines, labels, loc="upper left")


def plot_summary_figure(
    day_night_stats, hourly_stats, weekday_stats, season_stats, output_file
):
    """
    Draw and save the 2 x 2 summary figure.

    Parameters
    ----------
    day_night_stats : dict
        Day/night counts from :func:`calculate_temporal_metrics`.
    hourly_stats : dict
        Hourly counts from :func:`calculate_temporal_metrics`.
    weekday_stats : dict
        Weekday counts from :func:`calculate_temporal_metrics`.
    season_stats : dict
        Seasonal counts from :func:`calculate_temporal_metrics`.
    output_file : str
        Path to save the PNG to.

    Returns
    -------
    None
    """
    # Create the 2 x 2 grid of panels
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Flood Event Temporal Summary", fontsize=16)

    # ----- Top left: day vs. night -----
    ax = axes[0, 0]
    labels = ["Flood Events", "Flood Days"]

    # Daytime and nighttime values for each bar group
    day_vals = [
        day_night_stats["event_day_count"],
        day_night_stats["day_flood_days_count"],
    ]
    night_vals = [
        day_night_stats["event_night_count"],
        day_night_stats["night_flood_days_count"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    # Draw the daytime and nighttime bars side by side
    rects1 = ax.bar(
        x - width / 2, day_vals, width, label="Daytime", color=C_YELLOW
    )
    rects2 = ax.bar(
        x + width / 2, night_vals, width, label="Nighttime", color=C_BLUE
    )

    # Write each bar's value above it
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    # Label the panel
    ax.set_ylabel("Count")
    ax.set_title("Day vs. Night Occurrences")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Leave 15% headroom so the value labels fit
    ax.set_ylim(0, max(max(day_vals), max(night_vals)) * 1.15)
    ax.legend(loc="upper right")

    # ----- Top right: hour of day -----
    hours = np.arange(24)
    h_events = [hourly_stats["events"].get(h, 0) for h in hours]
    h_days = [hourly_stats["days"].get(h, 0) for h in hours]

    _add_dual_axis_plot(
        ax=axes[0, 1],
        x_positions=hours,
        x_labels=hours,
        events_data=h_events,
        days_data=h_days,
        title="Occurrences by Hour of Day",
        xlabel="Hour of Day (Eastern Time)",
        color_events=C_SKY_BLUE,
        color_days=C_PINK,
    )

    # ----- Bottom left: day of week (0 = Monday) -----
    weekdays_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    w_indices = np.arange(7)
    w_events = [weekday_stats["events"].get(w, 0) for w in w_indices]
    w_days = [weekday_stats["days"].get(w, 0) for w in w_indices]

    _add_dual_axis_plot(
        ax=axes[1, 0],
        x_positions=w_indices,
        x_labels=weekdays_labels,
        events_data=w_events,
        days_data=w_days,
        title="Occurrences by Day of Week",
        color_events=C_SKY_BLUE,
        color_days=C_ORANGE,
    )

    # ----- Bottom right: season -----
    s_indices = np.arange(len(SEASONS))
    s_events = [season_stats["events"].get(s, 0) for s in SEASONS]
    s_days = [season_stats["days"].get(s, 0) for s in SEASONS]

    _add_dual_axis_plot(
        ax=axes[1, 1],
        x_positions=s_indices,
        x_labels=SEASONS,
        events_data=s_events,
        days_data=s_days,
        title="Occurrences by Season",
        color_events=C_SKY_BLUE,
        color_days=C_ORANGE,
    )

    # Fit the panels below the title, save, and free the memory
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300)
    plt.close()


def main(filepath, lat, lon, output_file="flood_summary.png"):
    """
    Run the full temporal summary for one CSV.

    Parameters
    ----------
    filepath : str
        Path of the flood event CSV.
    lat : float
        Sensor latitude in decimal degrees.
    lon : float
        Sensor longitude in decimal degrees.
    output_file : str, optional
        Path to save the PNG to. Default is ``"flood_summary.png"``.

    Returns
    -------
    None
    """
    # Load the events, count them, and draw the figure
    df = load_and_prepare_data(filepath)
    day_night_stats, hourly_stats, weekday_stats, season_stats = (
        calculate_temporal_metrics(df, lat, lon)
    )
    plot_summary_figure(
        day_night_stats, hourly_stats, weekday_stats, season_stats, output_file
    )


def _parse_args():
    """
    Read the command-line arguments.

    Returns
    -------
    argparse.Namespace
        ``csv_path``, ``lat``, ``lon``, and ``output``.
    """
    parser = argparse.ArgumentParser(
        description="Plot the temporal distribution (day/night, hour, "
        "weekday, season) of flood events."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="CSV of flood events with start_time_UTC and end_time_UTC "
        "columns (e.g., approved events from qc_results.csv).",
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Sensor latitude in decimal degrees, used for sunrise/sunset "
        "(e.g., 34.8697).",
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Sensor longitude in decimal degrees, used for sunrise/sunset "
        "(e.g., -76.3934).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flood_summary.png",
        help="Output PNG path (default: flood_summary.png).",
    )
    return parser.parse_args()


# Run the summary only when executed as a script, not when imported
if __name__ == "__main__":
    args = _parse_args()
    main(args.csv_path, args.lat, args.lon, args.output)
