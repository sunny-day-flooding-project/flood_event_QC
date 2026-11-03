"""
Flood Event Temporal Summary.

Plots a 2x2 summary of when flood events occur: day vs. night (based on the
event start and local sunrise/sunset), hour of day, day of week, and season.

Example
-------
python time_dist_summary.py data/approved_events.csv --lat 34.8697 --lon -76.3934
"""

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astral import LocationInfo
from astral.sun import sun
import pytz


def load_and_prepare_data(filepath):
    """
    Load the flood event CSV and convert timestamps to Eastern Time.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the flood events.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the flood events with localized Eastern Time 
        datetime objects for start and end times.
    """
    df = pd.read_csv(filepath)
    start_utc = pd.to_datetime(df['start_time_UTC'])
    end_utc = pd.to_datetime(df['end_time_UTC'])
    
    df['start_time_Eastern'] = start_utc.dt.tz_convert('US/Eastern')
    df['end_time_Eastern'] = end_utc.dt.tz_convert('US/Eastern')
    
    return df


def get_season(month):
    """
    Map a calendar month to a meteorological season.
    
    Parameters
    ----------
    month : int
        The calendar month as an integer (1-12).
        
    Returns
    -------
    str
        The name of the corresponding season (Winter, Spring, Summer, Fall).
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def calculate_temporal_metrics(df, lat, lon):
    """
    Calculate required temporal statistics handling multi-span events and days.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the processed flood event data.
    lat : float
        Latitude for sunrise and sunset calculations.
    lon : float
        Longitude for sunrise and sunset calculations.
        
    Returns
    -------
    tuple
        Contains four elements: day/night statistics dictionary, 
        hourly dictionary (events and days), weekday dictionary (events and days), 
        and season dictionary (events and days).
    """
    loc = LocationInfo(latitude=lat, longitude=lon)
    eastern = pytz.timezone('US/Eastern')
    
    day_flood_days = set()
    night_flood_days = set()
    event_day_count = 0
    event_night_count = 0
    
    # Initialize dictionaries to hold event counts
    hourly_events = {h: 0 for h in range(24)}
    weekday_events = {w: 0 for w in range(7)}
    season_events = {'Winter': 0, 'Spring': 0, 'Summer': 0, 'Fall': 0}
    
    # Initialize dictionaries to track unique dates for days metrics
    hourly_days = {h: set() for h in range(24)}
    weekday_days = {w: set() for w in range(7)}
    season_days = {'Winter': set(), 'Spring': set(), 'Summer': set(), 'Fall': set()}
    
    sun_cache = {}
    
    for _, row in df.iterrows():
        start_time = row['start_time_Eastern']
        end_time = row['end_time_Eastern']
        
        # --- Day/Night classification based ONLY on start_time ---
        start_date = start_time.date()
        
        if start_date not in sun_cache:
            try:
                s = sun(loc.observer, date=start_date, tzinfo=eastern)
                sun_cache[start_date] = (s['sunrise'], s['sunset'])
            except Exception:
                sun_cache[start_date] = (None, None)
                
        sunrise, sunset = sun_cache[start_date]
        
        # Determine if the start time falls during the day
        is_day_event = False
        if sunrise and sunset and (sunrise <= start_time < sunset):
            is_day_event = True
            
        if is_day_event:
            event_day_count += 1
        else:
            event_night_count += 1
            
        # --- Processing temporal spread for the other metrics ---
        time_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        if len(time_range) == 0:
            time_range = pd.DatetimeIndex([start_time])
            
        # Count unique events per category
        for h in set(t.hour for t in time_range):
            hourly_events[h] += 1
            
        for w in set(t.dayofweek for t in time_range):
            weekday_events[w] += 1
            
        for s in set(get_season(t.month) for t in time_range):
            season_events[s] += 1
            
        # Track days
        for t in time_range:
            d = t.date()
            
            # Track unique days for hourly, weekday, and seasonal aggregations
            hourly_days[t.hour].add(d)
            weekday_days[t.dayofweek].add(d)
            season_days[get_season(t.month)].add(d)
            
            # Add all days this event spanned to the appropriate day/night set based on start_time classification
            if is_day_event:
                day_flood_days.add(d)
            else:
                night_flood_days.add(d)
                    
    day_night_stats = {
        'event_day_count': event_day_count,
        'event_night_count': event_night_count,
        'day_flood_days_count': len(day_flood_days),
        'night_flood_days_count': len(night_flood_days)
    }
    
    # Convert sets to lengths to yield the count of unique days
    hourly_stats = {
        'events': hourly_events, 
        'days': {h: len(days) for h, days in hourly_days.items()}
    }
    weekday_stats = {
        'events': weekday_events, 
        'days': {w: len(days) for w, days in weekday_days.items()}
    }
    season_stats = {
        'events': season_events, 
        'days': {s: len(days) for s, days in season_days.items()}
    }
    
    return day_night_stats, hourly_stats, weekday_stats, season_stats


def _add_dual_axis_plot(ax, x_positions, x_labels, events_data, days_data, 
                        title, xlabel=None, color_events="#009E73", 
                        color_days="#CC79A7", width=0.35):
    """
    Helper function to plot dual-axis bar charts for events and days.
    """
    # Create the secondary axis
    ax2 = ax.twinx()
    
    # Plot bars offset to avoid overlap
    rects1 = ax.bar(x_positions - width/2, events_data, width, 
                    label='Events', color=color_events)
    rects2 = ax2.bar(x_positions + width/2, days_data, width, 
                     label='Days', color=color_days)
    
    # Set labels and properties
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Events')
    ax2.set_ylabel('Number of Days')
    ax.set_title(title)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    
    # Combine legends from both axes
    lines_labels = [ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax.legend(lines, labels, loc='upper left')


def plot_summary_figure(day_night_stats, hourly_stats, weekday_stats, 
                        season_stats, output_file):
    """
    Generate and save a 2x2 grid of bar plots using the Okabe-Ito palette.
    
    Parameters
    ----------
    day_night_stats : dict
        Dictionary of counts for day/night events and days.
    hourly_stats : dict
        Nested dictionary of 'events' and 'days' counts mapped to hour (0-23).
    weekday_stats : dict
        Nested dictionary of 'events' and 'days' counts mapped to weekday (0-6).
    season_stats : dict
        Nested dictionary of 'events' and 'days' counts mapped to seasons.
    output_file : str
        Filepath where the resulting PNG should be saved.
    """
    c_orange = "#E69F00"
    c_sky_blue = "#56B4E9"
    c_green = "#009E73"
    c_vermilion = "#D55E00"
    c_pink = "#CC79A7"
    c_blue = "#0072B2"
    c_yellow = "#F0E442"
    
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Flood Event Temporal Summary", fontsize=16)
    
    # --- Top Left Plot: Day vs Night Counts ---
    ax = axes[0, 0]
    labels = ['Flood Events', 'Flood Days']
    
    day_vals = [day_night_stats['event_day_count'], 
                day_night_stats['day_flood_days_count']]
    night_vals = [day_night_stats['event_night_count'], 
                  day_night_stats['night_flood_days_count']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, day_vals, width, label='Daytime', color=c_yellow)
    rects2 = ax.bar(x + width/2, night_vals, width, label='Nighttime', color=c_blue)
    
    # Add text annotations on top of the bars
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    ax.set_ylabel('Count')
    ax.set_title('Day vs. Night Occurrences')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Ensure y-axis accommodates text annotations comfortably
    ax.set_ylim(0, max(max(day_vals), max(night_vals)) * 1.15) 
    ax.legend(loc='upper right')
    
    # --- Top Right Plot: Hourly Counts (Dual Axis) ---
    hours = np.arange(24)
    h_events = [hourly_stats['events'].get(h, 0) for h in hours]
    h_days = [hourly_stats['days'].get(h, 0) for h in hours]
    
    _add_dual_axis_plot(
        ax=axes[0, 1],
        x_positions=hours,
        x_labels=hours,
        events_data=h_events,
        days_data=h_days,
        title='Occurrences by Hour of Day',
        xlabel='Hour of Day (Eastern Time)',
        color_events=c_sky_blue,
        color_days=c_pink
    )
    
    # --- Bottom Left Plot: Day of Week (Dual Axis) ---
    weekdays_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    w_indices = np.arange(7)
    w_events = [weekday_stats['events'].get(w, 0) for w in w_indices]
    w_days = [weekday_stats['days'].get(w, 0) for w in w_indices]
    
    _add_dual_axis_plot(
        ax=axes[1, 0],
        x_positions=w_indices,
        x_labels=weekdays_labels,
        events_data=w_events,
        days_data=w_days,
        title='Occurrences by Day of Week',
        color_events=c_sky_blue,
        color_days=c_orange
    )
    
    # --- Bottom Right Plot: Seasonality (Dual Axis) ---
    seasons_labels = ['Winter', 'Spring', 'Summer', 'Fall']
    s_indices = np.arange(4)
    s_events = [season_stats['events'].get(s, 0) for s in seasons_labels]
    s_days = [season_stats['days'].get(s, 0) for s in seasons_labels]
    
    _add_dual_axis_plot(
        ax=axes[1, 1],
        x_positions=s_indices,
        x_labels=seasons_labels,
        events_data=s_events,
        days_data=s_days,
        title='Occurrences by Season',
        color_events=c_sky_blue,
        color_days=c_orange
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300)
    plt.close()


def main(filepath, lat, lon, output_file="flood_summary.png"):
    """
    Main execution function to read data, process metrics, and generate plots.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    lat : float
        Latitude of the sensor locations.
    lon : float
        Longitude of the sensor locations.
    output_file : str, optional
        Desired filepath for the generated plot PNG (default is 'flood_summary.png').
    """
    df = load_and_prepare_data(filepath)
    stats = calculate_temporal_metrics(df, lat, lon)
    
    day_night_stats, hourly_stats, weekday_stats, season_stats = stats
    
    plot_summary_figure(day_night_stats, hourly_stats, weekday_stats, 
                        season_stats, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the temporal distribution (day/night, hour, weekday, season) of flood events."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="CSV of flood events with start_time_UTC and end_time_UTC columns (e.g., approved events from qc_results.csv).",
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Sensor latitude in decimal degrees, used for sunrise/sunset (e.g., 34.8697).",
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Sensor longitude in decimal degrees, used for sunrise/sunset (e.g., -76.3934).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flood_summary.png",
        help="Output PNG path (default: flood_summary.png).",
    )

    args = parser.parse_args()
    main(args.csv_path, args.lat, args.lon, args.output)