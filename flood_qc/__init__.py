"""
Flood event QC tools for the Sunny Day Flooding Project.

Modules
-------
event_tracker
    Pull water level data from the SDFP API, detect flood events and
    sensor outages, and plot each flood event.
image_handler
    Copy camera images for flood events and outages from the SDFP image
    archive.
extreme_events
    Command-line tool that tags QC'd flood events with extreme event
    criteria, or builds a master list of extreme events.
nws_advisories
    Fetch NWS coastal and tropical advisories and draw a calendar of
    them.
wave_storms
    Detect wave-driven storms from NDBC buoy and NOAA tide gauge data.
flood_time_summary
    Plot when flood events occur (day/night, hour, weekday, season).
utils
    Shared logging helper.

Examples
--------
>>> from flood_qc import EventTracker, ImageHandler
"""

# Expose the two main classes at the package level for convenience
from .event_tracker import EventTracker
from .image_handler import ImageHandler

# Names imported by "from flood_qc import *"
__all__ = ["EventTracker", "ImageHandler"]
