# Flood Event QCing
This is a repository containing a Jupyter notebook for cataloging flood events from the Sunny Day Flooding Project water level API. Utility functions for 
processing are contained in utils.py. This Jupyter notebook requires a username and password to access the water level API. 

Outputs of the notebook include:
* abbr_flood_events.csv (An abbreviated list of flood events with start and end times, durations, and maximum water levels.)
* flood_events.csv (A full list of flood events with every recorded datapoint. Ending points of flood events include start and end times, durations, and maximum water levels.)
* sensor_outages.csv (A list of outages that occurred and flags for if they occurred during a flood event.)
* flood_plots (A folder containing plots of each flood event labeled with event number and sensor ID.)

Outputs are organized as follows in the repository directory:
```
flood_event_QC
│ README.md
│ index.html
│ LICENSE
│ flood_tracking.ipynb
│ utils.py
│ abbr_flood_events.csv
│ flood_events.csv
│ sensor_outages.csv
└───flood_plots
│ │ flood_event_{#}_{sensor_ID}.png
```
