# Flood Event QC

Tools for building a quality-controlled catalog of roadway flood events from the
Sunny Day Flooding Project (SDFP) water level sensors.

The workflow has two parts:

1. **Interactive QC (Streamlit app).** Pull water level data from the SDFP API, detect candidate flood
   events and sensor outages automatically, then step through each one in a browser with its water level
   plot and camera images. You record whether each event really flooded.
2. **Extreme event cross-reference (command-line scripts).** Tag each QC'd flood event by whether it
   happened during an extreme coastal event. These scripts use National Weather Service (NWS) records,
   NWS advisories, and observed wave conditions.

> **Access requirements.** You can't run the data pull from a fresh clone. It needs credentials and data
> that are deliberately left out of this public repository:
> - an **SDFP API username and password**, and
> - read access to the **SDFP camera image archive**.
>
> Ask the SDFP data team for both. **Never commit credentials, pulled data, or camera images to this
> repository.** See [Keeping data out of git](#keeping-data-out-of-git).

---

## Contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit QC dashboard: pulls the data and runs the review workflow. |
| `flood_tools.py` | Core library. `EventTracker` (API pull, event and outage detection, plots, error metrics) and `ImageHandler` (copying camera images from the archive). |
| `.streamlit/config.toml` | App theme. It also turns on Streamlit **static file serving**, which the image viewer needs. Don't delete it. |
| `logo.png` | Sidebar logo used by the app. |
| `environment.yml` | Conda environment definition (`flood-qc`). |
| `qc_for_extreme_events.py` | Tags each event in a QC results CSV with the extreme-event criteria. |
| `extreme_event_qc_config.yaml` | Example config for `qc_for_extreme_events.py`. |
| `gen_extreme_events_master.py` | Builds a standalone master list of extreme events for a meteorological year, independent of any flood events. |
| `extreme_event_master_config.yaml` | Example config for `gen_extreme_events_master.py`. |
| `pull_nws_advisories.py` | Downloads NWS coastal and tropical watches, warnings, and advisories from the Iowa Environmental Mesonet (IEM) and draws a calendar of them. Imported by both extreme-event scripts and also runs on its own. |
| `storms_from_waves_wtr_levels.py` | Finds wave-driven storm events at an NDBC buoy and the matching water level and surge at a NOAA tide gauge. Imported by both extreme-event scripts and also runs on its own. |
| `StormsFromWaveandWaterLevel.m` | The original MATLAB storm-finding script (L. Velasquez Montoya, 2017; later edits by E. Farquhar). `storms_from_waves_wtr_levels.py` is a Python port of it. Kept for reference. |
| `Significant_Events_<year>.csv` | Hand-curated lists of significant weather events from NWS Weather Forecast Offices (WFOs). Used as one of the extreme-event criteria. |
| `time_dist_summary.py` | Optional post-QC analysis: plots how flood events are distributed over day vs. night, hour of day, day of week, and season. |

---

## Setup

1. Install [Miniconda/Miniforge](https://github.com/conda-forge/miniforge) if you don't have conda.
2. Create and activate the environment from the repository root:

   ```bash
   conda env create -f environment.yml
   conda activate flood-qc
   ```

3. Make sure you can reach the camera image archive (for example, a mounted network drive). The app
   expects this folder layout under the path you give it:

   ```
   <Source Image Drive Path>/
   └── <YYYY> Archive/
       └── CAM_<sensor_ID>/            e.g. CAM_CB_01
           └── <YYYY-MM-DD>/           UTC date
               └── CAM_<sensor_ID>_..._<YYYYMMDDHHMMSS>....jpg
   ```

   Image filenames must start with `CAM_<sensor_ID>_`, end in `.jpg`, and contain a 14-digit UTC
   timestamp. The timestamp is how images are matched to events.

---

## Part 1: Interactive QC in the Streamlit app

### 1. Launch the app

Run this **from the repository root**:

```bash
conda activate flood-qc
streamlit run app.py
```

Streamlit opens the dashboard in your browser (usually http://localhost:8501).

> You must launch from the repository root. The app reads `.streamlit/config.toml`, writes its working
> files to `./data/` and `./static/`, and serves camera images from `./static/`, all relative to the
> folder you launch from.

### 2. Pull new data (sidebar, "Pull New Data")

Fill in the form and click **Fetch Data & Images**.

| Field | Notes |
|---|---|
| Query by | **Region Name** or **Specific Sensor IDs**. |
| API Username / Password | Your SDFP API credentials. They're typed in each session and held in memory only for the pull. They aren't saved to disk. |
| Region Name | One of `Carolina Beach` (CB_01–03), `Beaufort` (BF_01), `Down East` (DE_01–04), `New Bern` (NB_01–02), `North River` (NR_01). Not case-sensitive. To add a region, edit `EventTracker._sensor_list_generator` in `flood_tools.py`. |
| Sensor IDs | Comma-separated, e.g. `CB_01, CB_02`. |
| Start / End Date | Inclusive date range to pull. |
| Source Image Drive Path | Root of the camera image archive (see [Setup](#setup)). |
| Error CSV Path | *Optional.* A CSV with `Date`, `Mean`, `5th Pct`, and `95th Pct` columns. If you give one, the 7-day average of each column before every event's end time is added to that event and shown during review. |
| Filter outage images to daylight hours only | Recommended. Copies outage images only between sunrise and sunset at each sensor, which cuts out many unusable night images. |

A live process log appears while the pull runs. The pull does the following:

1. **Downloads water level data** from the SDFP API (`get_water_level` endpoint), split into 30-day
   chunks and fetched in parallel. Any chunk that fails is retried one day at a time.
2. **Detects flood events** for each sensor from `road_water_level_adj` (water level relative to the
   road surface, in ft). An event **starts** when the level rises above **0.02 ft** and **ends** when it
   falls back **below 0 ft**. Events at different sensors that overlap in time get the same
   `flood_event` number. An event is identified by its (`flood_event`, `sensor_ID`) pair.
3. **Detects sensor outages.** An outage is any gap of **more than 1 hour** between consecutive readings
   from one sensor. Outages that start during a flood event are cross-flagged in both CSVs.
4. **Plots each event**: water level from 1 day before to 1 day after the event, with the flood points
   marked.
5. **Adds error metrics** (only if an error CSV was given).
6. **Copies camera images** from the archive into `static/event_images/`: everything from **1 hour
   before to 1 hour after** each flood event, plus images covering each outage. Images that are already
   there are skipped.

### 3. Review events (Workflow Mode: **Standard QC**)

The app shows unreviewed events one at a time. For each one you see:

- event start and end (EST), maximum depth over the road, and duration
- the water level plot
- error metrics, if you added them
- a carousel of camera images from the event window (with a full-screen button)

Then record a decision and click **Save & Next**:

- **Event Status:** `Approved (Flooded)` or `Rejected (Not Flooded)`
- **Visual confirmation:** `Yes` / `No` (whether the images show flooding)
- **During King Tide?**
- **Flag for secondary review**
- **Notes**

Each decision is saved right away to `data/qc_results.csv`. You can close the app and pick up where you
left off later. The progress bar counts events that have a decision.

### 4. Second look (Workflow Mode: **Review Flagged Events**)

Shows only the events you flagged in Standard QC, so a second reviewer can confirm or change the
decision. Uncheck the flag when the event is resolved.

### 5. Check outages for missed floods (Workflow Mode: **Review Outages**)

While a sensor was down, a flood could have happened without being recorded. This mode steps through
each outage with its (daylight) images.

- If the images show flooding, choose **Yes**, then select the **start image** and **end image** of the
  flooding. The app creates a new flood event for those times and saves it to `data/qc_results.csv` as
  `Approved (Flooded)` with visual confirmation. Depth is left blank because no sensor data exists for
  it.
- Leave **Mark entire outage as fully reviewed** checked to move on. Reviewed outages are logged in
  `data/outages_reviewed.csv`. To log more than one flood in a single outage, uncheck it, save the first
  flood, and record the next one.

### 6. Make the flood calendar (sidebar, "Generate Calendar")

Pick a year and click **Generate Flood Calendar**. Every calendar day (EST) touched by an
`Approved (Flooded)` event in `data/qc_results.csv` is highlighted. The PNG is saved to
`data/flood_plots/flood_calendar_<year>.png` and can be downloaded from the app.

### Files the app creates

```
flood_event_QC/
├── data/
│   ├── flood_events.csv          # every data point inside each flood event
│   ├── abbr_flood_events.csv     # one row per event: times, duration, max depth, outage flag, error metrics
│   ├── sensor_outages.csv        # one row per outage, with daylight lat/lon and flood overlap flags
│   ├── qc_results.csv            # ★ your QC decisions (the main product of Part 1)
│   ├── outages_reviewed.csv      # outage numbers you've finished reviewing
│   └── flood_plots/
│       ├── flood_event_<#>_<sensor_ID>.png
│       └── flood_calendar_<year>.png
└── static/
    └── event_images/             # local copies of camera images for the image viewer
```

### Important: one data pull per QC session

Each pull renumbers **all** flood events and outages from scratch. QC decisions and reviewed outages are
stored by those numbers. If you pull again into a `data/` folder that already has QC work, saved
decisions can end up attached to the wrong events. Existing plots are also skipped, not redrawn, so they
may show the wrong event.

To do this safely:

- **Pull the full date range and all the sensors you need in one go**, then do the QC.
- To start a new QC period or region, **move or rename the existing `data/` and `static/` folders
  first**, for example to an archive location outside the repository.

---

## Part 2: Extreme event cross-reference

After QC, `qc_for_extreme_events.py` checks each event in `qc_results.csv` against three independent
signs of an extreme coastal event:

| Column | Criterion | Source |
|---|---|---|
| `significant_weather` | The event overlaps a significant weather event listed by one of the configured NWS offices. | `Significant_Events_<year>.csv` (in this repo) |
| `advisory_issued` | The event overlaps an NWS **Coastal Flood (CF), Hurricane (HU), Tropical Storm (TR), Storm Surge (SS), Gale (GL), or Hurricane Force Wind (HF)** warning, advisory, or watch from one of the configured offices. | IEM watch/warning/advisory archive |
| `obs_elevated_wtr_level` | The event overlaps a wave-driven storm: significant wave height **> 1.5 m for > 8 hours** at a configured NDBC buoy. Peak water level and surge at the paired NOAA gauge are recorded too. | NDBC + NOAA CO-OPS |

`meets_criteria_for_extreme_event` is `True` when **at least 2 of the 3** criteria are met.

### Configure

Copy `extreme_event_qc_config.yaml` and edit it:

```yaml
input_csv: "./data/qc_results.csv"   # QC results from Part 1
met_year: 2025                       # meteorological year: May 1, 2025 – Apr 30, 2026

nws_offices:                         # NWS WFO IDs to use for advisories and significant events
  - "ILM"                            # Wilmington, NC
  - "MHX"                            # Newport/Morehead City, NC

station_pairs:                       # [NDBC wave buoy, NOAA tide gauge]
  - ["44056", "8651370"]             # Duck FRF buoy / Duck pier
  - ["41110", "8658163"]             # Masonboro Inlet buoy / Wrightsville Beach
```

Choose the offices and station pairs closest to the sensors you QC'd.

> **Where the significant events CSVs need to be.** The script looks for
> `Significant_Events_<met_year>.csv` and `Significant_Events_<met_year + 1>.csv` **in the same folder as
> `input_csv`**, not in the repository root. Copy the matching files there (e.g. into `data/`) before you
> run it. A missing file only prints a warning, but every event will then be `significant_weather = False`.

### Run

```bash
python qc_for_extreme_events.py --config path/to/your_config.yaml
```

The `--config` flag is required in practice. Its default, `config.yaml`, isn't included in the repo.

Outputs, all written to the same folder as `input_csv`:

- `qc_results_with_extreme_events.csv`: your QC results with the four columns above added. **Rejected
  events are included too**, so filter on `qc_status` when you analyze the results.
- `<office>_events_<met_year>.csv` and `nws_met_calendar_<office>_<met_year>-<met_year+1>.png`: the NWS
  advisories pulled for each office.
- `StormTable_<buoy>_<year>_1.5_8.0.csv`: the storms found at each buoy.

### Optional: master list of extreme events

`gen_extreme_events_master.py` uses the same three sources, but doesn't look at flood events. It merges
all the overlapping time windows into one timeline for the meteorological year and marks which windows
meet the 2-of-3 criteria. This is useful for listing extreme events that had no recorded flooding.

Edit `extreme_event_master_config.yaml`. It uses the same keys, except `write_path` (the output CSV path)
replaces `input_csv`. Put the `Significant_Events_*.csv` files in the same folder as `write_path`. Then
run:

```bash
python gen_extreme_events_master.py --config path/to/your_master_config.yaml
```

### Adding a new year of significant events

Create `Significant_Events_<year>.csv` with these columns:

```csv
StartDate,EndDate,Office,EventDescription
2025-10-10 00:00:00,2025-10-14 00:00:00,ILM,Coastal Storm
```

Times are in local standard time (the scripts add 5 hours to convert to UTC). `Office` must match the
WFO IDs used in `nws_offices`. Commit the new file to this repo so the next person can use it.

---

## Optional: temporal summary plots

`time_dist_summary.py` makes a 2×2 figure (`flood_summary.png`) of flood events and flood days by:

- day vs. night, based on the event start time and local sunrise and sunset
- hour of day
- day of week
- season

It counts every row in the input CSV. Give it a CSV of only the events you want counted, for example
approved events only. The settings are hard-coded for now: before running, edit `TARGET_CSV_PATH`,
`SENSOR_LAT`, and `SENSOR_LON` at the bottom of the script. Then run:

```bash
python time_dist_summary.py
```

---

## Running the helper scripts on their own

Both helper scripts work without the rest of the pipeline:

```bash
# NWS coastal/tropical advisories and calendar for one office and meteorological year
python pull_nws_advisories.py MHX 2025 --outdir ./nws_output

# Wave-driven storms for one buoy/gauge pair and calendar year
python storms_from_waves_wtr_levels.py 2025 41110 8658163 --threshold 1.5 --duration 8 --outdir ./storm_output
```

---

## Notes and caveats

- **Time zones.** The flood event CSVs and the app use fixed **EST (UTC−5) all year**, with no daylight
  saving time. The NWS advisory calendar and `time_dist_summary.py` use US Eastern local time, which does
  observe daylight saving. All matching between datasets is done in UTC.
- **Meteorological year.** NWS advisories are pulled for May 1 of `met_year` through April 30 of the next
  year. Storm tables and significant events are loaded for both full calendar years.
- **Depth units.** `max_road_water_level_(ft)` is depth above the road surface. The `(m)` column is
  converted with ft ÷ 3.28.

## Keeping data out of git

This is a public repository. Pulled data, QC results, and camera images belong in the project's data
storage, not here. Before committing, run `git status` and make sure nothing from `data/`, `static/`, or
any generated CSV or PNG is staged.

## Credits

`storms_from_waves_wtr_levels.py` is adapted from `StormsFromWaveandWaterLevel.m` by Liliana Velasquez
Montoya (2017), with later modifications by Elizabeth Farquhar.

## License

MIT. See [LICENSE](LICENSE).
