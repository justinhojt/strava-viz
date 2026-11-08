# Strava Archive Analytics Dashboard

**Live demo:** https://justinhojt-stravaviz.streamlit.app/

A locally-hosted Streamlit dashboard that turns a raw Strava bulk export into a set of physiological training insights: cardiovascular adaptation, aerobic efficiency, and long-term fitness/fatigue trends.

---

## Key Features

### Home — Lifetime Overview
A macro view of your entire training history: total activities, distance, moving time, calories, elevation gain, longest activity, and weekly consistency averages (activities/week, moving time/week). Includes an interactive activity-type breakdown donut chart, filterable by activity type.

### Activity Viewer
Drill into second-by-second data for any individual session. Parses `.gpx` and `.fit` (including gzipped variants) to render:
- A time-in-zone heart rate breakdown (5 zones, based on % of max HR)
- Heart rate and elevation line charts over the course of the activity
- Activity-specific KPIs (pace and distance for runs/rides, 100m pace for swims, a flat time-based intensity score for weight training)

### Aerobic Efficiency Trends
Isolates steady-state runs (≥15 min moving time) and plots an efficiency ratio (grade-adjusted pace over average heart rate) with a 42-day rolling trend line, so you can see cardiovascular adaptation independent of day-to-day pacing choices.

- **True Fitness (ML-adjusted view):** once you have 50+ steady-state runs with matched weather data, a linear regression model learns your personal heat stress penalty (bpm increase per 1° of heat index) and normalizes every run's heart rate to a standard 28°C / 80% humidity baseline, so adverse weather conditions do not mask genuine fitness gains. Temperature and humidity are combined into a single heat index feature (Rothfusz regression) rather than modeled as two separate variables, since they tend to be collinear in consistently warm/humid climates and their physiological effect on heart rate is multiplicative, not additive.
- **Note:** the efficiency ratio's directionality (higher = fitter) assumes the underlying "pace" field increases with speed. If your export's pace column instead decreases as you get faster (i.e. is a true time-per-distance pace), the trend line's interpretation should be flipped accordingly.

### Fitness, Fatigue & Form
Implements the Banister TRIMP (Training Impulse) model:
| Metric | Full Name | Window | Meaning |
|---|---|---|---|
| **CTL** | Chronic Training Load | 42-day EWMA | Fitness |
| **ATL** | Acute Training Load | 7-day EWMA | Fatigue |
| **TSB** | Training Stress Balance | `CTL − ATL` | Form / readiness |

A date slider lets you inspect CTL/ATL/TSB for any day in your training history, with TSB plotted against four zones: Freshness, Maintenance, Optimal Training, and Overtraining Risk.

---

## Data Pipeline

The dashboard reads from a pre-processed CSV (`activities_refined.csv`), built once by `refine_dataset.py` from your raw Strava export:

1. **Clean** the raw `activities.csv` — drops rows with no linked activity file, removes constant/duplicate columns.
2. **Localize timestamps** — extracts the first GPS coordinate from each activity file (without loading the whole file) to resolve the activity's true local timezone, falling back to a fixed offset when no GPS data exists.
3. **Fetch historical weather** — for runs with GPS data, queries the Open-Meteo archive API and computes a duration-weighted average of temperature, humidity, and wind speed across every hour the run actually spanned (rather than a single hourly snapshot at the start time), so multi-hour runs crossing an hour boundary get accurately blended conditions.
4. **Classify workout style** — runs are tagged `Steady State` or `Interval` based on the ratio of moving time to elapsed time.

Re-run `refine_dataset.py` whenever you add a new Strava export.

---

## Repository Structure

All UI styling and constants are centralized in `config.py`, and frontend pages are decoupled from backend calculations and charting logic.

```text
├── data/                               # Local Strava export data
│   ├── activities/                     # Individual gpx/fit files
│   ├── activities.csv                  # Raw Strava bulk export summary
│   └── activities_refined.csv          # Cleaned dataset with weather & localized timezones
├── pages/                              # Frontend Streamlit sub-pages
│   ├── 1_Activity_Viewer.py
│   ├── 2_Aerobic_Efficiency_Trends.py
│   └── 3_Fitness_Fatigue_Form.py
├── utils/                              # Core pipeline logic
│   ├── data_loader.py                  # gpx/fit parsing
│   ├── functions.py                    # TRIMP score calculations and data transformations
│   └── plots.py                        # Centralized Altair charting components
├── .gitignore
├── Home.py                             # Main Streamlit entry point
├── README.md
├── config.py                           # Centralized configuration for UI colors, HR zones, and other constants
├── refine_dataset.py                   # Pre-processing script: cleans data, standardizes timezones, and fetches historical weather
└── requirements.txt                    # Python dependencies
```

---

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Export your Strava data** — request your bulk archive from Strava, then place the export's `activities.csv` and activity files folder into `data/` (matching the structure above).
3. **Build the refined dataset**
   ```bash
   python refine_dataset.py
   ```
   This produces `data/activities_refined.csv`. Re-run it any time you add new activities.
4. **Launch the dashboard**
   ```bash
   streamlit run Home.py
   ```

---

## Configuration

Physiological constants in `config.py` drive the TRIMP/CTL/ATL/TSB calculations and should be adjusted to reflect you personally, rather than left at their defaults:

- `DEFAULT_HR_MAX`, `DEFAULT_HR_REST`, `DEFAULT_GENDER` — used in the Banister TRIMP formula; inaccurate values will skew every training-load metric in the dashboard.
- `CTL_DAY_SPAN`, `ATL_DAY_SPAN` — the rolling windows (in days) for fitness and fatigue; 42/7 are standard defaults but can be tuned.
- Colors and heart rate zone boundaries are also centralized here for consistent theming across all charts.

**Caveats:**
- Weather enrichment (and therefore the ML "True Fitness" view) only applies to `Run` activities with GPS data.
- Weight Training / Workout sessions don't get a heart-rate-derived TRIMP score — the model instead assumes a flat 40 TRIMP/hour, since HR tends to underestimate strength-training effort.
- **The heat coefficient is associative, not causal.** It estimates how heart rate relates to heat index at a given pace, but it may also absorb other things that vary alongside heat, such as fitness changes over time or time of day (morning vs evening runs). Treat it as a rough personal estimate, not a precise physiological constant.
