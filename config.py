import os

DATA_DIR = 'data'
ACTIVITIES_CSV = os.path.join(DATA_DIR, 'activities_refined.csv')

# UI colour hex codes
COLOR_STRAVA_ORANGE = '#fc5200'
COLOR_MUTED_GREY = '#808080'
COLOR_PURE_WHITE = '#ffffff'

# Performance Metrics Lines
COLOR_FITNESS = '#1f77b4'        # Muted blue
COLOR_FATIGUE = '#ff7f0e'        # Muted orange

# Heart Rate Zone Breakdown
HR_ZONE_COLORS = ['#95a5a6', '#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
HR_ZONE_LABELS = ['Z1 Recovery', 'Z2 Aerobic', 'Z3 Tempo', 'Z4 Threshold', 'Z5 Anaerobic']
HR_ZONE_PCT_BOUNDARIES = [0, 0.60, 0.70, 0.80, 0.90, 1.50]  # % of HR max; last bound is an open-ended cap

# Activity composition breakdown (donut chart)
ACTIVITY_COLORS = {
    'Run': '#fc5200',              
    'Walk': '#ee9f28',             
    'Swim': '#f9dcb0',             
    'Workout': '#e17602',
    'Weight Training': '#c96f2e',
    'Default': '#808080'         
}

# TSB training zones
ZONE_COLORS = {
    'Freshness': '#6cf7a1',          # TSB > 0 (Green)
    'Maintenance': '#808080',        # TSB 0 to -10 (Grey)
    'Optimal_Training': '#ff9955',   # TSB -10 to -30 (Orange)
    'Overtraining': '#f44e65'        # TSB < -30 (Red)
}

# TSB zone breakpoints 
TSB_MAINTENANCE_UPPER = 0
TSB_OPTIMAL_UPPER = -10
TSB_OVERTRAINING_UPPER = -30

# Default cardiovascular metrics
DEFAULT_HR_MAX = 200
DEFAULT_HR_REST = 75
DEFAULT_GENDER = 'male'

# Banister Impulse-Response Model Constants (Rolling window spans in days)
CTL_DAY_SPAN = 42   # Chronic Training Load (Fitness)
ATL_DAY_SPAN = 7    # Acute Training Load (Fatigue)

MIN_RUNS = 30                # Minimum runs needed for a meaningful CV
BASELINE_WINDOW = '42D'      # Rolling average window for the baseline

# Aerobic Efficiency / Heat Stress ML Model
ML_MIN_SAMPLES = 50           # Minimum steady-state runs with weather data required to train the model
ML_BASELINE_TEMP_C = 28.0     # Standard heat-index normalization baseline (Singapore climate)
ML_BASELINE_HUMIDITY = 80.0
ML_CV_SPLITS = 4

TIMEZONE_TARGET = 'Asia/Singapore'
TIMEZONE_OFFSET_HOURS = 8

# K-means run clustering
KMEANS_MAX_K = 6        # silhouette loop tries k = 2..KMEANS_MAX_K
KMEANS_MIN_RUNS = 20    # minimum runs before clustering is worthwhile
