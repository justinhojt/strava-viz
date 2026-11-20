import os


DATA_DIR = 'data'
ACTIVITIES_CSV = os.path.join(DATA_DIR, 'activities_refined.csv')

# Brand
ACCENT = '#FC4C02'
CHART_SECONDARY = '#4D769B'
CHART_FONT = 'Inter, sans-serif'

# Chart neutrals and table colors. Text defaults match Streamlit's built-in themes.
CHART_NEUTRALS = {
    'light': {
        'text': '#31333F',
        'muted': '#596F80',
        'grid': '#D3DAE2',
        'table_header': '#EAEFF4',
        'table_border': '#D7DEE6',
        'table_highlight': '#FCE4D4',
        'table_highlight_text': '#85340F',
    },
    'dark': {
        'text': '#FAFAFA',
        'muted': '#9BAFC0',
        'grid': '#394451',
        'table_header': '#26303D',
        'table_border': '#3F4B59',
        'table_highlight': '#633923',
        'table_highlight_text': '#FFE0C8',
    },
}

# Semantic palettes
ACTIVITY_MIX_COLORS = {
    'Run': ACCENT,
    'Workout': '#96340F',
    'Walk': '#D98B2B',
    'Swim': '#F5BB86',
}
ACTIVITY_MIX_EXTRA_COLORS = ('#BC5C28', '#74351D', '#F08A53', '#E7A15D')

HR_ZONE_COLORS = ['#687486', CHART_SECONDARY, '#527B68', '#C88618', '#B94156']
HR_ZONE_LABELS = ['Z1 Recovery', 'Z2 Aerobic', 'Z3 Tempo', 'Z4 Threshold', 'Z5 Anaerobic']
HR_ZONE_PCT_BOUNDARIES = [0, 0.60, 0.70, 0.80, 0.90, 1.50]  # Fractions of HR max

TSB_ZONE_COLORS = {
    'Fresh / Recovery': '#38784F',
    'Maintenance Zone': '#606D77',
    'Optimal Training': '#C88618',
    'Overtraining Risk': '#B94156',
}
TSB_MAINTENANCE_UPPER = 0
TSB_OPTIMAL_UPPER = -10
TSB_OVERTRAINING_UPPER = -30

LOAD_LINE_COLORS = (CHART_SECONDARY, ACCENT)  # CTL, ATL

# Categorical series
CHART_CATEGORICAL_COLORS = (
    ACCENT, CHART_SECONDARY, '#527B68', '#8B709C',
    '#9C784C', '#687486', '#52818D', '#A66E82',
)

# Cardiovascular defaults and training-load windows
DEFAULT_HR_MAX = 200
DEFAULT_HR_REST = 75
DEFAULT_GENDER = 'male'
CTL_DAY_SPAN = 42
ATL_DAY_SPAN = 7
MIN_RUNS = 30
BASELINE_WINDOW = '42D'

# Aerobic efficiency and weather adjustment
ML_MIN_SAMPLES = 50
ML_BASELINE_TEMP_C = 28.0
ML_BASELINE_HUMIDITY = 80.0
ML_CV_SPLITS = 4

TIMEZONE_TARGET = 'Asia/Singapore'
TIMEZONE_OFFSET_HOURS = 8

# Run clustering
KMEANS_MAX_K = 6
KMEANS_MIN_RUNS = 20
