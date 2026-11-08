from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import streamlit as st
import numpy as np
import statsmodels.api as sm

from utils.data_loader import parse_csv
from utils.functions import compute_heat_index
from utils.plots import plot_aero
import config

# Fetch the shared dataset from session state if available, else load it in
if 'summary_df' in st.session_state:
    summary_df = st.session_state['summary_df']
else:
    summary_df = parse_csv()

st.subheader('🫀 Aerobic Efficiency Trends')
st.write('Runs with over 15 minutes of moving time')

runs = summary_df[summary_df['Activity Type'] == 'Run'].copy()
steady_runs = runs[runs['workout_style'] == 'Steady State'].copy()

# Filter for valid pacing/HR and minimum moving time
steady_runs = steady_runs[(steady_runs['Average Grade Adjusted Pace'] > 0) & 
                          (steady_runs['Average Heart Rate'] > 0) & 
                          (steady_runs['Moving Time'] >= 900)]

# Heart Rate Reserve (HRR) corrects for resting HR offset so effort = 0 maps to denominator = 0,
# instead of dividing directly by raw avg_hr which has a nonzero floor at rest
hr_rest = config.DEFAULT_HR_REST
hr_max = config.DEFAULT_HR_MAX

steady_runs['hrr'] = (steady_runs['Average Heart Rate'] - hr_rest) / (hr_max - hr_rest)
steady_runs = steady_runs[steady_runs['hrr'] > 0.05]    # Removes outliers where avg HR during activity is only slightly above rest, likely due to sensor error 

# Calculate raw efficiency using %HRR instead of raw average heart rate
steady_runs['aero_ratio'] = steady_runs['Average Grade Adjusted Pace'] / steady_runs['hrr']

# Filter out runs missing weather data for the ML model
ml_df = steady_runs.dropna(subset=['Average Grade Adjusted Pace', 'hrr', 'temperature_2m', 'relative_humidity_2m']).copy()

ml_df['heat_index'] = compute_heat_index(ml_df['temperature_2m'], ml_df['relative_humidity_2m'])
ml_df = ml_df.sort_values('Activity Date')

# Initialize ML state variables
has_enough_data = (len(ml_df) >= config.ML_MIN_SAMPLES)
model_trained = False

if has_enough_data:
    features = ['Average Grade Adjusted Pace', 'heat_index']
    base_features = ['Average Grade Adjusted Pace']    # pace only
    hrr_range = hr_max - hr_rest
    
    maes, base_maes = [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=config.ML_CV_SPLITS).split(ml_df):
        train, test = ml_df.iloc[train_idx], ml_df.iloc[test_idx]
    
        full = LinearRegression().fit(train[features], train['hrr'])
        base = LinearRegression().fit(train[base_features], train['hrr'])
    
        maes.append(np.mean(np.abs(full.predict(test[features]) - test['hrr'])) * hrr_range)
        base_maes.append(np.mean(np.abs(base.predict(test[base_features]) - test['hrr'])) * hrr_range)
    
    cv_mae, cv_std = np.mean(maes), np.std(maes)
    base_mae = np.mean(base_maes)

    X = ml_df[features]
    y = ml_df['hrr']
    
    model = LinearRegression()
    model.fit(X, y)

    ols = sm.OLS(ml_df['hrr'], sm.add_constant(ml_df[features])).fit()
    
    coef_heat_bpm = ols.params['heat_index'] * hrr_range
    ci_low, ci_high = ols.conf_int().loc['heat_index'] * hrr_range
        
    r_squared = model.score(X, y)
    n_samples = len(ml_df)
    
    # Define standard environmental baseline (28°C and 80% Humidity)
    standard_temp = config.ML_BASELINE_TEMP_C
    standard_hum = config.ML_BASELINE_HUMIDITY
    standard_heat_index = compute_heat_index(standard_temp, standard_hum)
    
    # Create a hypothetical feature set: actual pace, but standard heat index
    X_standard = X.copy()
    X_standard['heat_index'] = standard_heat_index
    
    # Shift each run's actual %HRR by the modelled heat effect (keeps its residual)
    ml_df['adjusted_hrr'] = ml_df['hrr'] + (model.predict(X_standard) - model.predict(X))
    ml_df = ml_df[ml_df['adjusted_hrr'] > 0.05].copy()    # drop runs where the adjusted value is ~0 or negative
    
    # Recalculate efficiency using the normalized %HRR
    ml_df['adjusted_aero_ratio'] = ml_df['Average Grade Adjusted Pace'] / ml_df['adjusted_hrr']
    model_trained = True

# Only show the ML toggle if we actually have enough data to train the model safely
use_ml = False
if model_trained:
    use_ml = st.checkbox(f'🔮 **Show Weather-Adjusted True Fitness** (Normalized to Singapore Baseline {config.ML_BASELINE_TEMP_C:.0f}°C / {config.ML_BASELINE_HUMIDITY:.0f}% Humidity)', value=False)
    st.caption(f'Chart shows the {len(ml_df)} steady-state runs with weather data.')
elif len(steady_runs) > 0:
    st.info(f'Keep running! You need at least {config.ML_MIN_SAMPLES} steady-state runs with weather data to unlock ML True Fitness tracking. (Currently have {len(ml_df)})')

# Use the same runs in both views so the toggle only changes the metric
if model_trained:
    plot_df = ml_df.copy()
    if use_ml:
        plot_df['aero_ratio'] = plot_df['adjusted_aero_ratio']
else:
    plot_df = steady_runs.copy()

run_chart_data = (
    plot_df.dropna(subset=['Activity Date', 'aero_ratio'])
    .sort_values('Activity Date')
    .copy()
)

if not run_chart_data.empty:
    st.altair_chart(plot_aero(run_chart_data), width='stretch')
else:
    st.warning('⚠️ No valid running rows containing both Heart Rate and Speed data were found to plot.')

# Personal environmental penalty
if model_trained:
    st.subheader('🧬 Personal Environmental Profile')
    st.write('Using linear regression against heat index (temperature + humidity combined), we can analyse how a specific body reacts to overall heat stress.')
    
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    prof_col1.metric('🌡️ Heat Penalty (per 1° HI)', f'{coef_heat_bpm:+.2f} bpm',
                     delta=f'95% CI [{ci_low:.2f}, {ci_high:.2f}]', delta_color='off')
    prof_col2.metric('🎯 CV MAE (full model)', f'{cv_mae:.1f} ± {cv_std:.1f} bpm',
                     delta=f'pace-only: {base_mae:.1f} bpm', delta_color='off')
    prof_col3.metric('📊 R² (in-sample)', f'{r_squared:.2f}')

    st.caption(
    f'Model fit on {n_samples} runs; chart shows {len(ml_df)} after dropping runs with adjusted %HRR ≤ 5%. R² is in-sample (fit on all runs).'
    'CV MAE is the average error on held-out later runs (time-series CV), compared against a pace-only baseline.'
    'The CI is a 95% interval on the heat coefficient. Heat index combines temperature and humidity into one value.'
    )
    
# Methodology Expander
with st.expander('🔬 View Aerobic Efficiency Methodology'):
    st.markdown("""
    ## 📈 Understanding Aerobic Efficiency

    Aerobic efficiency measures how much physical output (speed) your body can produce for a given cardiovascular input (heart rate).
    
    ---
    
    ### 🧮 The Calculation
    This dashboard calculates efficiency for steady-state runs using the following ratio:
    
    $$ \\text{Efficiency} = \\frac{\\text{Grade Adjusted Speed}}{\\text{\\% Heart Rate Reserve}} $$
    
    *Note: We specifically filter for "Steady State" runs with over 15 minutes of moving time and use Grade Adjusted metrics to ensure elevation changes and interval spikes do not heavily skew the data.*
    
    ---
    
    ### 🔮 True Fitness (Weather Normalization)
    If you live in an environment with distinct seasons or high heat, your raw efficiency will artificially drop in the summer due to cardiac drift, masking your actual fitness gains.
    
    When **True Fitness** is enabled, a machine learning model isolates your pace and adjusts each run’s heart rate by the estimated heat effect, as if it had been run at a standard heat index of 28°C at 80% humidity — combining temperature and humidity into one "feels like" value rather than treating them as independent factors.
    
    ---
    
    ### 📊 How to Read the Chart
    
    * **Upward Trend ↗️:** Cardiovascular adaptation is occurring. You are getting fitter and can hold the same pace at a lower heart rate.
    * **Downward Trend ↘️:** This can indicate accumulated fatigue, a loss of fitness, or external environmental factors like severe heat stress.
    * **Daily Variance 📉📈:** Factors like sleep quality, caffeine intake, ambient temperature, and hydration will cause daily fluctuations.
    """)
