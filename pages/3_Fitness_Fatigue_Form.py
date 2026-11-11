import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from utils.data_loader import parse_csv
from utils.functions import parse_granular
from utils.plots import plot_fitness_fatigue, plot_tsb_zones
import config

# Fetch the shared dataset from session state if available, else load it in
if 'summary_df' in st.session_state:
    summary_df = st.session_state['summary_df']
else:
    summary_df = parse_csv()

trimps = parse_granular(summary_df.copy())

st.subheader('Historical Performance Analysis')

# Date slider
min_date = trimps['Date'].min().to_pydatetime() # Streamlit uses python datetime instead of pandas datetime
max_date = summary_df['Activity Date'].max().to_pydatetime()

selected_date = st.slider(
    '📅 Use the slider to view stats for that day:',
    min_value=min_date,
    max_value=max_date,
    value=max_date,  
    format='YYYY-MM-DD'
)

# Filter the metrics down to the single selected day
matched_row = trimps[trimps['Date'] == pd.Timestamp(selected_date)]

if not matched_row.empty:
    metrics = matched_row.iloc[0]
    ctl = metrics['CTL']
    atl = metrics['ATL']
    tsb = metrics['TSB']
    
    # Dynamic KPI cards
    col1, col2, col3 = st.columns(3)
    col1.metric('Fitness (CTL)', f'{ctl:.2f}')
    col2.metric('Fatigue (ATL)', f'{atl:.2f}')
    
    if tsb < config.TSB_OVERTRAINING_UPPER:
        status, color = 'Overtraining Risk', 'inverse'
    elif config.TSB_OVERTRAINING_UPPER <= tsb < config.TSB_OPTIMAL_UPPER:
        status, color = 'Optimal Training', 'normal'
    elif config.TSB_OPTIMAL_UPPER <= tsb <= config.TSB_MAINTENANCE_UPPER:
        status, color = 'Maintenance Zone', 'normal'
    else:
        status, color = 'Fresh / Recovery', 'normal'
        
    col3.metric('Form (TSB)', f'{tsb:.2f}', delta=status, delta_color=color)

st.markdown('---')

# Charts and documentation
chart_tab1, chart_tab2 = st.tabs(['📊 Fitness & Fatigue Dynamics', '🚦 Training Stress Balance (TSB)'])

with chart_tab1:
    st.altair_chart(plot_fitness_fatigue(trimps, selected_date), width='stretch') 
    
with chart_tab2:
    st.altair_chart(plot_tsb_zones(trimps, selected_date), width='stretch')

st.markdown('---')

# ---------------------------------------------------------------
# Does form predict efficiency?
# ---------------------------------------------------------------
st.subheader('🔗 Does Form Predict Efficiency?')
st.write(
    "Do you run more efficiently when fresh? This tests whether the previous day's training state "
    f'improves on a {config.BASELINE_WINDOW} rolling average at predicting a steady-state run\'s aerobic efficiency.'
)

# TSB in parse_granular is already built from the previous day's CTL/ATL (shifted by 1),
# so it can be used as is. CTL is not shifted, so we lag it ourselves to avoid
# including the run's own training load.
predictor_options = {'Form (TSB)': 'TSB', 'Fitness (CTL)': 'CTL_prev'}
predictor_label = st.radio('Predictor', list(predictor_options.keys()), horizontal=True)
predictor = predictor_options[predictor_label]

form_df = trimps[['Date', 'TSB', 'CTL']].copy()
form_df['CTL_prev'] = form_df['CTL'].shift(1)

# Steady-state runs, same filters and efficiency ratio as the Aerobic Efficiency page
runs = summary_df[(summary_df['Activity Type'] == 'Run') & (summary_df['workout_style'] == 'Steady State')].copy()
runs = runs[(runs['Average Grade Adjusted Pace'] > 0) & 
            (runs['Average Heart Rate'] > 0) & 
            (runs['Moving Time'] >= 900)]

runs['hrr'] = (runs['Average Heart Rate'] - config.DEFAULT_HR_REST) / (config.DEFAULT_HR_MAX - config.DEFAULT_HR_REST)
runs = runs[runs['hrr'] > 0.05]
runs['aero_ratio'] = runs['Average Grade Adjusted Pace'] / runs['hrr']
runs = runs.sort_values('Activity Date')

# Baseline: average efficiency over the previous 42 days (closed='left' excludes the run itself)
runs['baseline'] = runs.rolling(config.BASELINE_WINDOW, on='Activity Date', closed='left')['aero_ratio'].mean()
runs['Date'] = runs['Activity Date'].dt.normalize()

# Attach form from the day before each run
df = runs.merge(form_df[['Date', predictor]], on='Date', how='inner')
df = df.dropna(subset=['baseline', predictor]).sort_values('Activity Date').reset_index(drop=True)

# Predict how far a run lands from its rolling average (the baseline predicts 0)
df['residual'] = df['aero_ratio'] - df['baseline']

if len(df) < config.MIN_RUNS:
    st.info(f'Need at least {config.MIN_RUNS} steady-state runs with form data to test this (currently {len(df)}).')
else:
    X = df[[predictor]]
    y = df['residual']

    # Time-series cross-validation: always train on the past, test on the future
    model_maes, base_maes = [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=config.ML_CV_SPLITS).split(df):
        model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])

        model_maes.append(np.mean(np.abs(preds - y.iloc[test_idx])))
        base_maes.append(np.mean(np.abs(y.iloc[test_idx])))    # baseline error = |actual - rolling average|

    cv_mae, cv_std = np.mean(model_maes), np.std(model_maes)
    base_mae = np.mean(base_maes)
    improvement = (base_mae - cv_mae) / base_mae * 100

    # Full-sample OLS for the slope and its confidence interval
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    slope_10 = ols.params[predictor] * 10                          # change in efficiency per +10 units
    ci_low, ci_high = ols.conf_int().loc[predictor] * 10

    m1, m2, m3 = st.columns(3)
    m1.metric(f'Effect of +10 {predictor_label.split()[-1]}', f'{slope_10:+.3f}',
              delta=f'95% CI [{ci_low:.3f}, {ci_high:.3f}]', delta_color='off', delta_arrow='off')
    m2.metric('CV MAE (with form)', f'{cv_mae:.3f} ± {cv_std:.3f}',
              delta=f'rolling avg only: {base_mae:.3f}', delta_color='off', delta_arrow='off')
    m3.metric('Runs analysed', f'{len(df)}')

    if improvement > 0:
        st.success(f'Adding {predictor_label} reduced prediction error by {improvement:.1f}% versus the rolling average alone.')
    else:
        st.info(f'Adding {predictor_label} did not beat the rolling average on held-out runs, so there is no clear evidence that it predicts efficiency.')

    st.caption(
        'Efficiency is measured as the gap between a run and its previous 42-day average, so the test isolates the effect of '
        'form from long-term fitness trends. Errors are measured on held-out later runs (time-series CV). '
        'The slope is associative, not causal, and the CI is likely too narrow because consecutive runs are not independent.'
    )

    st.scatter_chart(df, x=predictor, y='residual', x_label=predictor_label, y_label='Efficiency vs 42-day average')

with st.expander('🔬 View Physiological Model Methodology'):
    st.markdown("""
    ## 📊 Understanding Fitness, Fatigue, and Form

    Training effectively requires balancing hard work with structured recovery. These metrics, derived from the Banister Impulse-Response model, transform raw workout data into a predictive view of an individual's physiology.
    
    ---
    
    ### 📈 The Core Metrics
    At the heart of the graphs are three primary metrics that track training load:
    
    | Metric | Full Name | Time Horizon | What it means |
    | :--- | :--- | :--- | :--- |
    | **CTL** | Chronic Training Load | ~42 Days | Fitness (long-term adaptation) |
    | **ATL** | Acute Training Load | ~7 Days | Fatigue (short-term exhaustion) |
    | **TSB** | Training Stress Balance | Daily | Form (readiness to perform) |
    
    ---
    
    ### 🔄 How They Interact
    The charts visualize the continuous tug-of-war between fitness and fatigue, with form informing us of the current dominating side.
    
    #### 1. Fitness (CTL)
    CTL represents the long-term trend of training volume and intensity. Because it is calculated as a 42-day exponentially weighted moving average (EWMA), it changes slowly.
    
    #### 2. Fatigue (ATL)
    ATL is highly volatile. Representing a short 7-day EWMA, it reacts immediately to hard workouts or back-to-back training days. If ATL climbs sharply above CTL, fatigue is accumulating rapidly.
    
    #### 3. Form (TSB)
    Form is the mathematical difference between current fitness and current fatigue:
    
    $$TSB = CTL - ATL$$
    
    * **When TSB > 0:** State of freshness; ideal for a race or performance test.
    * **When TSB < 0:** Accumulating training stress. While a negative number is required to stimulate adaptation and build fitness, staying deeply negative for too long risks injury or burnout.
    
    ### 🚦 Reading the Training Zones
    
    * **🟢 Freshness (Above 0):** Minimal fatigue; peak state for racing or performance testing.
    * **⚫ Maintenance Zone (0 to -10):** Fitness is maintained but not actively building.
    * **🟠 Optimal Training (-10 to -30):** Productive training stress with managed fatigue. The "sweet spot" for building fitness safely.
    * **🔴 Overtraining (Below -30):** High risk of injury, illness, or chronic fatigue. 
    
    ---
    
    ### 🔗 Does Form Predict Efficiency?
    For each steady-state run, the baseline predicts that efficiency equals the average of the previous 42 days of runs. A linear regression then tries to predict the gap from that average using the previous day's TSB (or CTL). If the regression has lower error on held-out later runs, form carries useful information about how efficiently you run.
    
    """)
