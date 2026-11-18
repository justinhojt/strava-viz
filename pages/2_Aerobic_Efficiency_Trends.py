import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

import config
from utils.data_loader import parse_csv
from utils.functions import compute_heat_index, get_steady_runs
from utils.plots import plot_aero


if 'summary_df' in st.session_state:
    summary_df = st.session_state['summary_df']
else:
    summary_df = parse_csv()

st.subheader('🫀 Aerobic Efficiency Trends')
st.write('Runs with over 15 minutes of moving time')

steady_runs = get_steady_runs(summary_df)

pace = 'Average Grade Adjusted Pace'
weather = ['temperature_2m', 'relative_humidity_2m']

eligible = steady_runs.copy()
eligible['cv_time'] = pd.to_datetime(
    eligible['Activity Date'], format='mixed', errors='coerce', utc=True
)
eligible['duration_min'] = pd.to_numeric(
    eligible['Moving Time'], errors='coerce'
) / 60

numeric_cols = [pace, 'hrr', 'duration_min', *weather]
eligible[numeric_cols] = (
    eligible.reindex(columns=numeric_cols)
    .apply(pd.to_numeric, errors='coerce')
    .replace([np.inf, -np.inf], np.nan)
)
eligible = eligible.dropna(subset=['cv_time', pace, 'hrr', 'duration_min'])

ml_df = eligible.dropna(subset=weather).copy()
ml_df['heat_index'] = (
    compute_heat_index(ml_df[weather[0]], ml_df[weather[1]])
    if not ml_df.empty else np.nan
)
ml_df['heat_index'] = pd.to_numeric(ml_df['heat_index'], errors='coerce')
ml_df = (
    ml_df.replace({'heat_index': [np.inf, -np.inf]}, np.nan)
    .dropna(subset=['heat_index'])
    .sort_values('cv_time', kind='stable')
)

n_samples = len(ml_df)
timestamp_ids, timestamps = pd.factorize(ml_df['cv_time'])
n_splits = config.ML_CV_SPLITS
model_trained = False

if n_samples >= config.ML_MIN_SAMPLES and len(timestamps) > n_splits:
    features = [pace, 'duration_min', 'heat_index']
    feature_sets = {
        'Pace only': [pace],
        'Pace + heat': [pace, 'heat_index'],
        'Pace + duration': [pace, 'duration_min'],
        'Pace + duration + heat': features,
    }
    hrr_range = config.DEFAULT_HR_MAX - config.DEFAULT_HR_REST

    fold_rows = []
    # Keep runs with identical timestamps together.
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_times, test_times in splitter.split(timestamps):
        train = ml_df.loc[np.isin(timestamp_ids, train_times)]
        test = ml_df.loc[np.isin(timestamp_ids, test_times)]
        errors = {}
        for label, columns in feature_sets.items():
            fitted = LinearRegression().fit(train[columns], train['hrr'])
            residuals = fitted.predict(test[columns]) - test['hrr']
            errors[label] = np.abs(residuals).mean() * hrr_range
        fold_rows.append(errors)

    fold_mae = pd.DataFrame(fold_rows)
    fold_mae.index = pd.RangeIndex(1, len(fold_mae) + 1, name='Fold')
    paired_gain = fold_mae['Pace + duration'] - fold_mae['Pace + duration + heat']

    X = ml_df[features]
    y = ml_df['hrr']
    model = LinearRegression().fit(X, y)
    r_squared = model.score(X, y)

    design = sm.add_constant(X, has_constant='add')
    n_parameters = design.shape[1]
    heat_ci_available = (
        n_samples > n_parameters
        and np.linalg.matrix_rank(design.to_numpy()) == n_parameters
    )
    if heat_ci_available:
        ols = sm.OLS(y, design).fit()
        coef_heat_bpm = ols.params['heat_index'] * hrr_range
        ci_low, ci_high = ols.conf_int().loc['heat_index'] * hrr_range

    hi_p10, hi_p90 = ml_df['heat_index'].quantile([0.10, 0.90])
    hi_spread = hi_p90 - hi_p10

    standard_temp = config.ML_BASELINE_TEMP_C
    standard_hum = config.ML_BASELINE_HUMIDITY
    standard_heat_index = compute_heat_index(standard_temp, standard_hum)

    X_standard = X.copy()
    X_standard['heat_index'] = standard_heat_index

    # Apply the predicted change from replacing each run's heat index.
    ml_df['adjusted_hrr'] = y + model.predict(X_standard) - model.predict(X)
    ml_df = ml_df[ml_df['adjusted_hrr'] > 0.05].copy()
    ml_df['adjusted_aero_ratio'] = ml_df[pace] / ml_df['adjusted_hrr']
    model_trained = True

use_ml = False
if model_trained:
    use_ml = st.checkbox(
        '🔮 **Show Weather-Adjusted True Fitness** '
        f'(Normalized to Singapore Baseline {config.ML_BASELINE_TEMP_C:.0f}°C / '
        f'{config.ML_BASELINE_HUMIDITY:.0f}% Humidity)',
        value=False,
    )
    st.caption(f'Chart shows the {len(ml_df)} steady-state runs with weather data.')
elif len(steady_runs) > 0:
    st.info(
        f'Model fitting needs at least {config.ML_MIN_SAMPLES} runs with '
        f'usable weather across at least {n_splits + 1} distinct timestamps. '
        f'Currently have {n_samples} runs across {len(timestamps)} timestamps.'
    )

# Use the same runs in both chart views.
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
    st.warning(
        '⚠️ No valid running rows containing both Heart Rate and Speed '
        'data were found to plot.'
    )

if model_trained:
    st.subheader('Heat index model comparison')
    st.caption(
        f'{n_samples} of {len(eligible)} eligible steady-state runs had usable '
        f'weather and heat index. All models used those same runs and the same '
        f'{n_splits} chronological folds. MAEs are in bpm.'
    )

    cv_summary = pd.DataFrame({
        'Mean CV MAE (bpm)': fold_mae.mean(),
        'Fold SD (bpm)': fold_mae.std(ddof=0),
    })
    st.dataframe(cv_summary.round(2))

    st.write(
        f'The mean CV MAE reduction from adding heat to pace + duration was '
        f'{paired_gain.mean():+.2f} bpm '
        f'(paired fold SD {paired_gain.std(ddof=0):.2f} bpm). '
        'Positive values favor adding heat.'
    )
    with st.expander('MAE by fold'):
        fold_details = fold_mae.copy()
        fold_details['Heat gain with duration'] = paired_gain
        st.dataframe(fold_details.round(2))

    st.write(
        f'Heat index ranged from {hi_p10:.2f} at the 10th percentile to '
        f'{hi_p90:.2f} at the 90th percentile, a spread of {hi_spread:.2f}° HI.'
    )

    if heat_ci_available:
        p_value = ols.pvalues['heat_index']
        st.write(
            f'Pace + duration + heat estimated {coef_heat_bpm:+.2f} bpm per '
            f'1° HI (OLS 95% CI [{ci_low:+.2f}, {ci_high:+.2f}], '
            f'p = {p_value:.3g}, in-sample R² = {r_squared:.2f}). '
            f'The fitted heart-rate difference from P10 to P90 was '
            f'{hi_spread * coef_heat_bpm:+.2f} bpm at fixed pace and duration.'
        )
        st.caption(
            'The conventional OLS interval assumes independent errors across runs.'
        )
    else:
        st.caption(
            'Estimating the heat coefficient and its interval requires '
            'independent variation in the predictors and enough runs.'
        )

with st.expander('🔬 View Aerobic Efficiency Methodology'):
    st.markdown(r"""
## 📈 Understanding Aerobic Efficiency

Aerobic efficiency measures how much physical output (speed) your body can
produce for a given cardiovascular input (heart rate).

---

### 🧮 The Calculation

This dashboard calculates efficiency for steady-state runs using this ratio.

$$
\text{Efficiency} = \frac{\text{Grade Adjusted Speed}}{\text{\% Heart Rate Reserve}}
$$

*The dashboard uses steady-state runs with over 15 minutes of moving time and
grade-adjusted metrics to limit the influence of elevation and interval spikes.*

---

### 🔮 True Fitness (Weather Normalization)

Heat and humidity can raise heart rate at a given pace and affect raw aerobic
efficiency. With **True Fitness** enabled, the chart adjusts each run's heart
rate to the configured baseline heat index using the estimated heat coefficient
from the pace, duration, and heat model.

---

### 📊 How to Read the Chart

* **Upward Trend ↗️** may reflect an improved ability to hold the same pace at a
  lower heart rate.
* **Downward Trend ↘️** may reflect accumulated fatigue, a loss of fitness, or
  environmental factors such as heat stress.
* **Daily Variance 📉📈** can reflect sleep, caffeine intake, ambient temperature,
  and hydration.
""")
