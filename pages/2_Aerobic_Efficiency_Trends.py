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
    pace_heat_gain = fold_mae['Pace only'] - fold_mae['Pace + heat']
    duration_heat_gain = fold_mae['Pace + duration'] - fold_mae['Pace + duration + heat']

    X = ml_df[features]
    y = ml_df['hrr']
    model = LinearRegression().fit(X, y)

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
    predicted_change = model.predict(X_standard) - model.predict(X)
    ml_df['adjusted_hrr'] = y + predicted_change
    ml_df = ml_df[ml_df['adjusted_hrr'] > 0.05].copy()
    ml_df['adjusted_aero_ratio'] = ml_df[pace] / ml_df['adjusted_hrr']
    model_trained = True

use_ml = False
if model_trained:
    use_ml = st.checkbox(
        'Show weather-adjusted aerobic efficiency',
        value=False,
        help=(
            f'The reference weather is {standard_temp:.0f}°C '
            f'and {standard_hum:.0f}% relative humidity.'
        ),
    )
elif not steady_runs.empty:
    st.info(
        f'Weather adjustment needs {config.ML_MIN_SAMPLES} runs with usable weather '
        f'across at least {n_splits + 1} distinct start times. '
        f'Available so far: {n_samples} runs across {len(timestamps)} start times.'
    )

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
    if model_trained:
        st.caption(f'Both chart views show the same {len(run_chart_data)} runs.')
else:
    st.warning('No steady-state runs had enough data to plot aerobic efficiency.')

if model_trained:
    st.subheader('Heat index model comparison')

    verdict = []
    if heat_ci_available:
        verdict.append(
            f'At the same pace and duration, a 1° rise in heat index was associated '
            f'with a **{coef_heat_bpm:+.2f} bpm** change in heart rate.'
        )
        if ci_low <= 0 <= ci_high:
            verdict.append('The estimated heat effect is uncertain.')
    else:
        verdict.append(
            'These runs provide too little information to estimate the heat effect.'
        )

    mean_heat_gain = round(duration_heat_gain.mean(), 2)
    if mean_heat_gain > 0:
        verdict.append(
            f'Adding heat to pace and duration lowered average prediction error on '
            f'later runs by **{mean_heat_gain:.2f} bpm**.'
        )
    elif mean_heat_gain < 0:
        verdict.append(
            f'Adding heat to pace and duration raised average prediction error on '
            f'later runs by **{abs(mean_heat_gain):.2f} bpm**.'
        )
    else:
        verdict.append(
            'Adding heat to pace and duration left average prediction error '
            'on later runs essentially unchanged.'
        )
    st.markdown(' '.join(verdict))

    run_count = f'{n_samples} steady-state runs'
    if n_samples < len(eligible):
        run_count += f' with usable weather ({len(eligible)} eligible runs)'
    st.caption(
        f'All models used the same {run_count} and {len(fold_mae)} chronological folds. '
        'Lower prediction error is better.'
    )

    mean_mae = fold_mae.mean()
    comparison_rows = []
    for baseline, with_heat, gains in (
        ('Pace only', 'Pace + heat', pace_heat_gain),
        ('Pace + duration', 'Pace + duration + heat', duration_heat_gain),
    ):
        folds_helped = gains.round(2).gt(0).sum()
        comparison_rows.append({
            'Baseline model': baseline,
            'Without heat (bpm)': f'{mean_mae[baseline]:.2f}',
            'With heat (bpm)': f'{mean_mae[with_heat]:.2f}',
            'Folds where heat helped': f'{folds_helped} of {len(fold_mae)}',
        })
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True)

    st.markdown(
        f'**Duration.** Adding duration to pace changed average prediction error '
        f'from **{mean_mae["Pace only"]:.2f} to {mean_mae["Pace + duration"]:.2f} bpm**. '
        'Duration may capture cardiac drift (heart rate gradually rising during a run) '
        'or differences in run type and pacing.'
    )

    with st.expander('Fold results and heat estimate'):
        st.caption(
            'Each fold tests on runs later than its training runs. '
            'The table shows mean absolute prediction error in bpm.'
        )
        st.dataframe(fold_mae.round(2))

        st.write(
            f'The middle 80% of heat-index values ran from {hi_p10:.2f} to '
            f'{hi_p90:.2f}° HI, a spread of {hi_spread:.2f}° HI.'
        )
        if heat_ci_available:
            st.write(
                f'Heat coefficient with pace and duration held fixed was '
                f'{coef_heat_bpm:+.2f} bpm per 1° HI '
                f'(OLS 95% CI [{ci_low:+.2f}, {ci_high:+.2f}]). '
                f'The fitted heart-rate difference across that heat-index range was '
                f'{hi_spread * coef_heat_bpm:+.2f} bpm.'
            )
            st.caption(
                'The conventional OLS interval assumes independent errors across runs.'
            )

with st.expander('Aerobic efficiency methodology'):
    st.markdown(r'''
Aerobic efficiency describes the grade-adjusted speed a runner can sustain
for a given share of their heart-rate reserve.

$$
\text{Efficiency} = \frac{\text{Grade-adjusted speed}}{\text{\% Heart-rate reserve}}
$$

The chart uses steady-state runs with more than 15 minutes of moving time.
The weather-adjusted view estimates each run's heart rate at the reference
heat index with its pace and duration held fixed. Heat index combines air
temperature and humidity.

A rising trend can reflect improved aerobic fitness. Fatigue, recovery,
hydration, and weather can also affect individual runs.
''')
