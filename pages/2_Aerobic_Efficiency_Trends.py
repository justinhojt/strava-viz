import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

import config
from utils.data_loader import get_summary
from utils.functions import (
    clean_numeric,
    compute_heat_index,
    get_steady_runs,
    heart_rate_reserve,
)
from utils.plots import get_theme, plot_aero, plot_heat_effect, plot_prediction_error
from utils.ui import metric_cards, styled_table


PACE = 'Average Grade Adjusted Pace'
WEATHER = ['temperature_2m', 'relative_humidity_2m']
FULL = 'Pace + duration + heat'
FEATURE_SETS = {
    'Pace only': [PACE],
    'Pace + heat': [PACE, 'heat_index'],
    'Pace + duration': [PACE, 'duration_min'],
    FULL: [PACE, 'duration_min', 'heat_index'],
}
HRR_RANGE = heart_rate_reserve()
HI = '°HI'    # the one heat-index unit used on this page (matches the plot axis)


def heat_verdict(coef, low, high, gain, helped, folds):
    """Plain-language summary of the heat effect, without overclaiming."""
    size = f'{coef:+.2f} bpm per {HI}'
    if gain > 0:
        cv = f'Adding heat lowered held-out error by {gain:.2f} bpm (better in {helped} of {folds} folds).'
    else:
        cv = f'Adding heat raised held-out error by {-gain:.2f} bpm (better in {helped} of {folds} folds).'

    if low > 0 or high < 0:
        direction = 'higher' if coef > 0 else 'lower'
        return (
            f'Hotter runs were associated with a {direction} heart rate at the same pace and '
            f'duration ({size}; 95% interval excludes zero). {cv}'
        )
    return (
        f'The heat effect is small and not clearly distinguishable from zero ({size}; '
        f'95% interval includes zero). {cv} Treat the weather adjustment as modest.'
    )


st.title('Aerobic Efficiency Trends')
st.caption('Steady-state runs with at least 15 minutes of moving time.')

summary_df = get_summary()
if summary_df.empty:
    st.info('No activities are available.')
    st.stop()

steady_runs = get_steady_runs(summary_df)
if steady_runs.empty:
    st.info('No steady-state runs had enough data to plot aerobic efficiency.')
    st.stop()

theme = get_theme()

# ---------- Prepare eligible runs ----------
eligible = steady_runs.copy()
eligible['cv_time'] = pd.to_datetime(
    eligible['Activity Date'], format='mixed', errors='coerce', utc=True,
)
eligible['duration_min'] = clean_numeric(eligible['Moving Time']) / 60

numeric_cols = [PACE, 'hrr', *WEATHER]
eligible[numeric_cols] = clean_numeric(eligible.reindex(columns=numeric_cols))
eligible = eligible.dropna(subset=['cv_time', PACE, 'hrr', 'duration_min'])

# Runs with usable weather, in chronological order
ml_df = eligible.dropna(subset=WEATHER).copy()
if not ml_df.empty:
    ml_df['heat_index'] = clean_numeric(
        pd.Series(compute_heat_index(ml_df[WEATHER[0]], ml_df[WEATHER[1]]), index=ml_df.index)
    )
    ml_df = ml_df.dropna(subset=['heat_index'])
ml_df = ml_df.sort_values('cv_time', kind='stable')

n_samples = len(ml_df)
n_splits = config.ML_CV_SPLITS
timestamp_ids, timestamps = pd.factorize(ml_df['cv_time'])
model_trained = False

# ---------- Time-series CV, final fit, weather adjustment ----------
if n_samples >= config.ML_MIN_SAMPLES and len(timestamps) > n_splits:
    fold_rows = []
    # Split on unique timestamps so runs with identical times stay together
    for train_times, test_times in TimeSeriesSplit(n_splits=n_splits).split(timestamps):
        train = ml_df.loc[np.isin(timestamp_ids, train_times)]
        test = ml_df.loc[np.isin(timestamp_ids, test_times)]

        errors = {}
        for label, columns in FEATURE_SETS.items():
            fitted = LinearRegression().fit(train[columns], train['hrr'])
            residuals = fitted.predict(test[columns]) - test['hrr']
            errors[label] = np.abs(residuals).mean() * HRR_RANGE
        fold_rows.append(errors)

    fold_mae = pd.DataFrame(fold_rows)
    fold_mae.index = pd.RangeIndex(1, len(fold_mae) + 1, name='Fold')
    mean_mae = fold_mae.mean()
    heat_gain = fold_mae['Pace + duration'] - fold_mae[FULL]    # positive favours adding heat
    helped = int(heat_gain.round(2).gt(0).sum())

    # Final fit on all runs: sklearn for adjustment, statsmodels for the interval
    X = ml_df[FEATURE_SETS[FULL]]
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
        coef_heat_bpm = ols.params['heat_index'] * HRR_RANGE
        ci_low, ci_high = ols.conf_int().loc['heat_index'] * HRR_RANGE

    # Shift each run's actual %HRR by the modelled heat effect (keeps its residual)
    standard_temp = config.ML_BASELINE_TEMP_C
    standard_hum = config.ML_BASELINE_HUMIDITY
    standard_heat_index = float(compute_heat_index(standard_temp, standard_hum))

    X_standard = X.copy()
    X_standard['heat_index'] = standard_heat_index
    ml_df['adjusted_hrr'] = y + (model.predict(X_standard) - model.predict(X))

    # Drop runs where the adjusted value is ~0 or negative, since the ratio would blow up
    matched_runs = ml_df[ml_df['adjusted_hrr'] > 0.05].copy()
    matched_runs['adjusted_aero_ratio'] = matched_runs[PACE] / matched_runs['adjusted_hrr']
    model_trained = True

# ---------- Choose which runs and metric to plot ----------
# Raw always uses every eligible run, so the headline trend never depends on whether the
# weather model trained. Weather-adjusted can only use runs with usable weather.
kpi_slot = st.container()
view = 'Raw'

if model_trained and not matched_runs.empty:
    view = st.segmented_control(
        'Efficiency view',
        ['Raw', 'Weather-adjusted'],
        default='Raw',
        label_visibility='collapsed',
        help=(
            f'Reference weather is {standard_temp:.0f}°C and {standard_hum:.0f}% humidity. '
            'The heat effect is estimated from your runs and may be weak, so treat the '
            'adjustment as modest. Weather-adjusted uses only runs with usable weather.'
        ),
    ) or 'Raw'

if view == 'Weather-adjusted':
    display_runs = matched_runs.assign(aero_ratio=matched_runs['adjusted_aero_ratio'])
else:
    display_runs = eligible

runs = (
    display_runs.assign(aero_ratio=clean_numeric(display_runs['aero_ratio']))
    .dropna(subset=['cv_time', 'aero_ratio'])
    .sort_values('cv_time', kind='stable')[['cv_time', 'aero_ratio']]
    .copy()
)

# ---------- KPIs ----------
latest_ratio = change = np.nan
if not runs.empty:
    window = pd.Timedelta(config.BASELINE_WINDOW)
    runs['rolling_ratio'] = runs.rolling(config.BASELINE_WINDOW, on='cv_time')['aero_ratio'].mean()
    latest_ratio = runs['rolling_ratio'].iloc[-1]

    reference_date = runs['cv_time'].iloc[-1] - pd.DateOffset(months=6)
    previous = runs.loc[
        runs['cv_time'].gt(reference_date - window) & runs['cv_time'].le(reference_date),
        'aero_ratio',
    ].mean()
    if pd.notna(previous) and previous > 0:
        change = (latest_ratio / previous - 1) * 100

with kpi_slot:
    metric_cards([
        ('Latest 42-day efficiency', f'{latest_ratio:.2f}' if pd.notna(latest_ratio) else '—'),
        ('Change vs. 6 months ago', f'{change:+.1f}%' if pd.notna(change) else '—'),
        ('Runs in view', f'{len(runs):,}'),
    ])

# ---------- Main chart ----------
if runs.empty:
    st.info('No steady-state runs had enough data to plot aerobic efficiency.')
else:
    st.altair_chart(plot_aero(runs, theme=theme), width='stretch')
    st.caption(
        'Efficiency ratio is grade-adjusted speed divided by the fraction '
        'of heart-rate reserve used.'
    )
    if view == 'Weather-adjusted':
        st.caption(
            f'Showing {len(runs):,} of {len(eligible):,} eligible runs: only runs with usable '
            'weather can be adjusted, so this view covers fewer runs than Raw.'
        )
    elif model_trained:
        st.caption(
            f'Showing all {len(eligible):,} eligible runs. The weather-adjusted view uses '
            f'{len(matched_runs):,} of them.'
        )

# ---------- Heat model comparison ----------
if model_trained:
    st.header('Heat index model comparison')
    st.caption(
        f'{n_samples} of {len(eligible)} eligible steady-state runs had usable weather. '
        f'All models used those runs and the same {len(fold_mae)} chronological folds. '
        'MAEs are in bpm; lower is better.'
    )

    if heat_ci_available:
        st.write(heat_verdict(coef_heat_bpm, ci_low, ci_high, heat_gain.mean(), helped, len(fold_mae)))
        metric_cards([
            (f'Heat coefficient (bpm per {HI})', f'{coef_heat_bpm:+.2f}'),
            (f'95% interval (bpm per {HI})', f'{ci_low:+.2f} to {ci_high:+.2f}'),
            ('Error reduction with heat', f'{heat_gain.mean():+.2f} bpm'),
        ])
    else:
        st.write(
            'The available runs provide too little independent variation '
            'to estimate the heat coefficient.'
        )

    error_panel, heat_panel = st.columns([1, 1.2], gap='medium')
    with error_panel:
        st.subheader('Prediction error by model')
        st.altair_chart(plot_prediction_error(mean_mae, theme=theme), width='stretch')
        st.caption(
            f'Mean held-out error across {len(fold_mae)} time-ordered folds. '
            'Duration may capture cardiac drift, run type, or pacing.'
        )

    with heat_panel:
        st.subheader('Heart rate vs. heat index')
        if heat_ci_available:
            # Put every run at the median pace and duration, keeping its residual
            reference = design.copy()
            reference[PACE] = design[PACE].median()
            reference['duration_min'] = design['duration_min'].median()
            fitted_bpm = config.DEFAULT_HR_REST + ols.predict(reference) * HRR_RANGE
            adjusted_bpm = fitted_bpm + ols.resid * HRR_RANGE

            st.altair_chart(
                plot_heat_effect(ml_df['heat_index'], adjusted_bpm, fitted_bpm, theme=theme),
                width='stretch',
            )
            st.caption('Points and fitted line show heart rate at the median pace and duration.')
        else:
            st.caption('A fitted heat relationship is unavailable for this sample.')

    with st.expander('Fold results and heat estimate'):
        st.caption('Each fold tests on later runs. Highlighted cells have the lowest error in their row.')
        fold_table = fold_mae.assign(**{'Heat gain with duration': heat_gain}).reset_index()
        formats = {'Fold': '{:d}', **{col: '{:.2f}' for col in fold_table.columns[1:]}}
        with st.container(border=True):
            st.html(styled_table(
                fold_table, formats, theme, highlight_min=list(FEATURE_SETS),
            ))

        st.write(
            f'Adding heat to pace and duration lowered error in {helped} of {len(fold_mae)} folds. '
            f'Adding duration to pace changed mean error from {mean_mae["Pace only"]:.2f} '
            f'to {mean_mae["Pace + duration"]:.2f} bpm.'
        )
        if heat_ci_available:
            hi_p10, hi_p90 = ml_df['heat_index'].quantile([0.10, 0.90])
            st.write(
                f'The middle 80% of heat-index readings ranged from {hi_p10:.2f} to {hi_p90:.2f} {HI}. '
                f'The fitted heart-rate difference across that range was '
                f'{(hi_p90 - hi_p10) * coef_heat_bpm:+.2f} bpm at the same pace and duration.'
            )
            st.caption(
                'The OLS interval assumes independent errors across runs, so it is likely '
                'somewhat too narrow. The coefficient is associative, not causal: it may absorb '
                'fitness drift or time-of-day effects.'
            )
else:
    st.caption(
        f'Weather adjustment needs {config.ML_MIN_SAMPLES} comparable runs with usable weather '
        f'across at least {n_splits + 1} distinct start times. {n_samples} runs currently qualify.'
    )

# ---------- Methodology ----------
with st.expander('Aerobic efficiency methodology'):
    st.markdown(
        r"""
Aerobic efficiency divides grade-adjusted speed by the fraction of heart-rate reserve used:

$$ \text{Efficiency} = \frac{\text{Grade-adjusted speed}}{\text{Fraction of heart-rate reserve used}} $$

Only steady-state runs with at least 15 minutes of moving time are used. The 42-day average
weights each run equally within the previous 42 days. The six-month comparison uses the
42-day window ending six calendar months before the latest run in view.

The Raw view uses every eligible run. The weather-adjusted view shifts each run's heart rate
by the estimated heat effect, as if it had been run at the reference heat index, with pace
and duration held fixed. It can only include runs with usable weather. Heat index (°HI)
combines air temperature and humidity into one value.
"""
    )
