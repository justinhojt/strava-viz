import altair as alt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

import config
from utils.data_loader import parse_csv
from utils.functions import compute_heat_index, get_steady_runs


PACE = 'Average Grade Adjusted Pace'
WEATHER = ['temperature_2m', 'relative_humidity_2m']
ACCENT = getattr(config, 'STRAVA_ACCENT', '#FC4C02')
DARK = st.get_option('theme.base') == 'dark'
TEXT = st.get_option('theme.textColor') or ('#EDF1F6' if DARK else '#1E2938')
GRID = '#394451' if DARK else '#D3DAE2'
MUTED = '#9BAFC0' if DARK else '#657C90'
FONT = 'Inter, sans-serif'


def style_chart(chart):
    return (
        chart.configure(font=FONT)
        .configure_axis(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=TEXT,
            titleColor=TEXT,
            labelFontSize=12,
            titleFontSize=12,
            gridColor=GRID,
            domain=False,
        )
        .configure_legend(
            labelFont=FONT,
            labelFontSize=12,
            labelColor=TEXT,
            orient='bottom',
            direction='horizontal',
        )
        .configure_view(stroke=None)
    )


def metric_cards(items):
    for column, (label, value) in zip(st.columns(len(items), gap='small'), items):
        with column.container(border=True):
            st.metric(label, value)


def efficiency_chart(runs):
    trend = runs.drop_duplicates('cv_time', keep='last')
    series = alt.Color(
        'Series:N',
        scale=alt.Scale(
            domain=['Individual run', '42-day average'],
            range=[MUTED, ACCENT],
        ),
        legend=alt.Legend(title=None),
    )
    x = alt.X(
        'cv_time:T',
        title=None,
        axis=alt.Axis(format='%b %Y', tickCount=7, labelAngle=0, grid=False),
    )
    y_axis = alt.Axis(titleAngle=0, titleAlign='left', titleX=0, titleY=-10)
    scale = alt.Scale(zero=False)

    points = (
        alt.Chart(runs)
        .transform_calculate(Series="'Individual run'")
        .mark_circle(size=32, opacity=0.45)
        .encode(
            x=x,
            y=alt.Y('aero_ratio:Q', title='Efficiency ratio', scale=scale, axis=y_axis),
            color=series,
            tooltip=[
                alt.Tooltip('cv_time:T', title='Run date', format='%d %b %Y'),
                alt.Tooltip('aero_ratio:Q', title='Efficiency ratio', format='.2f'),
            ],
        )
    )
    average = (
        alt.Chart(trend)
        .transform_calculate(Series="'42-day average'")
        .mark_line(strokeWidth=2.8, point=alt.OverlayMarkDef(size=13, filled=True))
        .encode(
            x=x,
            y=alt.Y('rolling_ratio:Q', title='Efficiency ratio', scale=scale, axis=y_axis),
            color=series,
            tooltip=[
                alt.Tooltip('cv_time:T', title='Run date', format='%d %b %Y'),
                alt.Tooltip('rolling_ratio:Q', title='42-day average', format='.2f'),
            ],
        )
    )
    return style_chart((points + average).properties(height=295))


def prediction_error_chart(mean_mae):
    errors = mean_mae.rename_axis('Model').reset_index(name='MAE')
    errors['Label'] = errors['MAE'].map('{:.2f}'.format)
    displayed_errors = errors['MAE'].round(2)
    errors['Lowest error'] = displayed_errors.eq(displayed_errors.min())

    base = alt.Chart(errors).encode(
        y=alt.Y('Model:N', sort=errors['Model'].tolist(), axis=alt.Axis(title=None)),
        x=alt.X(
            'MAE:Q',
            axis=None,
            scale=alt.Scale(domain=[0, max(1, errors['MAE'].max() * 1.18)], nice=False),
        ),
    )
    bars = base.mark_bar(size=23, cornerRadiusEnd=4).encode(
        color=alt.condition(
            alt.datum['Lowest error'], alt.value(ACCENT), alt.value(MUTED),
        ),
        tooltip=[
            alt.Tooltip('Model:N'),
            alt.Tooltip('MAE:Q', title='MAE (bpm)', format='.2f'),
        ],
    )
    labels = base.mark_text(align='left', dx=6, color=TEXT, size=12).encode(text='Label:N')
    return style_chart((bars + labels).properties(height=205))


def heat_chart(heat_index, design, fit, hrr_range):
    # Put every run at the same pace and duration while retaining its fitted residual.
    reference = design.copy()
    reference[PACE] = design[PACE].median()
    reference['duration_min'] = design['duration_min'].median()
    expected_bpm = config.DEFAULT_HR_REST + fit.predict(reference) * hrr_range

    points = pd.DataFrame({
        'Heat index': heat_index.to_numpy(),
        'Heart rate': (expected_bpm + fit.resid * hrr_range).to_numpy(),
    })
    fitted = pd.DataFrame({
        'Heat index': heat_index.to_numpy(),
        'Heart rate': expected_bpm.to_numpy(),
    }).sort_values('Heat index')

    x = alt.X(
        'Heat index:Q',
        title='Heat index (°HI)',
        scale=alt.Scale(zero=False),
        axis=alt.Axis(tickCount=5, grid=False),
    )
    y = alt.Y('Heart rate:Q', title='Heart rate (bpm)', scale=alt.Scale(zero=False))
    dots = alt.Chart(points).mark_circle(color=MUTED, opacity=0.55, size=35).encode(
        x=x,
        y=y,
        tooltip=[
            alt.Tooltip('Heat index:Q', format='.2f'),
            alt.Tooltip('Heart rate:Q', title='Adjusted heart rate (bpm)', format='.2f'),
        ],
    )
    line = alt.Chart(fitted).mark_line(color=ACCENT, strokeWidth=2.5).encode(x=x, y=y)
    return style_chart((dots + line).properties(height=205))


def fold_table(fold_mae):
    header = '#26303D' if DARK else '#EAEFF4'
    border = '#3F4B59' if DARK else '#D7DEE6'
    winner = ('#633923', '#FFE0C8') if DARK else ('#FCE4D4', '#85340F')

    table = (
        fold_mae.round(2).reset_index().style
        .hide(axis='index')
        .format('{:.2f}', subset=fold_mae.columns)
        .highlight_min(
            axis=1,
            subset=fold_mae.columns,
            props=f'background-color: {winner[0]}; color: {winner[1]}; font-weight: 600;',
        )
        .set_table_styles([
            {'selector': '', 'props': 'width: 100%; border-collapse: collapse;'},
            {
                'selector': 'th, td',
                'props': (
                    f'text-align: right; padding: 0.55rem 0.7rem; '
                    f'border: 1px solid {border}; white-space: nowrap;'
                ),
            },
            {'selector': 'th', 'props': f'background-color: {header}; font-weight: 600;'},
            {'selector': 'th:first-child, td:first-child', 'props': 'text-align: left;'},
        ])
    )
    return f'<div style="overflow-x:auto">{table.to_html()}</div>'


st.title('Aerobic Efficiency Trends')
st.caption('Steady-state runs with over 15 minutes of moving time.')

summary_df = st.session_state['summary_df'] if 'summary_df' in st.session_state else parse_csv()
steady_runs = get_steady_runs(summary_df)

eligible = steady_runs.copy()
run_times = pd.to_datetime(
    steady_runs['Activity Date'], format='mixed', errors='coerce', utc=True,
)
eligible['cv_time'] = run_times
eligible['duration_min'] = pd.to_numeric(eligible['Moving Time'], errors='coerce') / 60

numeric_cols = [PACE, 'hrr', 'duration_min', *WEATHER]
eligible[numeric_cols] = (
    eligible.reindex(columns=numeric_cols)
    .apply(pd.to_numeric, errors='coerce')
    .replace([np.inf, -np.inf], np.nan)
)
eligible = eligible.dropna(subset=['cv_time', PACE, 'hrr', 'duration_min'])

ml_df = eligible.dropna(subset=WEATHER).copy()
ml_df['heat_index'] = (
    compute_heat_index(ml_df[WEATHER[0]], ml_df[WEATHER[1]])
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
    features = [PACE, 'duration_min', 'heat_index']
    feature_sets = {
        'Pace only': [PACE],
        'Pace + heat': [PACE, 'heat_index'],
        'Pace + duration': [PACE, 'duration_min'],
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

    standard_temp = config.ML_BASELINE_TEMP_C
    standard_hum = config.ML_BASELINE_HUMIDITY
    standard_heat_index = compute_heat_index(standard_temp, standard_hum)

    X_standard = X.copy()
    X_standard['heat_index'] = standard_heat_index
    predicted_change = model.predict(X_standard) - model.predict(X)
    ml_df['adjusted_hrr'] = y + predicted_change
    matched_runs = ml_df[ml_df['adjusted_hrr'] > 0.05].copy()
    matched_runs['adjusted_aero_ratio'] = matched_runs[PACE] / matched_runs['adjusted_hrr']
    model_trained = True

kpi_slot = st.container()
if model_trained and not matched_runs.empty:
    view = st.segmented_control(
        'Efficiency view',
        ['Raw', 'Weather-adjusted'],
        default='Raw',
        label_visibility='collapsed',
        help=f'Reference weather is {standard_temp:.0f}°C and {standard_hum:.0f}% relative humidity.',
    ) or 'Raw'

    display_runs = matched_runs.copy()
    if view == 'Weather-adjusted':
        display_runs['aero_ratio'] = display_runs['adjusted_aero_ratio']
else:
    display_runs = steady_runs.assign(cv_time=run_times)

display_runs['aero_ratio'] = pd.to_numeric(display_runs['aero_ratio'], errors='coerce')
runs = (
    display_runs.replace({'aero_ratio': [np.inf, -np.inf]}, np.nan)
    .dropna(subset=['cv_time', 'aero_ratio'])
    .sort_values('cv_time', kind='stable')
    [['cv_time', 'aero_ratio']]
    .copy()
)

latest_ratio = change = np.nan
if not runs.empty:
    window = pd.Timedelta(days=42)
    runs['rolling_ratio'] = runs.rolling(window, on='cv_time')['aero_ratio'].mean()
    latest_date = runs['cv_time'].iloc[-1]
    latest_ratio = runs['rolling_ratio'].iloc[-1]

    reference_date = latest_date - pd.DateOffset(months=6)
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

if runs.empty:
    st.info('No steady-state runs had enough data to plot aerobic efficiency.')
else:
    st.html('''
        <style>
        @media (min-width: 64rem) {
            .st-key-efficiency_chart {
                width: calc(100% + 2rem);
                max-width: none;
                margin-inline: -1rem;
            }
        }
        </style>
    ''')
    with st.container(key='efficiency_chart'):
        st.altair_chart(efficiency_chart(runs), width='stretch')
    st.caption(
        'Efficiency ratio is grade-adjusted speed divided by the fraction '
        'of heart-rate reserve used.'
    )
    if model_trained and not matched_runs.empty:
        st.caption('Both views use the same runs with usable weather.')

if model_trained:
    st.header('Heat index model comparison')
    metric_cards([
        ('Heat coefficient (bpm per °HI)', f'{coef_heat_bpm:+.2f}' if heat_ci_available else '—'),
        ('95% interval (bpm per °HI)', f'{ci_low:+.2f} to {ci_high:+.2f}' if heat_ci_available else '—'),
        ('Error reduction with heat', f'{duration_heat_gain.mean():+.2f} bpm'),
    ])

    if not heat_ci_available:
        st.write(
            'The available runs provide too little independent variation '
            'to estimate the heat coefficient.'
        )
    elif ci_low <= 0 <= ci_high:
        st.write('The 95% interval includes zero, so the data leave a zero heat effect plausible.')
    elif ci_low > 0:
        st.write('At the same pace and duration, hotter runs were associated with a higher heart rate.')
    else:
        st.write('At the same pace and duration, hotter runs were associated with a lower heart rate.')

    error_panel, heat_panel = st.columns([1, 1.2], gap='medium')
    with error_panel:
        st.subheader('Prediction error by model')
        st.altair_chart(prediction_error_chart(fold_mae.mean()), width='stretch')
        st.caption(
            f'Mean held-out error in bpm across {len(fold_mae)} time-ordered folds '
            f'using the same {n_samples} runs. Lower is better.'
        )

    with heat_panel:
        st.subheader('Heart rate vs. heat index')
        if heat_ci_available:
            st.altair_chart(
                heat_chart(ml_df['heat_index'], design, ols, hrr_range),
                width='stretch',
            )
            st.caption('Points and fitted line show heart rate at the median pace and duration.')
        else:
            st.caption('A fitted heat relationship is unavailable for this sample.')

    with st.expander('Fold results and heat estimate'):
        st.caption('Each fold tests on later runs. Highlighted cells have the lowest error in their row.')
        with st.container(border=True):
            st.html(fold_table(fold_mae))

        mean_mae = fold_mae.mean()
        helped = duration_heat_gain.round(2).gt(0).sum()
        st.write(
            f'Adding heat to pace and duration lowered error in {helped} of {len(fold_mae)} folds. '
            f'Adding duration to pace changed mean error from {mean_mae["Pace only"]:.2f} '
            f'to {mean_mae["Pace + duration"]:.2f} bpm. Duration may capture cardiac drift, '
            'run type, or differences in pacing.'
        )
        if heat_ci_available:
            hi_p10, hi_p90 = ml_df['heat_index'].quantile([0.10, 0.90])
            st.write(
                f'The middle 80% of heat-index readings ranged from {hi_p10:.2f} to {hi_p90:.2f}° HI. '
                f'The fitted heart-rate difference across that range was '
                f'{(hi_p90 - hi_p10) * coef_heat_bpm:+.2f} bpm at the same pace and duration.'
            )
            st.caption('The conventional OLS interval assumes independent errors across runs.')
elif not steady_runs.empty:
    st.caption(
        f'Weather adjustment needs {config.ML_MIN_SAMPLES} comparable runs with usable weather '
        f'across at least {n_splits + 1} distinct start times. {n_samples} runs currently qualify.'
    )

with st.expander('Aerobic efficiency methodology'):
    st.markdown(
        'Aerobic efficiency divides grade-adjusted speed by the fraction of heart-rate '
        'reserve used. The 42-day average weights each run equally within the previous 42 days. '
        'The six-month comparison uses the 42-day window ending six calendar months before '
        'the latest run in view.\n\n'
        'The weather-adjusted view estimates each run’s heart rate at the reference heat index, '
        'with its pace and duration held fixed. Heat index combines air temperature and humidity.'
    )
