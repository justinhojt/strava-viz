import pandas as pd
import streamlit as st

import config
from utils.data_loader import get_summary, load_recording
from utils.functions import get_trimp_for_row
from utils.plots import (
    format_clock,
    get_theme,
    plot_ele_series,
    plot_hr_series,
    plot_hr_zones,
)

MAX_GAP_SECONDS = 10  # gaps longer than this are treated as auto-pause


def optional_metric(value, format_string):
    return format_string.format(value) if pd.notna(value) else '—'


def prepare_series(recorded):
    """Clean timestamps and add elapsed-time columns used by the plots."""
    series = recorded.copy()
    if 'timestamp' not in series:
        return series.iloc[:0]

    series['timestamp'] = pd.to_datetime(series['timestamp'], errors='coerce', utc=True)
    series = series.dropna(subset=['timestamp']).sort_values('timestamp', kind='stable')
    series['Elapsed seconds'] = (series['timestamp'] - series['timestamp'].min()).dt.total_seconds()
    series['Elapsed'] = series['Elapsed seconds'].map(format_clock)

    for field in ('heart_rate', 'elevation'):
        if field in series:
            series[field] = pd.to_numeric(series[field], errors='coerce')
    return series


def zone_bands_and_minutes(series):
    """Return zone bands (for shading) and minutes spent in each zone."""
    labels = list(config.HR_ZONE_LABELS)
    edges = [pct * config.DEFAULT_HR_MAX for pct in config.HR_ZONE_PCT_BOUNDARIES]
    bands = pd.DataFrame({'Zone': labels, 'Lower': edges[:-1], 'Upper': edges[1:]})

    zone = pd.cut(series['heart_rate'], bins=edges, labels=labels)
    gap_seconds = (
        series['timestamp'].diff().dt.total_seconds().fillna(1).clip(upper=MAX_GAP_SECONDS)
    )
    minutes = (
        gap_seconds.groupby(zone, observed=False).sum().div(60)
        .rename_axis('Zone').reset_index(name='Minutes')
    )
    return labels, bands, minutes


def build_metrics(session, selected_type, trimp):
    moving = session['Moving Time']
    distance = session['Distance']
    moving_label = format_clock(moving) if pd.notna(moving) else '—'

    hr_metrics = [
        ('Average heart rate', optional_metric(session['Average Heart Rate'], '{:.0f} bpm')),
        ('Maximum heart rate', optional_metric(session['Max Heart Rate'], '{:.0f} bpm')),
        ('Training load', optional_metric(trimp, '{:.2f}')),
    ]

    if selected_type in ('Workout', 'Weight Training'):
        return [('Moving time', moving_label), *hr_metrics], 2

    swim = selected_type == 'Swim'
    pace_distance = 100 if swim else 1000
    pace = (
        format_clock(moving * pace_distance / distance)
        if pd.notna(moving) and pd.notna(distance) and distance > 0 else '—'
    )
    distance_label = (
        optional_metric(distance, '{:,.0f} m') if swim
        else optional_metric(distance / 1000, '{:,.2f} km')
    )
    return [
        ('Distance', distance_label),
        ('Moving time', moving_label),
        ('Pace (min/100 m)' if swim else 'Pace (min/km)', pace),
        *hr_metrics,
    ], 3


# ---------- Sidebar: choose activity ----------
summary = get_summary()
activity_types = summary['Activity Type'].dropna().unique() if not summary.empty else []
if len(activity_types) == 0:
    st.info('No activities are available.')
    st.stop()

st.sidebar.header('Activity filter')
selected_type = st.sidebar.selectbox('Activity type', activity_types)
sessions = summary[summary['Activity Type'] == selected_type].reset_index(drop=True)

dates = pd.to_datetime(sessions['Activity Date'], format='mixed', errors='coerce')
names = sessions['Activity Name'].fillna('Untitled activity')
choices = dates.dt.strftime('%Y-%m-%d').fillna('Date unavailable') + ' · ' + names

duplicate = choices.duplicated(keep=False)
if duplicate.any():
    occurrence = choices.groupby(choices, sort=False).cumcount() + 1
    choices.loc[duplicate] += ' (' + occurrence.loc[duplicate].astype(str) + ')'

selected = st.sidebar.selectbox(
    'Session', range(len(sessions)), format_func=choices.iloc.__getitem__,
)
session = sessions.iloc[selected]
date, name = dates.iloc[selected], names.iloc[selected]
st.title(f'{name}, {date.day} {date:%b %Y}' if pd.notna(date) else name)

# ---------- Load data ----------
with st.spinner('Loading session data…'):
    recorded = load_recording(session['Filename'])

if recorded is None:
    st.error('This session needs a GPX or FIT activity file.')
    st.stop()
if recorded.empty:
    st.warning('The activity file contains no recorded samples.')

trimp = get_trimp_for_row(session, recorded)
metrics, metric_columns = build_metrics(session, selected_type, trimp)
series = prepare_series(recorded)

has_hr = 'heart_rate' in series and series['heart_rate'].notna().any()
has_elevation = 'elevation' in series and series['elevation'].notna().any()

# ---------- Layout ----------
theme = get_theme()

kpis, time_in_zones = st.columns(2, gap='small')
with kpis.container(border=True):
    st.markdown('## Session metrics')
    columns = st.columns(metric_columns)
    for i, (label, value) in enumerate(metrics):
        columns[i % metric_columns].metric(label, value)

if has_hr:
    labels, bands, zone_minutes = zone_bands_and_minutes(series)

with time_in_zones.container(border=True):
    st.markdown('## Time in heart rate zones')
    if has_hr:
        st.altair_chart(plot_hr_zones(zone_minutes, labels, theme=theme), width='stretch')
    else:
        st.caption('No timed heart rate data for this session.')

if has_hr:
    st.markdown('## Heart rate')
    st.altair_chart(plot_hr_series(series, bands=bands, theme=theme), width='stretch')

if has_elevation:
    st.markdown('## Elevation')
    st.altair_chart(plot_ele_series(series, theme=theme), width='stretch')
