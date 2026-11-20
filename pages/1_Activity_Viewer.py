import math
from itertools import cycle, islice

import altair as alt
import pandas as pd
import streamlit as st

import config
from utils.data_loader import parse_csv, parse_fit, parse_gpx
from utils.functions import get_trimp_for_row


SERIES_HEIGHT = 224
CHART_FONT = 'Inter, sans-serif'
DEFAULT_ZONE_COLORS = ('#678599', '#568773', '#AD914F', '#D07D50', '#B9606B')

dark = st.get_option('theme.base') == 'dark'
text_color = st.get_option('theme.textColor') or ('#EDF1F6' if dark else '#1E2938')
grid_color = '#394451' if dark else '#D3DAE2'


def style_chart(chart):
    return (
        chart.configure(font=CHART_FONT)
        .configure_axis(
            labelFont=CHART_FONT,
            titleFont=CHART_FONT,
            labelFontSize=12,
            titleFontSize=13,
            labelColor=text_color,
            titleColor=text_color,
            gridColor=grid_color,
            gridOpacity=1,
            domainColor=grid_color,
            tickColor=grid_color,
        )
        .configure_view(stroke=None)
    )


def elapsed_axis(duration):
    if duration < 60:
        step = max(1, math.ceil(duration / 5))
    elif duration < 300:
        step = 60
    elif duration <= 2700:
        step = 300
    else:
        step = 600 * math.ceil(duration / 7200)

    return alt.Axis(
        values=list(range(0, int(duration) + 1, step)),
        labelExpr="floor(datum.value / 60) + ':' + format(datum.value % 60, '02d')",
        labelPadding=8,
        grid=False,
    )


def plot_series(data, metric, unit, color, bands=None, zone_color=None):
    duration = max(float(data['Elapsed seconds'].max()), 1)
    x = alt.X(
        'Elapsed seconds:Q',
        title='Elapsed time (m:ss)',
        scale=alt.Scale(domain=[0, duration], nice=False),
        axis=elapsed_axis(duration),
    )

    y_scale = alt.Scale(zero=False)
    if bands is not None:
        low = max(0, math.floor((data[metric].min() - 4) / 5) * 5)
        high = math.ceil((data[metric].max() + 4) / 5) * 5
        y_scale = alt.Scale(domain=[low, high], nice=False)
        visible = bands.assign(
            Lower=bands['Lower'].clip(lower=low),
            Upper=bands['Upper'].clip(upper=high),
        )
        visible = visible[visible['Lower'] < visible['Upper']]

    line = alt.Chart(data[['Elapsed seconds', 'Elapsed', metric]]).mark_line(
        color=color,
        strokeWidth=2.1,
        clip=True,
    ).encode(
        x=x,
        y=alt.Y(
            f'{metric}:Q',
            title=unit,
            scale=y_scale,
            axis=alt.Axis(tickCount=5, grid=True),
        ),
        tooltip=[
            alt.Tooltip('Elapsed:N'),
            alt.Tooltip(f'{metric}:Q', title=unit, format='.0f'),
        ],
    )

    if bands is None:
        return style_chart(line.properties(height=SERIES_HEIGHT))

    shading = alt.Chart(visible).mark_rect(opacity=0.15).encode(
        x=alt.value(0),
        x2=alt.value('width'),
        y=alt.Y('Lower:Q', scale=y_scale, title=unit),
        y2='Upper:Q',
        color=zone_color,
    )
    return style_chart((shading + line).properties(height=SERIES_HEIGHT))


def plot_zone_minutes(zones, labels, zone_color):
    zones = zones.assign(Label=zones['Minutes'].map(lambda minutes: f'{minutes:.1f} min'))
    base = alt.Chart(zones).encode(
        y=alt.Y(
            'Zone:N',
            sort=labels,
            axis=alt.Axis(title=None, ticks=False, domain=False, labelPadding=8),
        ),
        x=alt.X(
            'Minutes:Q',
            axis=None,
            scale=alt.Scale(domain=[0, max(1, zones['Minutes'].max() * 1.28)], nice=False),
        ),
    )
    bars = base.mark_bar(size=22, cornerRadiusEnd=4).encode(
        color=zone_color,
        tooltip=[
            alt.Tooltip('Zone:N'),
            alt.Tooltip('Minutes:Q', format='.2f'),
        ],
    )
    labels_at_end = base.mark_text(
        align='left',
        baseline='middle',
        dx=6,
        size=12,
        color=text_color,
    ).encode(text='Label:N')

    return style_chart((bars + labels_at_end).properties(height=max(150, 29 * len(labels))))


def format_clock(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f'{minutes}:{seconds:02d}'


def optional_metric(value, format_string):
    return format_string.format(value) if pd.notna(value) else '—'


summary = st.session_state['summary_df'] if 'summary_df' in st.session_state else parse_csv()
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
    'Session',
    range(len(sessions)),
    format_func=choices.iloc.__getitem__,
)
session = sessions.iloc[selected]
date = dates.iloc[selected]
name = names.iloc[selected]
st.title(f'{name}, {date.day} {date:%b %Y}' if pd.notna(date) else name)

filename = session['Filename']
suffix = str(filename).lower()
with st.spinner('Loading session data…'):
    if suffix.endswith(('.gpx', '.gpx.gz')):
        recorded = parse_gpx(filename)
    elif suffix.endswith(('.fit', '.fit.gz')):
        recorded = parse_fit(filename)
    else:
        st.error('This session needs a GPX or FIT activity file.')
        st.stop()

if recorded.empty:
    st.warning('The activity file contains no recorded samples.')

trimp = get_trimp_for_row(session, recorded)
moving = session['Moving Time']
distance = session['Distance']
moving_label = format_clock(moving) if pd.notna(moving) else '—'

hr_metrics = [
    ('Average heart rate', optional_metric(session['Average Heart Rate'], '{:.0f} bpm')),
    ('Maximum heart rate', optional_metric(session['Max Heart Rate'], '{:.0f} bpm')),
    ('Training intensity score', optional_metric(trimp, '{:.2f}')),
]
if selected_type in ('Workout', 'Weight Training'):
    metrics = [('Moving time', moving_label), *hr_metrics]
    metric_columns = 2
else:
    swim = selected_type == 'Swim'
    pace_unit = '100 m' if swim else 'km'
    pace_distance = 100 if swim else 1000
    pace = (
        f'{format_clock(moving * pace_distance / distance)} / {pace_unit}'
        if pd.notna(moving) and pd.notna(distance) and distance > 0 else '—'
    )
    distance_label = (
        optional_metric(distance, '{:,.0f} m') if swim
        else optional_metric(distance / 1000, '{:,.2f} km')
    )
    metrics = [
        ('Distance', distance_label),
        ('Moving time', moving_label),
        ('Pace', pace),
        *hr_metrics,
    ]
    metric_columns = 3

series = recorded.copy()
if 'timestamp' in series:
    series['timestamp'] = pd.to_datetime(series['timestamp'], errors='coerce', utc=True)
    series = series.dropna(subset=['timestamp']).sort_values('timestamp', kind='stable')
    series['Elapsed seconds'] = (series['timestamp'] - series['timestamp'].min()).dt.total_seconds()
    series['Elapsed'] = series['Elapsed seconds'].map(format_clock)
else:
    series = series.iloc[:0]

for field in ('heart_rate', 'elevation'):
    if field in series:
        series[field] = pd.to_numeric(series[field], errors='coerce')

has_hr = 'heart_rate' in series and series['heart_rate'].notna().any()
has_elevation = 'elevation' in series and series['elevation'].notna().any()

if has_hr:
    labels = list(config.HR_ZONE_LABELS)
    edges = [fraction * config.DEFAULT_HR_MAX for fraction in config.HR_ZONE_PCT_BOUNDARIES]
    bands = pd.DataFrame({'Zone': labels, 'Lower': edges[:-1], 'Upper': edges[1:]})

    palette = getattr(config, 'HR_ZONE_COLORS', None) or DEFAULT_ZONE_COLORS
    if isinstance(palette, dict):
        zone_colors = [
            palette.get(label, fallback)
            for label, fallback in zip(labels, cycle(DEFAULT_ZONE_COLORS))
        ]
    else:
        zone_colors = list(islice(cycle(palette), len(labels)))

    zone_color = alt.Color(
        'Zone:N',
        scale=alt.Scale(domain=labels, range=zone_colors),
        legend=None,
    )

    zone = pd.cut(series['heart_rate'], bins=edges, labels=labels)
    gap_seconds = series['timestamp'].diff().dt.total_seconds().fillna(1).clip(upper=10)
    zones = (
        gap_seconds.groupby(zone, observed=False).sum().div(60)
        .rename_axis('Zone').reset_index(name='Minutes')
    )

kpis, time_in_zones = st.columns(2, gap='small')
with kpis.container(border=True):
    st.markdown('## Session metrics')
    columns = st.columns(metric_columns)
    for index, (label, value) in enumerate(metrics):
        columns[index % metric_columns].metric(label, value)

with time_in_zones.container(border=True):
    st.markdown('## Time in heart rate zones')
    if has_hr:
        st.altair_chart(plot_zone_minutes(zones, labels, zone_color), width='stretch')
    else:
        st.caption('No timed heart rate data for this session.')

if has_hr:
    st.markdown('## Heart rate')
    hr_chart = plot_series(
        series, 'heart_rate', 'Heart rate (bpm)', text_color, bands, zone_color,
    )
    st.altair_chart(hr_chart, width='stretch')

if has_elevation:
    st.markdown('## Elevation')
    elevation_color = getattr(config, 'CHART_SECONDARY', '#4D769B')
    st.altair_chart(
        plot_series(series, 'elevation', 'Elevation (m)', elevation_color),
        width='stretch',
    )
