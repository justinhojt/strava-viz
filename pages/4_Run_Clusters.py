from html import escape
from itertools import cycle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config
from utils.data_loader import parse_csv
from utils.functions import get_steady_runs, name_clusters


PACE = 'Average Grade Adjusted Pace'
HEART_RATE = 'Average Heart Rate'
CLUSTER = 'Cluster Name'

ACCENT = getattr(config, 'STRAVA_ACCENT', '#FC4C02')
PALETTE = getattr(config, 'CHART_CATEGORICAL_COLORS', None) or (
    ACCENT, '#4D769B', '#527B68', '#8B709C', '#9C784C', '#687486',
)
DARK = st.get_option('theme.base') == 'dark'
TEXT = st.get_option('theme.textColor') or ('#EDF1F6' if DARK else '#1E2938')
GRID = '#394451' if DARK else '#D3DAE2'
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
            labelColor=TEXT,
            labelFontSize=12,
            orient='bottom',
            direction='horizontal',
        )
        .configure_view(stroke=None)
    )


def profile_table(profiles, colors):
    def cluster_label(name):
        return (
            f'<span aria-hidden="true" style="display:inline-block;width:0.7rem;'
            f'height:0.7rem;margin-right:0.45rem;background:{colors[name]};'
            f'border-radius:2px"></span>{escape(name)}'
        )

    header = '#26303D' if DARK else '#EAEFF4'
    border = '#3F4B59' if DARK else '#D7DEE6'

    table = (
        profiles.style.hide(axis='index')
        .format({
            CLUSTER: cluster_label,
            'Runs': '{:,.0f}',
            'Share of runs': '{:.0%}',
            'Distance (km)': '{:.2f}',
            'Avg HR (bpm)': '{:.0f}',
            'Efficiency ratio': '{:.2f}',
        }, na_rep='—')
        .set_table_styles([
            {'selector': '', 'props': 'width: 100%; border-collapse: collapse;'},
            {
                'selector': 'th, td',
                'props': (
                    f'border: 1px solid {border}; padding: 0.55rem 0.7rem; '
                    'text-align: right; white-space: nowrap;'
                ),
            },
            {'selector': 'th', 'props': f'background-color: {header}; font-weight: 600;'},
            {'selector': 'th:first-child, td:first-child', 'props': 'text-align: left;'},
        ])
    )
    return f'<div style="overflow-x:auto">{table.to_html()}</div>'


def efficiency_chart(points, colors):
    series = []
    for _, group in points.groupby(CLUSTER, sort=False):
        group = group.sort_values('Date', kind='stable').copy()
        group['42-day average'] = (
            group.rolling('42D', on='Date')['Efficiency ratio'].mean()
        )
        series.append(group.drop_duplicates('Date', keep='last'))
    trend = pd.concat(series)

    active_names = set(points[CLUSTER])
    names = [name for name in colors if name in active_names]
    color = alt.Color(
        f'{CLUSTER}:N',
        title=None,
        scale=alt.Scale(domain=names, range=[colors[name] for name in names]),
        legend=alt.Legend(symbolType='square', symbolOpacity=1),
    )
    x = alt.X(
        'Date:T',
        title=None,
        axis=alt.Axis(
            format='%b %Y',
            tickCount=7,
            labelAngle=0,
            labelOverlap='greedy',
            grid=False,
        ),
    )
    y_scale = alt.Scale(zero=False)

    dots = alt.Chart(points).mark_circle(size=33, opacity=0.40).encode(
        x=x,
        y=alt.Y('Efficiency ratio:Q', scale=y_scale),
        color=color,
        tooltip=[
            alt.Tooltip('Date:T', format='%d %b %Y'),
            alt.Tooltip(f'{CLUSTER}:N'),
            alt.Tooltip('Efficiency ratio:Q', format='.2f'),
        ],
    )
    lines = alt.Chart(trend).mark_line(strokeWidth=2.6, opacity=1).encode(
        x=x,
        y=alt.Y('42-day average:Q', title='Efficiency ratio', scale=y_scale),
        color=color,
        tooltip=[
            alt.Tooltip('Date:T', format='%d %b %Y'),
            alt.Tooltip(f'{CLUSTER}:N'),
            alt.Tooltip('42-day average:Q', format='.2f'),
        ],
    )
    return style_chart((dots + lines).properties(height=345))


def recent_takeaway(points, end):
    window = pd.Timedelta(days=90)
    cutoff = end - window
    previous = points.loc[
        points['Date'].gt(cutoff - window) & points['Date'].le(cutoff)
    ]
    recent = points.loc[points['Date'].gt(cutoff)]

    def window_means(values):
        grouped = values.groupby(CLUSTER)['Efficiency ratio']
        return grouped.mean()[grouped.size().ge(3)]

    earlier_mean = window_means(previous)
    recent_mean = window_means(recent)
    change = (
        (recent_mean / earlier_mean[earlier_mean.gt(0)] - 1) * 100
    ).dropna().round(1)

    if change.empty:
        return 'A recent comparison needs at least three runs per cluster in each 90-day window.'

    best = ', '.join(change.index[change.eq(change.max())])
    gain = change.max()
    if len(change) == 1:
        return f'{best} changed {gain:+.1f}% in mean efficiency versus the preceding 90 days.'
    if gain > 0:
        return (
            f'{best} had the largest recent efficiency gain '
            f'({gain:+.1f}% versus the preceding 90 days).'
        )
    return 'Mean efficiency was flat or lower across the comparable clusters in the latest 90 days.'


def silhouette_chart(scores, best_k):
    base = alt.Chart(scores).encode(
        x=alt.X(
            'k:Q',
            title='Number of clusters (k)',
            scale=alt.Scale(
                domain=[scores['k'].min() - 0.5, scores['k'].max() + 0.5],
                nice=False,
            ),
            axis=alt.Axis(values=scores['k'].tolist(), format='d', grid=False),
        ),
        y=alt.Y('score:Q', title='Silhouette score', scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('k:Q', format='d'),
            alt.Tooltip('score:Q', format='.3f'),
        ],
    )
    line = base.mark_line(color='#62798C', point=True)
    best = base.transform_filter(alt.datum.k == best_k)
    point = best.mark_circle(color=ACCENT, size=120, opacity=1)
    label = best.mark_text(dy=-15, color=TEXT, size=12).encode(
        text=alt.value(f'Best k = {best_k}')
    )
    return style_chart((line + point + label).properties(height=215, padding={'top': 20}))


@st.cache_data(show_spinner=False)
def score_clusters(features, max_k):
    assignments = {}
    scores = []
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(features)
        if len(np.unique(labels)) == k:
            assignments[k] = labels
            scores.append({'k': k, 'score': silhouette_score(features, labels)})
    return assignments, pd.DataFrame(scores)


st.title('Run Clusters')
st.caption('Steady-state runs grouped by distance, grade-adjusted pace, and heart-rate reserve.')

summary_df = st.session_state['summary_df'] if 'summary_df' in st.session_state else parse_csv()
if summary_df.empty:
    st.info('No activities are available.')
    st.stop()

runs = get_steady_runs(summary_df).copy()
features = ['distance_km', PACE, 'hrr']
runs['distance_km'] = pd.to_numeric(runs['Distance'], errors='coerce') / 1000
runs[features] = (
    runs[features]
    .apply(pd.to_numeric, errors='coerce')
    .replace([np.inf, -np.inf], np.nan)
)
runs = runs.dropna(subset=features).copy()

if len(runs) < config.KMEANS_MIN_RUNS:
    st.info(
        f'Need at least {config.KMEANS_MIN_RUNS} steady-state runs to cluster. '
        f'{len(runs)} qualify.'
    )
    st.stop()

X = StandardScaler().fit_transform(runs[features])
max_k = min(config.KMEANS_MAX_K, len(runs) - 1, len(np.unique(X, axis=0)))
assignments, scores_df = score_clusters(X, max_k)

if scores_df.empty:
    st.info('The qualifying runs have too little variation to form two distinct clusters.')
    st.stop()

best_k = int(scores_df.loc[scores_df['score'].idxmax(), 'k'])
k = st.segmented_control(
    'Number of clusters', list(assignments), default=best_k,
) or best_k
runs['Cluster'] = assignments[k]

runs[HEART_RATE] = pd.to_numeric(runs[HEART_RATE], errors='coerce')
runs['Efficiency ratio'] = pd.to_numeric(runs['aero_ratio'], errors='coerce').replace(
    [np.inf, -np.inf], np.nan,
)
profiles = runs.groupby('Cluster').agg(
    Runs=('Cluster', 'size'),
    **{
        'Distance (km)': ('distance_km', 'mean'),
        'Avg HR (bpm)': (HEART_RATE, 'mean'),
        'Efficiency ratio': ('Efficiency ratio', 'mean'),
    },
).sort_values('Distance (km)')

names = pd.Series(name_clusters(runs), dtype='string').reindex(profiles.index)
fallback = pd.Series({cluster: f'Cluster {cluster + 1}' for cluster in profiles.index})
names = names.fillna(fallback)
duplicate = names.duplicated(keep=False)
names.loc[duplicate] += ' (' + (names.index[duplicate] + 1).astype(str) + ')'
runs[CLUSTER] = runs['Cluster'].map(names)

profiles.insert(0, CLUSTER, names)
profiles.insert(2, 'Share of runs', profiles['Runs'] / len(runs))
colors = dict(zip(profiles[CLUSTER], cycle(PALETTE)))

st.header('Cluster profiles')
with st.container(border=True):
    st.html(profile_table(profiles, colors))

st.write(
    '“Easy” and “hard” describe relative effort within this dataset. '
    'Heart-rate reserve is one of the inputs used to form the clusters.'
)

st.header('Efficiency over time by cluster')
runs['Date'] = pd.to_datetime(
    runs['Activity Date'], format='mixed', errors='coerce', utc=True,
)
points = runs[['Date', CLUSTER, 'Efficiency ratio']].dropna()

if points.empty:
    st.info('No clustered runs have both a date and an efficiency value.')
else:
    active_names = set(points[CLUSTER])
    names_in_chart = [name for name in colors if name in active_names]
    focus = st.selectbox('Highlight cluster', ['All clusters', *names_in_chart])
    shown = points if focus == 'All clusters' else points.loc[points[CLUSTER].eq(focus)]

    st.write(recent_takeaway(shown, points['Date'].max()))
    st.altair_chart(efficiency_chart(shown, colors), width='stretch')
    st.caption('Dots are individual runs. Lines are 42-day rolling averages for each cluster.')

with st.expander('Choosing k'):
    st.altair_chart(silhouette_chart(scores_df, best_k), width='stretch')
    st.caption(
        'Higher silhouette scores indicate tighter, better-separated clusters. '
        f'The best tested score was {scores_df["score"].max():.3f} at k = {best_k}.'
    )
