from html import escape
from itertools import cycle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config
from utils.data_loader import get_summary
from utils.functions import get_steady_runs, name_clusters
from utils.plots import get_theme, plot_cluster_efficiency, plot_silhouette
from utils.ui import styled_table

PACE = 'Average Grade Adjusted Pace'
CLUSTER = 'Cluster Name'
FEATURES = ['distance_km', PACE, 'hrr']
RECENT_DAYS = 90


# Fits K-means for every k and keeps the silhouette score of each valid fit
@st.cache_data(show_spinner=False)
def score_clusters(X, max_k):
    assignments, scores = {}, []
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if len(np.unique(labels)) == k:    # skip fits where a cluster ended up empty
            assignments[k] = labels
            scores.append({'k': k, 'score': silhouette_score(X, labels)})
    return assignments, pd.DataFrame(scores)


# % change in mean efficiency: last 90 days vs the 90 days before, per cluster
def recent_change(points):
    window = pd.Timedelta(days=RECENT_DAYS)
    cutoff = points['Date'].max() - window
    recent = points[points['Date'] > cutoff]
    previous = points[(points['Date'] > cutoff - window) & (points['Date'] <= cutoff)]

    recent_stats = recent.groupby(CLUSTER)['Efficiency ratio'].agg(['mean', 'size'])
    previous_stats = previous.groupby(CLUSTER)['Efficiency ratio'].agg(['mean', 'size'])
    both = recent_stats.join(previous_stats, lsuffix='_recent', rsuffix='_previous', how='inner')
    both = both[(both['size_recent'] >= 3) & (both['size_previous'] >= 3)]    # need enough runs per window
    return (both['mean_recent'] / both['mean_previous'] - 1) * 100


st.title('Run Clusters')
st.caption('Steady-state runs grouped by distance, grade-adjusted pace, and heart-rate reserve.')

summary_df = get_summary()
if summary_df.empty:
    st.info('No activities are available.')
    st.stop()

# ---------- Prepare runs ----------
runs = get_steady_runs(summary_df)    # adds hrr and aero_ratio
runs['distance_km'] = pd.to_numeric(runs['Distance'], errors='coerce') / 1000
runs[FEATURES] = runs[FEATURES].replace([np.inf, -np.inf], np.nan)
runs = runs.dropna(subset=FEATURES).copy()

if len(runs) < config.KMEANS_MIN_RUNS:
    st.info(
        f'Need at least {config.KMEANS_MIN_RUNS} steady-state runs to cluster. '
        f'{len(runs)} qualify.'
    )
    st.stop()

# ---------- Scale, then cluster for every k ----------
# Scaling puts km, pace and %HRR on the same footing so no feature dominates
X = StandardScaler().fit_transform(runs[FEATURES])
max_k = min(config.KMEANS_MAX_K, len(runs) - 1)
assignments, scores_df = score_clusters(X, max_k)

if scores_df.empty:
    st.info('The qualifying runs have too little variation to form distinct clusters.')
    st.stop()

best_k = int(scores_df.loc[scores_df['score'].idxmax(), 'k'])
k = st.segmented_control(
    'Number of clusters', list(assignments), default=best_k,
) or best_k

runs['Cluster'] = assignments[k]
runs[CLUSTER] = runs['Cluster'].map(name_clusters(runs))
runs['Efficiency ratio'] = runs['aero_ratio']
runs['Date'] = pd.to_datetime(
    runs['Activity Date'], format='mixed', errors='coerce', utc=True,
)

# ---------- Cluster profiles ----------
theme = get_theme()

profiles = (
    runs.groupby(CLUSTER)
    .agg(
        Runs=('Cluster', 'size'),
        **{
            'Distance (km)': ('distance_km', 'mean'),
            'Avg HR (bpm)': ('Average Heart Rate', 'mean'),
            'Efficiency ratio': ('Efficiency ratio', 'mean'),
        },
    )
    .sort_values('Distance (km)')
    .reset_index()
)
profiles.insert(2, 'Share of runs', profiles['Runs'] / len(runs))

# Same colour per cluster in the table order and in the chart
colors = dict(zip(profiles[CLUSTER], cycle(config.CHART_CATEGORICAL_COLORS)))

st.header('Cluster profiles')
table = profiles.assign(**{CLUSTER: profiles[CLUSTER].map(escape)})
formats = {
    'Runs': '{:,.0f}',
    'Share of runs': '{:.0%}',
    'Distance (km)': '{:.2f}',
    'Avg HR (bpm)': '{:.0f}',
    'Efficiency ratio': '{:.2f}',
}
with st.container(border=True):
    st.html(styled_table(table, formats, theme))
st.caption(
    '"Easy" and "hard" are relative to your own median effort. Heart-rate reserve is '
    'one of the clustering inputs, so the labels are partly defined by heart rate itself.'
)

# ---------- Efficiency over time ----------
st.header('Efficiency over time by cluster')
points = runs[['Date', CLUSTER, 'Efficiency ratio']].dropna()

if points.empty:
    st.info('No clustered runs have both a date and an efficiency value.')
else:
    focus = st.selectbox('Highlight cluster', ['All clusters', *profiles[CLUSTER]])
    shown = points if focus == 'All clusters' else points[points[CLUSTER] == focus]

    change = recent_change(shown)
    if change.empty:
        st.write(
            f'A recent comparison needs at least three runs per cluster in each '
            f'{RECENT_DAYS}-day window.'
        )
    else:
        st.write(
            f'Mean efficiency, last {RECENT_DAYS} days vs the {RECENT_DAYS} days before: '
            + ', '.join(f'**{name}** {value:+.1f}%' for name, value in change.items())
        )

    st.altair_chart(
        plot_cluster_efficiency(shown, colors, theme=theme), width='stretch',
    )
    st.caption('Dots are individual runs. Lines are 42-day rolling averages per cluster.')

# ---------- Choosing k ----------
with st.expander('Choosing k'):
    st.altair_chart(plot_silhouette(scores_df, best_k, theme=theme), width='stretch')
    st.caption(
        'Higher silhouette scores mean tighter, better-separated clusters. '
        f'The best tested score was {scores_df["score"].max():.3f} at k = {best_k}.'
    )
