import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.data_loader import parse_csv
from utils.functions import name_clusters
from utils.plots import plot_silhouette, plot_cluster_efficiency
import config

# Fetch the shared dataset from session state if available, else load it in
if 'summary_df' in st.session_state:
    summary_df = st.session_state['summary_df']
else:
    summary_df = parse_csv()

st.subheader('🧩 Run Types (K-means Clustering)')
st.write('Groups your steady-state runs by distance, pace and effort, then checks whether each group is getting more efficient.')

# Same filters as the Aerobic Efficiency page
runs = summary_df[(summary_df['Activity Type'] == 'Run') & (summary_df['workout_style'] == 'Steady State')].copy()
runs = runs[(runs['Average Grade Adjusted Pace'] > 0) &
            (runs['Average Heart Rate'] > 0) &
            (runs['Moving Time'] >= 900)]

runs['hrr'] = (runs['Average Heart Rate'] - config.DEFAULT_HR_REST) / (config.DEFAULT_HR_MAX - config.DEFAULT_HR_REST)
runs = runs[runs['hrr'] > 0.05]
runs['aero_ratio'] = runs['Average Grade Adjusted Pace'] / runs['hrr']
runs['distance_km'] = runs['Distance'] / 1000
runs = runs.dropna(subset=['distance_km', 'Average Grade Adjusted Pace', 'hrr'])

if len(runs) < config.KMEANS_MIN_RUNS:
    st.info(f'Need at least {config.KMEANS_MIN_RUNS} steady-state runs to cluster (currently {len(runs)}).')
    st.stop()

# Scale features so distance (km) doesn't dominate pace or %HRR
features = ['distance_km', 'Average Grade Adjusted Pace', 'hrr']
X = StandardScaler().fit_transform(runs[features])

# 1. Silhouette score for each k
scores = []
for k_try in range(2, config.KMEANS_MAX_K + 1):
    labels = KMeans(n_clusters=k_try, n_init=10, random_state=42).fit_predict(X)
    scores.append({'k': k_try, 'score': silhouette_score(X, labels)})
scores_df = pd.DataFrame(scores)
best_k = int(scores_df.loc[scores_df['score'].idxmax(), 'k'])

st.markdown('#### Choosing k')
st.altair_chart(plot_silhouette(scores_df), width='stretch')
st.caption(f'Higher is better (clusters are tight and well separated). The best score is at k = {best_k}.')

k = st.slider('Number of clusters', 2, config.KMEANS_MAX_K, value=best_k)

# 2. Fit the final model and name the clusters
runs['Cluster'] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
runs['Cluster Name'] = runs['Cluster'].map(name_clusters(runs))

# Cluster summary table
summary = runs.groupby('Cluster Name').agg(
    Runs=('Cluster', 'size'),
    Distance_km=('distance_km', 'mean'),
    Avg_HR=('Average Heart Rate', 'mean'),
    Efficiency=('aero_ratio', 'mean')
).round(2).sort_values('Distance_km')
st.markdown('#### Cluster profiles')
st.dataframe(summary, width='stretch')

# 3. Efficiency over time by cluster
st.markdown('#### Efficiency over time by run type')
st.altair_chart(plot_cluster_efficiency(runs), width='stretch')

st.caption('Efficiency = grade-adjusted pace / %HRR. Lines are 42-day rolling averages per cluster. '
           'Clusters use %HRR as an input, so "easy" and "hard" are partly defined by heart rate itself.')
