import pandas as pd
import streamlit as st

from utils.data_loader import get_summary, parse_dates
from utils.plots import plot_donut, plot_monthly_distance
from utils.ui import metric_card

NON_DISTANCE_TYPES = ['Workout', 'Weight Training']


def hours_and_minutes(seconds):
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:,}h {minutes}m'


def column_sum(df, column):
    """Sum a numeric column, returning 0 if the export doesn't have it."""
    if column not in df:
        return 0
    return pd.to_numeric(df[column], errors='coerce').sum()


st.set_page_config(layout='wide', page_title='Strava Analytics')
st.html('''
<style>
[data-testid="stMainBlockContainer"] {
    padding-top: 3.5rem;
    padding-bottom: 1.5rem;
}
</style>
''')

# Load data (cached in session state, shared with the other pages)
data = get_summary().copy()
data['Activity Date'] = parse_dates(
    data['Activity Date'], utc=False, preserve_wall_time=True,
)

archive_start = data['Activity Date'].min()
archive_end = data['Activity Date'].max()

st.title('Strava Archive Analytics')
if pd.notna(archive_start):
    first_month = archive_start.strftime('%b %Y')
    last_month = archive_end.strftime('%b %Y')
    span = first_month if first_month == last_month else f'{first_month} – {last_month}'
    st.caption(f'Strava export, {span}')
else:
    st.caption('Strava export')

# Sidebar filter
activity_types = ['All', *sorted(data['Activity Type'].dropna().unique())]
selected_type = st.sidebar.selectbox('Activity type', activity_types)
filtered = data if selected_type == 'All' else data[data['Activity Type'] == selected_type]

# Distance (exclude non-distance activities)
distances = filtered.loc[
    ~filtered['Activity Type'].isin(NON_DISTANCE_TYPES),
    ['Activity Date', 'Distance'],
].copy()
distances['Distance km'] = pd.to_numeric(distances['Distance'], errors='coerce') / 1000
total_km = distances['Distance km'].sum()
longest_km = distances['Distance km'].max() if distances['Distance km'].notna().any() else 0

# Totals
total_seconds = column_sum(filtered, 'Moving Time')
calories = column_sum(filtered, 'Calories')
elevation = column_sum(filtered, 'Elevation Gain')

# Weekly consistency
activities_per_week = 0
seconds_per_week = 0
dates = filtered['Activity Date'].dropna()
if len(dates) > 1:
    weeks = (dates.max() - dates.min()).days / 7
    activities_per_week = len(filtered) / weeks if weeks > 0 else len(filtered)
    seconds_per_week = total_seconds / weeks if weeks > 0 else total_seconds

# KPI cards
volume, consistency = st.columns(2, gap='small')
with metric_card(volume, 'Training volume'):
    first, second = st.columns(2)
    first.metric('Activities', f'{len(filtered):,}')
    second.metric('Distance', f'{total_km:,.1f} km')
    first.metric('Moving time', hours_and_minutes(total_seconds))
    second.metric('Calories burned', f'{calories:,.0f} kcal')

with metric_card(consistency, 'Performance and consistency'):
    first, second = st.columns(2)
    first.metric('Elevation gain', f'{elevation:,.0f} m')
    second.metric('Longest distance', f'{longest_km:,.1f} km')
    first.metric('Moving time / week', hours_and_minutes(seconds_per_week))
    second.metric('Activities / week', f'{activities_per_week:.1f}')

# Charts
trend, mix = st.columns([1.35, 1], gap='medium')
with trend:
    st.markdown('## Monthly distance')
    dated_distances = distances.dropna(subset=['Activity Date', 'Distance km'])
    if dated_distances.empty:
        st.caption('No dated distance records in this view.')
    else:
        months = pd.period_range(archive_start, archive_end, freq='M').to_timestamp()
        st.altair_chart(plot_monthly_distance(dated_distances, months), width='stretch')

with mix:
    st.markdown('## Activity mix')
    if filtered.empty:
        st.caption('No activities to show.')
    elif selected_type == 'All':
        st.altair_chart(plot_donut(filtered), width='stretch')
    else:
        st.caption('Select All to see the activity mix across the archive.')

with st.expander('About this project'):
    st.write(
        'This dashboard uses a Strava activity export to explore training volume, '
        'consistency, and aerobic efficiency over time. Home summarizes the archive. '
        'The sidebar opens the detailed analysis pages.'
    )
