from itertools import cycle

import altair as alt
import pandas as pd
import streamlit as st

import config
from utils.data_loader import parse_csv


def activity_mix_chart(activities):
    counts = (
        activities.astype('string').fillna('Unspecified').value_counts()
        .rename_axis('Activity')
        .reset_index(name='Count')
    )
    total = counts['Count'].sum()
    counts['Share'] = counts['Count'] / total
    counts['Order'] = range(len(counts))
    counts['Slice label'] = counts['Share'].map(
        lambda share: f'{share:.0%}' if share >= 0.07 else ''
    )
    counts['Legend'] = [
        f'{activity} · {share:.1%}'
        for activity, share in zip(counts['Activity'], counts['Share'])
    ]

    colors = config.ACTIVITY_MIX_COLORS.copy()
    extra_colors = cycle(config.ACTIVITY_MIX_EXTRA_COLORS)
    for activity in sorted(set(counts['Activity']) - colors.keys()):
        colors[activity] = next(extra_colors)

    base = alt.Chart(counts).encode(
        theta=alt.Theta('Count:Q').stack(True),
        order=alt.Order('Order:Q', sort='ascending'),
        detail='Activity:N',
    )
    arcs = base.mark_arc(
        innerRadius=46,
        outerRadius=68,
        stroke=config.UI_BACKGROUND,
        strokeWidth=2,
    ).encode(
        color=alt.Color(
            'Legend:N',
            scale=alt.Scale(
                domain=counts['Legend'].tolist(),
                range=counts['Activity'].map(colors).tolist(),
            ),
            legend=alt.Legend(title=None, orient='right', labelLimit=190),
        ),
        tooltip=[
            alt.Tooltip('Activity:N'),
            alt.Tooltip('Count:Q', title='Activities', format=',d'),
            alt.Tooltip('Share:Q', format='.1%'),
        ],
    )
    labels = base.mark_text(
        radius=83, size=11, color=config.UI_TEXT,
    ).encode(text='Slice label:N')

    center = alt.Chart(pd.DataFrame({'Total': [f'{total:,}']}))
    count = center.mark_text(
        dy=-8, size=25, fontWeight=600, color=config.UI_TEXT,
    ).encode(text='Total:N')
    caption = center.mark_text(
        dy=13, size=11, color=config.UI_MUTED_TEXT,
    ).encode(text=alt.value('activities'))

    return (
        alt.layer(arcs, labels, count, caption)
        .properties(height=180)
        .configure_view(stroke=None)
    )


def monthly_distance_chart(distance_runs, months):
    monthly = (
        distance_runs.assign(
            Month=distance_runs['Activity Date'].dt.to_period('M').dt.to_timestamp(),
        )
        .groupby('Month')['Distance km']
        .sum()
        .reindex(months, fill_value=0)
        .rename_axis('Month')
        .reset_index()
    )

    return alt.Chart(monthly).mark_bar(
        color=config.STRAVA_ACCENT,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
    ).encode(
        x=alt.X(
            'yearmonth(Month):T',
            title=None,
            axis=alt.Axis(
                format='%b %y',
                tickCount=6,
                labelOverlap='greedy',
                grid=False,
            ),
        ),
        y=alt.Y('Distance km:Q', title='km', axis=alt.Axis(tickCount=4)),
        tooltip=[
            alt.Tooltip('Month:T', format='%b %Y'),
            alt.Tooltip('Distance km:Q', title='Distance (km)', format=',.1f'),
        ],
    ).properties(height=180).configure_view(stroke=None)


def hours_and_minutes(seconds):
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:,}h {minutes}m'


st.set_page_config(layout='wide', page_title='Strava Analytics')
st.html('''
<style>
[data-testid="stMainBlockContainer"] {
    padding-top: 3.5rem;
    padding-bottom: 1.5rem;
}
</style>
''')

if 'summary_df' not in st.session_state:
    st.session_state['summary_df'] = parse_csv()

data = st.session_state['summary_df'].copy()
data['Activity Date'] = pd.to_datetime(
    data['Activity Date'], format='mixed', errors='coerce', utc=True,
).dt.tz_localize(None)

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

activity_types = ['All', *sorted(data['Activity Type'].dropna().unique())]
selected_type = st.sidebar.selectbox('Activity type', activity_types)
filtered = data if selected_type == 'All' else data[data['Activity Type'] == selected_type]

distances = filtered.loc[
    ~filtered['Activity Type'].isin(['Workout', 'Weight Training']),
    ['Activity Date', 'Distance'],
].copy()
distances['Distance km'] = pd.to_numeric(distances['Distance'], errors='coerce') / 1000
total_km = distances['Distance km'].sum()
longest_km = distances['Distance km'].max() if distances['Distance km'].notna().any() else 0

total_seconds = pd.to_numeric(filtered['Moving Time'], errors='coerce').sum()
calories = pd.to_numeric(filtered['Calories'], errors='coerce').sum()
elevation = (
    pd.to_numeric(filtered['Elevation Gain'], errors='coerce').sum()
    if 'Elevation Gain' in filtered else 0
)

activities_per_week = 0
seconds_per_week = 0
dates = filtered['Activity Date'].dropna()
if len(dates) > 1:
    weeks = (dates.max() - dates.min()).days / 7
    activities_per_week = len(filtered) / weeks if weeks > 0 else len(filtered)
    seconds_per_week = total_seconds / weeks if weeks > 0 else total_seconds

volume, consistency = st.columns(2, gap='small')
with volume.container(border=True):
    st.markdown('## Training volume')
    first, second = st.columns(2)
    first.metric('Activities', f'{len(filtered):,}')
    second.metric('Distance', f'{total_km:,.1f} km')
    first.metric('Moving time', hours_and_minutes(total_seconds))
    second.metric('Calories burned', f'{calories:,.0f} kcal')

with consistency.container(border=True):
    st.markdown('## Performance and consistency')
    first, second = st.columns(2)
    first.metric('Elevation gain', f'{elevation:,.0f} m')
    second.metric('Longest distance', f'{longest_km:,.1f} km')
    first.metric('Moving time / week', hours_and_minutes(seconds_per_week))
    second.metric('Activities / week', f'{activities_per_week:.1f}')

trend, mix = st.columns([1.35, 1], gap='medium')
with trend:
    st.markdown('## Monthly distance')
    dated_distances = distances.dropna(subset=['Activity Date', 'Distance km'])
    if dated_distances.empty:
        st.caption('No dated distance records in this view.')
    else:
        months = pd.period_range(archive_start, archive_end, freq='M').to_timestamp()
        st.altair_chart(monthly_distance_chart(dated_distances, months), width='stretch')

with mix:
    st.markdown('## Activity mix')
    if filtered.empty:
        st.caption('No activities to show.')
    elif selected_type == 'All':
        st.altair_chart(activity_mix_chart(filtered['Activity Type']), width='stretch')
    else:
        st.caption('Select All to see the activity mix across the archive.')

with st.expander('About this project'):
    st.write(
        'This dashboard uses a Strava activity export to explore training volume, '
        'consistency, and aerobic efficiency over time. Home summarizes the archive. '
        'The sidebar opens the detailed analysis pages.'
    )
