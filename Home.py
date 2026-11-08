import streamlit as st
import altair as alt
import config

from utils.data_loader import parse_csv
from utils.plots import plot_donut

st.set_page_config(layout='wide', page_title='Strava Analytics')

# Load data into session state so other pages can see it
if 'summary_df' not in st.session_state:
    st.session_state['summary_df'] = parse_csv()
summary_df = st.session_state['summary_df']

st.set_page_config(layout='wide', page_title='Strava Analytics')

st.title('Strava Archive Analytics Dashboard')
st.subheader('Lifetime Training Overview')

# Activity filter
st.sidebar.header('Activity Filter')
activity_types = ['All'] + list(summary_df['Activity Type'].unique())
selected_type = st.sidebar.selectbox('Select Activity Type', activity_types)

if selected_type != 'All':
    filtered_df = summary_df[summary_df['Activity Type'] == selected_type]
else:
    filtered_df = summary_df
    
# KPIs
total_activities = len(filtered_df)

dist_df = filtered_df[~filtered_df['Activity Type'].isin(['Workout', 'Weight Training'])]
total_distance_km = dist_df['Distance'].sum() / 1000

total_seconds = filtered_df['Moving Time'].sum()
total_hours = total_seconds // 3600
total_minutes = (total_seconds % 3600) // 60

total_calories = filtered_df['Calories'].sum()

total_elevation = filtered_df['Elevation Gain'].sum() if 'Elevation Gain' in filtered_df.columns else 0
max_distance_km = dist_df['Distance'].max() / 1000 if not dist_df.empty else 0

# Weekly consistency metrics
if not filtered_df.empty and len(filtered_df) > 1:
    min_date = filtered_df['Activity Date'].min()
    max_date = filtered_df['Activity Date'].max()
    weeks_active = (max_date - min_date).days / 7
    
    # Activities per week
    weekly_avg = total_activities / weeks_active if weeks_active > 0 else total_activities
    
    # Moving time per week
    avg_seconds_per_week = total_seconds / weeks_active if weeks_active > 0 else total_seconds
    avg_weekly_hours = avg_seconds_per_week // 3600
    avg_weekly_minutes = (avg_seconds_per_week % 3600) // 60
else:
    weekly_avg = 0
    avg_weekly_hours = 0
    avg_weekly_minutes = 0

st.subheader('⚡ Core Metrics')

# 60/40 split for KPIs and chart
top_left, top_right = st.columns([1.5, 1])

with top_left:
    row1_col1, row1_col2 = st.columns(2)
    row1_col1.metric('🏃‍♂️ Total Activities', f'{total_activities}')
    row1_col2.metric('📏 Total Distance', f'{total_distance_km:,.1f} km')
    
    row2_col1, row2_col2 = st.columns(2)
    row2_col1.metric('⏱️ Moving Time', f'{total_hours:.0f}h {total_minutes:.0f}m')
    row2_col2.metric('🔥 Calories Burned', f'{total_calories:,.0f} kcal')
    
with top_right:
    if selected_type == 'All' and not filtered_df.empty:
        donut_chart = plot_donut(filtered_df)
        st.altair_chart(donut_chart, width='stretch')
    else:
        st.info(f'Viewing filtered data for: **{selected_type}**.\n\nSelect "All" in the sidebar to view activity composition chart.')

st.markdown('---')

st.subheader('⭐ Performance Highs & Consistency')
ext_col1, ext_col2, ext_col3, ext_col4 = st.columns(4)

ext_col1.metric('⛰️ Total Elevation Gain', f'{total_elevation:,.0f} m')
ext_col2.metric('🗺️ Longest Activity Distance', f'{max_distance_km:,.1f} km')

ext_col3.metric('⏱️ Avg Moving Time / Week', f'{avg_weekly_hours:.0f}h {avg_weekly_minutes:.0f}m')    
ext_col4.metric('📅 Activities / Week', f'{weekly_avg:.1f}')
