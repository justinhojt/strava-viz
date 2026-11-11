import streamlit as st
import pandas as pd

from utils.data_loader import parse_csv, parse_gpx, parse_fit
from utils.functions import get_trimp_for_row
from utils.plots import plot_hr_zones, plot_hr_series, plot_ele_series
import config

# Fetch the shared dataset from session state if available, else load it in
if 'summary_df' in st.session_state:
    summary_df = st.session_state['summary_df']
else:
    summary_df = parse_csv()

st.subheader('⚡ Session Overview')

# Sidebar navigation/filtering
st.sidebar.header('Activity Filter')
activity_types = summary_df['Activity Type'].unique()
selected_type = st.sidebar.selectbox('Select Activity Type', activity_types)

filtered_summary = summary_df[summary_df['Activity Type'] == selected_type]

# Activity selector dropdown
activity_map = {
    f'{date.strftime("%Y-%m-%d")} - {name}': filename 
    for date, name, filename in zip(
        filtered_summary['Activity Date'], 
        filtered_summary['Activity Name'], 
        filtered_summary['Filename']
    )
}

selected_activity_label = st.sidebar.selectbox('Select Specific Session', list(activity_map.keys()))
target_filename = activity_map[selected_activity_label]

# Parse second-by-second activity data
with st.spinner('Parsing data...'):
    if target_filename.endswith('.gpx') or target_filename.endswith('.gpx.gz'):
        time_series_df = parse_gpx(target_filename)
    elif target_filename.endswith('.fit') or target_filename.endswith('.fit.gz'):
        time_series_df = parse_fit(target_filename) 
    else:
        st.error('Unsupported file format.')
        st.stop()

if time_series_df.empty:
    st.warning('This specific activity has a file entry but contains no coordinate data streams.')

selected_row = filtered_summary[filtered_summary['Filename'] == target_filename].iloc[0]

# KPI calculations
time = f'{selected_row["Moving Time"] // 60:.0f}m {selected_row["Moving Time"] % 60:.0f}s'
dist_m = f'{selected_row["Distance"]:.0f} m'
dist_km = f'{selected_row["Distance"] / 1000:.2f} km'
avg_hr = f'{selected_row["Average Heart Rate"]:.0f} bpm'
max_hr = f'{selected_row["Max Heart Rate"]:.0f} bpm'
cal = f'{selected_row["Calories"]:.0f} cal'

adjusted_trimp = get_trimp_for_row(selected_row, time_series_df)
trimp = f'{adjusted_trimp:.2f}'

seconds_100m = selected_row['Moving Time'] / (selected_row['Distance'] / 100) if selected_row['Distance'] > 0 else 0
pace_100m = f'{seconds_100m // 60:.0f}m {seconds_100m % 60:.0f}s/100m'

seconds_km = selected_row['Moving Time'] / (selected_row['Distance'] / 1000) if selected_row['Distance'] > 0 else 0
pace_km = f'{seconds_km // 60:.0f}m {seconds_km % 60:.0f}s/km'

# 60/40 split for KPIs and chart
top_left, top_right = st.columns([1.8, 1])

# KPIs
with top_left:
    if selected_row['Activity Type'] in ['Workout', 'Weight Training']:
        r1_col1, r1_col2, r1_col3 = st.columns(3)
        r1_col1.metric('Moving Time', time)
        r1_col2.metric('Average Heart Rate', avg_hr)
        r1_col3.metric('Maximum Heart Rate', max_hr)
        
        r2_col1, r2_col2, r2_col3 = st.columns(3)
        r2_col1.metric('Training Intensity Score', trimp)
        
    elif selected_row['Activity Type'] == 'Swim':
        r1_col1, r1_col2, r1_col3 = st.columns([1, 1, 1.5])
        r1_col1.metric('Distance', dist_m)
        r1_col2.metric('Moving Time', time)
        r1_col3.metric('Pace', pace_100m)
        
        r2_col1, r2_col2, r2_col3 = st.columns([1, 1, 1.5])
        r2_col1.metric('Average Heart Rate', avg_hr)
        r2_col2.metric('Maximum Heart Rate', max_hr)
        r2_col3.metric('Training Intensity Score', trimp)

    else:
        r1_col1, r1_col2, r1_col3 = st.columns([1, 1, 1.5])
        r1_col1.metric('Distance', dist_km)
        r1_col2.metric('Moving Time', time)
        r1_col3.metric('Pace', pace_km)
        
        r2_col1, r2_col2, r2_col3 = st.columns([1, 1, 1.5])
        r2_col1.metric('Average Heart Rate', avg_hr)
        r2_col2.metric('Maximum Heart Rate', max_hr)
        r2_col3.metric('Training Intensity Score', trimp)

# Heart rate zone duration bar chart 
with top_right:
    if 'heart_rate' in time_series_df.columns and time_series_df['heart_rate'].notna().any():
        
        act_max_hr = config.DEFAULT_HR_MAX
            
        # Calculate standard personalized zones based on % of max HR
        bins = [b * act_max_hr for b in config.HR_ZONE_PCT_BOUNDARIES]
        labels = config.HR_ZONE_LABELS
        
        # Map each second of data to a zone
        time_series_df['HR_Zone'] = pd.cut(time_series_df['heart_rate'], bins=bins, labels=labels)
        
        # Aggregate time 
        zone_counts = time_series_df['HR_Zone'].value_counts().reset_index()    # reset_index() to change zone from index to column
        zone_counts.columns = ['Zone', 'Time (s)']
        zone_counts['Minutes'] = zone_counts['Time (s)'] / 60
        
        hr_bar = plot_hr_zones(zone_counts, labels)
        st.altair_chart(hr_bar, width='stretch')
    else:
        st.info('No Heart Rate data recorded for this session.')
        
st.markdown('---')
    
# Plot heart rate and elevation data
if 'heart_rate' in time_series_df.columns and time_series_df['heart_rate'].notna().any():
    st.subheader('❤️ Heart Rate')
    hr_chart = plot_hr_series(time_series_df)
    st.altair_chart(hr_chart, width='stretch')

if 'elevation' in time_series_df.columns and time_series_df['elevation'].notna().any():
    st.subheader('⛰️ Elevation')
    elevation_chart = plot_ele_series(time_series_df)
    st.altair_chart(elevation_chart, width='stretch')
