import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import parse_gpx, parse_fit
import config

# Calculates cumulative TRIMP score using time-series data
def calc_trimps(df, hr_max=config.DEFAULT_HR_MAX, hr_rest=config.DEFAULT_HR_REST, gender=config.DEFAULT_GENDER):
    if df.empty or 'heart_rate' not in df or 'timestamp' not in df:
        return 0.0

    # Calculates the time elapsed between consecutive heart rate readings in minutes
    delta_t_minutes = pd.to_datetime(df['timestamp']).diff().dt.total_seconds().fillna(1.0) / 60.0
    
    hr = pd.to_numeric(df['heart_rate'], errors='coerce').ffill().bfill()    # Coerce, forward fill and backward fill ensure no NaN values
    delta_hr = (hr - hr_rest) / (hr_max - hr_rest)
    delta_hr = delta_hr.clip(0.0, 1.0)
    
    if gender.lower() == 'male':
        y = 0.64 * np.exp(1.92 * delta_hr)
    else:
        y = 0.86 * np.exp(1.67 * delta_hr)
        
    row_trimp = delta_t_minutes * delta_hr * y
    
    return float(row_trimp.sum())

# TRIMP score fetcher, adjusts for workout
def get_trimp_for_row(row, time_series_df=None):
    trimp = 0.0
    
    if time_series_df is not None and not time_series_df.empty:
        trimp = calc_trimps(time_series_df)
        
    # Default to 40 TRIMP score/h as heart rate data underestimates weightlifting efforts
    if row['Activity Type'] in ['Workout', 'Weight Training']:
        return (row['Moving Time']/3600) * 40 
            
    return trimp

# Parses all granular data
@st.cache_data
def parse_granular(df):
    daily = df.groupby(df['Activity Date'].dt.normalize())['trimp'].sum()
    daily = daily[daily > 0].rename('trimps')

    if daily.empty:
        return pd.DataFrame(columns=['Date', 'trimps', 'CTL', 'ATL', 'TSB'])

    full_range = pd.date_range(daily.index.min(), pd.Timestamp.today().normalize())
    trimps = daily.reindex(full_range, fill_value=0).rename_axis('Date').reset_index()

    trimps['CTL'] = trimps['trimps'].ewm(alpha=1/config.CTL_DAY_SPAN, adjust=False).mean()
    trimps['ATL'] = trimps['trimps'].ewm(alpha=1/config.ATL_DAY_SPAN, adjust=False).mean()
    trimps['TSB'] = (trimps['CTL'].shift(1) - trimps['ATL'].shift(1)).fillna(0)
    return trimps

# Computes heat index (Rothfusz regression) to collapse temp + humidity into a single
# physiologically-meaningful, non-collinear feature that encodes their interaction
def compute_heat_index(temp_c, rh):
    temp_f = temp_c * 9 / 5 + 32
    
    # Simple formula, valid across the full range but most accurate <80°F (26.7°C)
    hi_simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094)
    
    # Rothfusz regression, only valid/accurate >=80°F (26.7°C)
    hi_rothfusz = (
        -42.379 + 2.049 * temp_f + 10.143 * rh - 0.225 * temp_f * rh
        - 6.838e-3 * temp_f**2 - 5.482e-2 * rh**2
        + 1.22874e-3 * temp_f**2 * rh + 8.528e-4 * temp_f * rh**2
        - 1.99e-6 * temp_f**2 * rh**2
    )
    
    hi_f = np.where((temp_f + hi_simple) / 2 >= 80, hi_rothfusz, hi_simple)
    return (hi_f - 32) * 5 / 9

# Returns valid steady-state runs with %HRR and the raw aerobic efficiency ratio
def get_steady_runs(summary_df):
    runs = summary_df[(summary_df['Activity Type'] == 'Run') & 
                      (summary_df['workout_style'] == 'Steady State')].copy()
    
    # Filter for valid pacing/HR and minimum moving time (15 min)
    runs = runs[(runs['Average Grade Adjusted Pace'] > 0) & 
                (runs['Average Heart Rate'] > 0) & 
                (runs['Moving Time'] >= 900)]
    
    # Heart Rate Reserve corrects for the resting HR offset
    hr_rest = config.DEFAULT_HR_REST
    hr_max = config.DEFAULT_HR_MAX
    runs['hrr'] = (runs['Average Heart Rate'] - hr_rest) / (hr_max - hr_rest)
    runs = runs[runs['hrr'] > 0.05]    # Removes likely sensor errors
    
    runs['aero_ratio'] = runs['Average Grade Adjusted Pace'] / runs['hrr']
    return runs

# Names each cluster from its average distance and effort vs the overall median
def name_clusters(df):
    med_dist = df['distance_km'].median()
    med_hrr = df['hrr'].median()
    means = df.groupby('Cluster')[['distance_km', 'hrr']].mean()

    names = {}
    for c, row in means.iterrows():
        length = 'Long' if row['distance_km'] > med_dist else 'Short'
        effort = 'hard' if row['hrr'] > med_hrr else 'easy'
        names[c] = f'{length} & {effort}'

    # If two clusters get the same name, add the cluster number so the legend stays unique
    counts = pd.Series(names).value_counts()
    return {c: n if counts[n] == 1 else f'{n} (#{c})' for c, n in names.items()}
