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
@st.cache_data(show_spinner='Calculating historical training load metrics...')
def parse_granular(df):
    workout_records = []

    for index, row in df.iterrows():
        target_filename = row.get('Filename') 
        activity_date = row.get('Activity Date')       
        activity_type = row.get('Activity Type') 
        
        trimp_score = 0
        file_parsed_successfully = False

        # If activity file exists, parse it and calculate TRIMP score
        if not pd.isna(target_filename): 
            try:
                if target_filename.endswith('.gpx') or target_filename.endswith('.gpx.gz'):
                    time_series_df = parse_gpx(target_filename)
                elif target_filename.endswith('.fit') or target_filename.endswith('.fit.gz'):
                    time_series_df = parse_fit(target_filename)
                else:
                    raise ValueError(f'Unsupported file format: {target_filename}')
                
                file_parsed_successfully = True
                trimp_score = get_trimp_for_row(row, time_series_df)
                
            except Exception as e:
                st.warning(f'⚠️ Failed to parse {target_filename}: {e}')
                trimp_score = get_trimp_for_row(row, None)
        
        if trimp_score > 0 or file_parsed_successfully:
            workout_records.append({
                'Date': activity_date,
                'trimps': trimp_score
            })
            
    if not workout_records:
        return pd.DataFrame(columns=['Date', 'trimps', 'CTL', 'ATL', 'TSB'])
        
    summary_df = pd.DataFrame(workout_records)
    summary_df['Date'] = pd.to_datetime(summary_df['Date']).dt.normalize()
    
    daily_stress = summary_df.groupby('Date')['trimps'].sum().reset_index()
    daily_stress.set_index('Date', inplace=True)
    
    active_days = daily_stress[daily_stress['trimps'] > 0]
    if not active_days.empty:
        start_date = active_days.index.min()
    else:
        start_date = daily_stress.index.min()
        
    if pd.isna(start_date):
        return pd.DataFrame(columns=['Date', 'trimps', 'CTL', 'ATL', 'TSB'])
        
    end_date = pd.to_datetime('today').normalize()
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    trimps = daily_stress.reindex(full_range, fill_value=0).reset_index()
    trimps.rename(columns={'index': 'Date'}, inplace=True)
    
    trimps['CTL'] = trimps['trimps'].ewm(alpha=1/config.CTL_DAY_SPAN, adjust=False).mean()    # 42 day exponentially weighted moving average
    trimps['ATL'] = trimps['trimps'].ewm(alpha=1/config.ATL_DAY_SPAN, adjust=False).mean()     # 7 day exponentially weighted moving average
    trimps['TSB'] = trimps['CTL'].shift(1) - trimps['ATL'].shift(1)
    trimps['TSB'] = trimps['TSB'].fillna(0)
    
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
