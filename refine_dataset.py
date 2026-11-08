import pandas as pd
import requests
import logging
import config
import gzip
import time
import os
import io
import re

from timezonefinder import TimezoneFinder
from fitparse import FitFile
from tqdm import tqdm

# Globally disable logging warnings to keep our progress bar clean
logging.disable(logging.WARNING)

CSV_PATH = 'data/activities.csv'
OUTPUT_PATH = 'data/activities_refined.csv'

# Parses and cleans the initial activities CSV
def clean_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Could not find activities CSV at {file_path}')
        
    df = pd.read_csv(file_path)

    # Drop columns with no file name/all identical values
    df = df.dropna(subset=['Filename'])
    df = df.loc[:, df.nunique() > 1]

    # Strava export data has 2 distance columns, first one has inconsistent units, second one standardizes to meters (which we keep)
    if 'Distance.1' in df.columns:
        df = df.drop(columns=['Distance'])                    # Drop first distance column
        df = df.rename(columns={'Distance.1': 'Distance'})    # Rename the second distance column so it doesnt get flagged as a duplicate
    elif 'Distance' not in df.columns:
        raise KeyError(
            'Expected a "Distance" column in the Strava export, but none was found.'
        )
        
    # Drop remaining duplicate columns
    df = df.drop(columns=[*[col for col in df.columns if '.1' in col]], errors='ignore')
        
    return df

# Fetches weather averaged across every hour the activity actually spans,
# weighted by the fraction of the run that occurred in each hour
def get_weather_with_timeout(session, lat, lon, activity_start, duration_seconds, retries=3):
    url = 'https://archive-api.open-meteo.com/v1/archive'
    
    activity_end = activity_start + pd.Timedelta(seconds=duration_seconds)
    
    start_date_str = activity_start.strftime('%Y-%m-%d')
    end_date_str = activity_end.strftime('%Y-%m-%d')    # may differ if run crosses midnight
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date_str,
        'end_date': end_date_str,
        'hourly': ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m'],
        'timezone': 'auto'
    }
    
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=15)
            
            if response.status_code == 429:
                if attempt < retries - 1:
                    tqdm.write(' ⚠️ API Rate Limit Hit! Sleeping for 60 seconds...')
                    time.sleep(60)
                    continue
                else:
                    tqdm.write(f'   ❌ Still rate-limited after {retries} attempts for {start_date_str}.')
                    break
                
            response.raise_for_status()
            data = response.json()
            
            hourly_times = pd.to_datetime(data['hourly']['time'])
            temps = data['hourly']['temperature_2m']
            humidities = data['hourly']['relative_humidity_2m']
            winds = data['hourly']['wind_speed_10m']
            
            # Weight each hour by how much of the run overlapped it
            weighted_temp, weighted_hum, weighted_wind, total_weight = 0.0, 0.0, 0.0, 0.0
            
            for i, hour_start in enumerate(hourly_times):
                hour_end = hour_start + pd.Timedelta(hours=1)
                overlap_start = max(activity_start, hour_start)
                overlap_end = min(activity_end, hour_end)
                overlap_seconds = max(0, (overlap_end - overlap_start).total_seconds())
                
                if overlap_seconds > 0 and temps[i] is not None:
                    weighted_temp += temps[i] * overlap_seconds
                    weighted_hum += humidities[i] * overlap_seconds
                    weighted_wind += winds[i] * overlap_seconds
                    total_weight += overlap_seconds
            
            if total_weight > 0:
                return {
                    'temp': weighted_temp / total_weight,
                    'humidity': weighted_hum / total_weight,
                    'wind': weighted_wind / total_weight
                }
            else:
                # Fallback: nearest single hour if overlap weighting somehow fails
                nearest = (hourly_times - activity_start).abs().argmin()
                return {
                    'temp': temps[nearest],
                    'humidity': humidities[nearest],
                    'wind': winds[nearest]
                }
            
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                tqdm.write(f'   ⏳ Read timeout on attempt {attempt + 1}. Retrying in 2 seconds...')
                time.sleep(2)
            else:
                tqdm.write(f'   ❌ API Timeout failed after {retries} attempts for {start_date_str}.')
        except Exception as e:
            tqdm.write(f'   ❌ Weather API Fetch Failed for {start_date_str}: {e}')
            break
            
    return {'temp': None, 'humidity': None, 'wind': None}

# Instantly extracts the first GPS coordinate without loading the whole file into memory
def extract_start_coords(filename):
    file_path = os.path.join('data', filename)
    if not os.path.exists(file_path):
        tqdm.write(f'   ⚠️ [File Missing] Cannot find file at: {file_path}')
        return None, None
        
    try:
        if filename.endswith('.gpx') or filename.endswith('.gpx.gz'):
            open_func = gzip.open if filename.endswith('.gz') else open
            mode = 'rt' if filename.endswith('.gz') else 'r'
            
            with open_func(file_path, mode, encoding='utf-8') as f:
                for line in f:
                    if '<trkpt' in line or '<wpt' in line:
                        lat_match = re.search(r'lat=["\']([^"\']+)["\']', line)
                        lon_match = re.search(r'lon=["\']([^"\']+)["\']', line)
                        if lat_match and lon_match:
                            return float(lat_match.group(1)), float(lon_match.group(1))
            return None, None

        elif filename.endswith('.fit') or filename.endswith('.fit.gz'):
            open_func = gzip.open if filename.endswith('.gz') else open
            with open_func(file_path, 'rb') as f:
                fitfile = FitFile(io.BytesIO(f.read()) if filename.endswith('.gz') else f)
                
            for record in fitfile.get_messages('record'):
                values = record.get_values()
                if 'position_lat' in values and 'position_long' in values:
                    lat, lon = values['position_lat'], values['position_long']
                    if lat and lon:
                        if abs(lat) > 180: lat = lat * (180.0 / 2**31)
                        if abs(lon) > 180: lon = lon * (180.0 / 2**31)
                        return lat, lon
    except Exception as e:
        tqdm.write(f'   ❌ [Parsing Error] Failed to read {filename}: {e}')
    return None, None

def main():
    print(f'Loading and cleaning {CSV_PATH}...')
    df = clean_csv(CSV_PATH)
    
    activity_type_col = 'Activity Type' if 'Activity Type' in df.columns else 'type'
        
    df['temperature_2m'] = None
    df['relative_humidity_2m'] = None
    df['wind_speed_10m'] = None
    df['workout_style'] = 'Unknown'
    df['local_tz'] = config.TIMEZONE_TARGET
    
    df['Activity Date'] = df['Activity Date'].astype(object)    # Cast column to object type to provide fallback protection against PyArrow
    
    tzf = TimezoneFinder()
    
    # Spin up persistent connection session
    with requests.Session() as session:
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing Activities'):
            filename = row['Filename']
            is_run = str(row.get(activity_type_col, '')).lower() == 'run'
            
            if is_run:
                avg_speed = row.get('Average Speed')
                moving_time = row.get('Moving Time')
                elapsed_time = row.get('Elapsed Time')
                
                if (pd.isna(avg_speed) or avg_speed == 0 
                        or pd.isna(elapsed_time) or elapsed_time == 0 
                        or pd.isna(moving_time)):
                    style = 'Unknown'
                elif (moving_time / elapsed_time) < 0.7:
                    style = 'Interval'
                else:
                    style = 'Steady State'
                    
                df.at[index, 'workout_style'] = style
            
            lat, lon = extract_start_coords(filename)
                
            # Find the local timezone
            if lat is not None and lon is not None:
                tz_name = tzf.timezone_at(lng=lon, lat=lat) 
            else:
                tz_name = None
            
            raw_date = pd.to_datetime(row['Activity Date'])

            # Convert to local time
            if tz_name:
                df.at[index, 'local_tz'] = tz_name
                if raw_date.tz is not None:
                    localized_date = raw_date.tz_convert(tz_name).tz_localize(None)
                else:
                    localized_date = raw_date.tz_localize('UTC').tz_convert(tz_name).tz_localize(None)
            else:
                if raw_date.tz is not None:
                    localized_date = raw_date.tz_convert(config.TIMEZONE_TARGET).tz_localize(None)
                else:
                    localized_date = raw_date + pd.Timedelta(hours=config.TIMEZONE_OFFSET_HOURS)
                
            df.at[index, 'Activity Date'] = str(localized_date)    # Cast timestamp to str to bypass PyArrow string dtype requirements
            
            # Fetch weather data for runs with gps coordinates
            if is_run and lat is not None and lon is not None:
                duration_seconds = row.get('Elapsed Time', row.get('Moving Time', 0))
                weather = get_weather_with_timeout(session, lat, lon, localized_date, duration_seconds)
                
                df.at[index, 'temperature_2m'] = weather['temp']
                df.at[index, 'relative_humidity_2m'] = weather['humidity']
                df.at[index, 'wind_speed_10m'] = weather['wind']
                
                time.sleep(0.1)   # Respect rate limits
        
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSuccess! Dataset safely saved to {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
