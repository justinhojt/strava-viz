import gzip
import os

import gpxpy
import pandas as pd
import streamlit as st
from fitparse import FitFile

import config


def parse_dates(values, *, utc=True, preserve_wall_time=False):
    """Parse UTC by default, with options to retain offsets or clock times."""
    if utc and not preserve_wall_time:
        return pd.to_datetime(values, format='mixed', errors='coerce', utc=True)

    # Parsing individually accepts columns that mix offsets and naive dates.
    parsed = values.map(lambda value: pd.to_datetime(value, errors='coerce'))

    if preserve_wall_time:
        return parsed.map(
            lambda timestamp: pd.NaT if pd.isna(timestamp) else timestamp.tz_localize(None)
        ).astype('datetime64[ns]')

    return parsed.astype('datetime64[ns]') if parsed.isna().all() else parsed


@st.cache_data
def parse_csv():
    path = config.ACTIVITIES_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f'Could not find activities dataset at {path}')

    df = pd.read_csv(path)
    df['Activity Date'] = parse_dates(df['Activity Date'], utc=False)
    return df


def get_summary():
    """Load the activity summary once per Streamlit session."""
    if 'summary_df' not in st.session_state:
        st.session_state['summary_df'] = parse_csv()
    return st.session_state['summary_df']


def process_timestamps(df, local_tz=config.TIMEZONE_TARGET):
    """Convert GPS timestamps to local clock time for the Activity Viewer."""
    if not df.empty:
        timestamps = parse_dates(df['timestamp'])
        df['timestamp'] = timestamps.dt.tz_convert(local_tz).dt.tz_localize(None)
        df['graph_timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    return df


def _open_activity(filename, mode, **kwargs):
    path = os.path.join(config.DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')

    opener = gzip.open if path.lower().endswith('.gz') else open
    return opener(path, mode, **kwargs)


def _gpx_heart_rate(extensions):
    for extension in extensions:
        for element in extension.iter():
            name = element.tag.rsplit('}', 1)[-1].lower()
            if name in ('hr', 'heartrate'):
                try:
                    return int(element.text)
                except (TypeError, ValueError):
                    continue
    return None


@st.cache_data
def parse_gpx(gpx_filename, local_tz=config.TIMEZONE_TARGET):
    with _open_activity(gpx_filename, 'rt', encoding='utf-8') as file:
        gpx = gpxpy.parse(file)

    track_data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                track_data.append({
                    'timestamp': point.time,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'heart_rate': _gpx_heart_rate(point.extensions),
                })

    return process_timestamps(pd.DataFrame(track_data), local_tz)


@st.cache_data
def parse_fit(fit_filename, local_tz=config.TIMEZONE_TARGET):
    track_data = []

    with _open_activity(fit_filename, 'rb') as file:
        # FitFile reads record messages lazily.
        fitfile = FitFile(file)
        for record in fitfile.get_messages('record'):
            values = record.get_values()
            if 'timestamp' not in values:
                continue

            elevation = values.get('enhanced_altitude')
            if elevation is None:
                elevation = values.get('altitude')

            latitude = values.get('position_lat')
            longitude = values.get('position_long')
            if latitude is not None and longitude is not None:
                degrees_per_semicircle = 180 / 2**31
                latitude *= degrees_per_semicircle
                longitude *= degrees_per_semicircle

            track_data.append({
                'timestamp': values['timestamp'],
                'latitude': latitude,
                'longitude': longitude,
                'elevation': elevation,
                'heart_rate': values.get('heart_rate'),
            })

    return process_timestamps(pd.DataFrame(track_data), local_tz)
