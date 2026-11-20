import altair as alt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

import config
from utils.data_loader import parse_csv
from utils.functions import get_steady_runs, parse_granular


DARK = st.get_option('theme.base') == 'dark'
UI_TEXT = st.get_option('theme.textColor') or getattr(
    config, 'UI_TEXT', '#EDF1F6' if DARK else '#1E2938',
)
GRID = '#394451' if DARK else '#D3DAE2'
FONT = 'Inter, sans-serif'

ZONE_COLORS = {
    'Fresh / Recovery': '#6C8F79',
    'Maintenance Zone': '#858D97',
    'Optimal Training': '#B58948',
    'Overtraining Risk': '#B56A65',
}
RANGES = {'3 months': 3, '6 months': 6, '1 year': 12, 'All': None}


def calendar_days(dates):
    return pd.to_datetime(dates, errors='coerce').dt.tz_localize(None).dt.normalize()


def form_status(value):
    if value < config.TSB_OVERTRAINING_UPPER:
        return 'Overtraining Risk'
    if value < config.TSB_OPTIMAL_UPPER:
        return 'Optimal Training'
    if value <= config.TSB_MAINTENANCE_UPPER:
        return 'Maintenance Zone'
    return 'Fresh / Recovery'


def datum(value):
    return f'{value:.2f}' if pd.notna(value) else '—'


def style_chart(chart):
    return (
        chart.properties(height=300)
        .configure(font=FONT)
        .configure_axis(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=UI_TEXT,
            titleColor=UI_TEXT,
            labelFontSize=12,
            titleFontSize=12,
            gridColor=GRID,
            domain=False,
        )
        .configure_legend(
            labelFont=FONT,
            labelColor=UI_TEXT,
            labelFontSize=12,
            orient='bottom',
        )
        .configure_view(stroke=None)
    )


def date_axis():
    return alt.X(
        'Date:T',
        title=None,
        axis=alt.Axis(
            format='%d %b %y',
            tickCount=7,
            labelOverlap='greedy',
            grid=False,
        ),
    )


def date_marker(selected, x):
    return alt.Chart(pd.DataFrame({'Date': [selected]})).mark_rule(
        color=UI_TEXT,
        strokeDash=[4, 4],
        strokeWidth=1.5,
        opacity=0.7,
    ).encode(x=x)


def plot_training_load(days, selected):
    metrics = ['Fitness (CTL)', 'Fatigue (ATL)']
    data = (
        days[['Date', 'CTL', 'ATL']]
        .rename(columns={'CTL': metrics[0], 'ATL': metrics[1]})
        .melt(id_vars='Date', var_name='Metric', value_name='Load')
    )
    x = date_axis()
    line = alt.Chart(data).mark_line(strokeWidth=2.4, point=len(days) == 1).encode(
        x=x,
        y=alt.Y('Load:Q', title='Training load'),
        color=alt.Color(
            'Metric:N',
            title=None,
            scale=alt.Scale(
                domain=metrics,
                range=[
                    getattr(config, 'CHART_SECONDARY', '#4D769B'),
                    getattr(config, 'STRAVA_ACCENT', '#FC4C02'),
                ],
            ),
        ),
        tooltip=[
            alt.Tooltip('Date:T', format='%d %b %Y'),
            alt.Tooltip('Metric:N'),
            alt.Tooltip('Load:Q', format='.2f'),
        ],
    )
    return style_chart(line + date_marker(selected, x))


def plot_form(days, selected):
    data = days[['Date', 'TSB']]
    values = data['TSB'].dropna()
    balance = config.TSB_MAINTENANCE_UPPER
    low, high = min(values.min(), balance), max(values.max(), balance)
    padding = max(2, (high - low) * 0.08)
    low, high = low - padding, high + padding
    scale = alt.Scale(domain=[low, high], nice=False, zero=False)

    thresholds = [
        -np.inf,
        config.TSB_OVERTRAINING_UPPER,
        config.TSB_OPTIMAL_UPPER,
        balance,
        np.inf,
    ]
    bands = pd.DataFrame({
        'Zone': list(reversed(ZONE_COLORS)),
        'Lower': [max(low, value) for value in thresholds[:-1]],
        'Upper': [min(high, value) for value in thresholds[1:]],
    })
    bands = bands[bands['Lower'] < bands['Upper']]

    shading = alt.Chart(bands).mark_rect(opacity=0.30).encode(
        x=alt.value(0),
        x2=alt.value('width'),
        y=alt.Y('Lower:Q', title='Training stress balance', scale=scale),
        y2='Upper:Q',
        color=alt.Color(
            'Zone:N',
            title=None,
            scale=alt.Scale(
                domain=list(ZONE_COLORS),
                range=list(ZONE_COLORS.values()),
            ),
            legend=alt.Legend(symbolType='square', symbolOpacity=1, columns=2),
        ),
    )
    x = date_axis()
    trace = alt.Chart(data).mark_line(
        color=UI_TEXT,
        strokeWidth=2.6,
        point=len(values) == 1,
    ).encode(
        x=x,
        y=alt.Y('TSB:Q', title='Training stress balance', scale=scale),
        tooltip=[
            alt.Tooltip('Date:T', format='%d %b %Y'),
            alt.Tooltip('TSB:Q', title='Form (TSB)', format='.2f'),
        ],
    )
    return style_chart(shading + date_marker(selected, x) + trace)


st.title('Fitness Fatigue Form')
st.caption('Training load, fatigue, and form over time.')

summary_df = st.session_state['summary_df'] if 'summary_df' in st.session_state else parse_csv()
if summary_df.empty:
    st.info('No training history is available.')
    st.stop()

trimps = parse_granular(summary_df.copy()).copy()
if trimps.empty:
    st.info('No daily training load is available.')
    st.stop()

trimps['Date'] = calendar_days(trimps['Date'])
trimps = trimps.dropna(subset=['Date']).sort_values('Date', kind='stable')
if trimps.empty:
    st.info('No daily training load has a valid date.')
    st.stop()

range_slot, date_slot = st.columns([1.1, 1], gap='medium')
with range_slot:
    selected_range = st.segmented_control(
        'Chart range',
        list(RANGES),
        default='6 months',
        key='fitness_chart_range',
    ) or '6 months'

end = trimps['Date'].iloc[-1]
months = RANGES[selected_range]
visible = trimps if months is None else trimps.loc[
    trimps['Date'].ge(end - pd.DateOffset(months=months))
]
available_dates = visible['Date'].dt.date.drop_duplicates().tolist()

with date_slot:
    if len(available_dates) > 1:
        date = st.select_slider(
            'Date',
            options=available_dates,
            value=available_dates[-1],
            format_func=lambda day: day.strftime('%d %b %Y'),
            key='fitness_date',
        )
    else:
        date = available_dates[0]
        st.caption(date.strftime('%d %b %Y'))

selected_date = pd.Timestamp(date)
day = visible.loc[visible['Date'].eq(selected_date)].iloc[0]

ctl_col, atl_col, tsb_col = st.columns(3, gap='small')
with ctl_col.container(border=True):
    st.metric('Fitness (CTL)', datum(day['CTL']))

with atl_col.container(border=True):
    st.metric('Fatigue (ATL)', datum(day['ATL']))

with tsb_col.container(border=True):
    st.metric('Form (TSB)', datum(day['TSB']))
    if pd.notna(day['TSB']):
        status = form_status(day['TSB'])
        shade = ZONE_COLORS[status]
        st.html(
            f'<span style="display:inline-block; color:{UI_TEXT}; '
            f'background:{shade}30; border:1px solid {shade}; border-radius:999px; '
            f'padding:0.15rem 0.55rem; font-size:0.85rem">{status}</span>'
        )

st.caption('CTL and ATL show end-of-day load. TSB shows form at the start of the selected day.')

fitness_tab, form_tab = st.tabs(['Fitness and fatigue', 'Training stress balance'])
with fitness_tab:
    st.altair_chart(plot_training_load(visible, selected_date), width='stretch')

with form_tab:
    if visible['TSB'].notna().any():
        st.altair_chart(plot_form(visible, selected_date), width='stretch')
    else:
        st.caption('No form values are available for this range.')

with st.expander('Methodology'):
    st.markdown("""
| Metric | Meaning |
| --- | --- |
| CTL · Chronic training load | Longer-term fitness estimate, weighted toward the past 42 days |
| ATL · Acute training load | Recent fatigue estimate, weighted toward the past 7 days |
| TSB · Training stress balance | Previous day’s CTL minus ATL, used as the current day’s form |

CTL and ATL are exponentially weighted averages of daily training load. The form test
checks whether adding start-of-day TSB or CTL improves on the rolling efficiency baseline
for later runs.
""")
