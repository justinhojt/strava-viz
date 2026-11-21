import pandas as pd
import streamlit as st

import config
from utils.data_loader import get_summary
from utils.functions import parse_granular
from utils.plots import filter_to_last_activity, get_theme, plot_fitness_fatigue, plot_tsb_zones
from utils.ui import metric_card

RANGES = {'3 months': 3, '6 months': 6, '1 year': 12, 'All': None}


def form_status(value):
    if value < config.TSB_OVERTRAINING_UPPER:
        return 'Overtraining Risk'
    if value < config.TSB_OPTIMAL_UPPER:
        return 'Optimal Training'
    if value <= config.TSB_MAINTENANCE_UPPER:
        return 'Maintenance Zone'
    return 'Fresh / Recovery'


st.title('Fitness, Fatigue & Form')
st.caption('Training load, fatigue, and form over time.')

summary_df = get_summary()
if summary_df.empty:
    st.info('No training history is available.')
    st.stop()

trimps = parse_granular(summary_df.copy())
if trimps.empty:
    st.info('No daily training load is available.')
    st.stop()

trimps['Date'] = pd.to_datetime(trimps['Date'], errors='coerce').dt.normalize()
trimps = trimps.dropna(subset=['Date']).sort_values('Date', kind='stable')
if trimps.empty:
    st.info('No daily training load has a valid date.')
    st.stop()

# End the timeline on the last day with training load (parse_granular pads out to today)
trimps = filter_to_last_activity(trimps)

# ---------- Controls ----------
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

# ---------- KPI cards ----------
theme = get_theme()
ctl_col, atl_col, tsb_col = st.columns(3, gap='small')

with metric_card(ctl_col):
    st.metric('Fitness (CTL)', f"{day['CTL']:.2f}")

with metric_card(atl_col):
    st.metric('Fatigue (ATL)', f"{day['ATL']:.2f}")

with metric_card(tsb_col):
    st.metric('Form (TSB)', f"{day['TSB']:.2f}")
    status = form_status(day['TSB'])
    shade = config.TSB_ZONE_COLORS[status]
    st.html(
        f'<span style="display:inline-block; color:{theme["text"]}; '
        f'background:{shade}30; border:1px solid {shade}; border-radius:999px; '
        f'padding:0.15rem 0.55rem; font-size:0.85rem">{status}</span>'
    )

st.caption('CTL and ATL show end-of-day load. TSB shows form going into the selected day '
           "(the previous day's CTL − ATL).")

# ---------- Charts ----------
fitness_tab, form_tab = st.tabs(['Fitness and fatigue', 'Training stress balance'])
with fitness_tab:
    st.altair_chart(
        plot_fitness_fatigue(visible, selected_date, theme=theme),
        width='stretch',
    )

with form_tab:
    st.altair_chart(
        plot_tsb_zones(visible, selected_date, theme=theme),
        width='stretch',
    )

with st.expander('Methodology'):
    st.markdown(f"""
| Metric | Meaning |
| --- | --- |
| CTL · Chronic training load | Longer-term fitness estimate, weighted toward the past {config.CTL_DAY_SPAN} days |
| ATL · Acute training load | Recent fatigue estimate, weighted toward the past {config.ATL_DAY_SPAN} days |
| TSB · Training stress balance | Previous day’s CTL minus ATL, used as the current day’s form |

CTL and ATL are exponentially weighted averages of daily training load.
""")
