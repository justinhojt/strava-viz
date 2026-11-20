import math
from itertools import cycle
from math import inf

import altair as alt
import pandas as pd
import streamlit as st

import config


SERIES_HEIGHT = 224
CLUSTER = 'Cluster Name'


def get_theme():
    base = st.get_option('theme.base') or 'light'
    mode = st.context.theme.type or base
    colors = config.CHART_NEUTRALS[mode]
    text = st.get_option(f'theme.{mode}.textColor')
    if text is None and mode == base:
        text = st.get_option('theme.textColor')
    return {**colors, 'text': text or colors['text']}


def style_chart(chart, theme=None, *, height=None, legend_orient='bottom'):
    theme = get_theme() if theme is None else theme
    font = config.CHART_FONT
    if height is not None:
        chart = chart.properties(height=height)

    legend_direction = 'vertical' if legend_orient in ('left', 'right') else 'horizontal'
    return (
        chart.configure(font=font)
        .configure_axis(
            labelFont=font, titleFont=font, labelColor=theme['text'],
            titleColor=theme['text'], labelFontSize=12, titleFontSize=12,
            gridColor=theme['grid'], tickColor=theme['grid'], domain=False,
        )
        .configure_legend(
            labelFont=font, labelColor=theme['text'], labelFontSize=12,
            orient=legend_orient, direction=legend_direction,
        )
        .configure_view(stroke=None)
    )


def filter_to_last_activity(df):
    if 'trimps' in df.columns:
        last_active_date = df.loc[df['trimps'] > 0, 'Date'].max()
        if pd.notna(last_active_date):
            return df.loc[df['Date'] <= last_active_date]
    return df


def plot_donut(df, *, theme=None):
    theme = get_theme() if theme is None else theme
    counts = (
        df['Activity Type'].astype('string').fillna('Unspecified').value_counts()
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
        innerRadius=46, outerRadius=68, padAngle=0.025, strokeWidth=0,
    ).encode(
        color=alt.Color(
            'Legend:N',
            scale=alt.Scale(
                domain=counts['Legend'].tolist(),
                range=counts['Activity'].map(colors).tolist(),
            ),
            legend=alt.Legend(
                title=None, orient='right', direction='vertical', labelLimit=190,
            ),
        ),
        tooltip=[
            alt.Tooltip('Activity:N'),
            alt.Tooltip('Count:Q', title='Activities', format=',d'),
            alt.Tooltip('Share:Q', format='.1%'),
        ],
    )
    labels = base.mark_text(
        radius=83, size=11, color=theme['text'],
    ).encode(text='Slice label:N')

    center = alt.Chart(pd.DataFrame({'Total': [f'{total:,}']}))
    count = center.mark_text(
        dy=-8, size=25, fontWeight=600, color=theme['text'],
    ).encode(text='Total:N')
    caption = center.mark_text(
        dy=13, size=11, color=theme['muted'],
    ).encode(text=alt.value('activities'))

    return style_chart(
        alt.layer(arcs, labels, count, caption).properties(height=180), theme,
    )


def plot_monthly_distance(distance_runs, months, *, theme=None):
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

    chart = alt.Chart(monthly).mark_bar(
        color=config.ACCENT,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
    ).encode(
        x=alt.X(
            'yearmonth(Month):T',
            title=None,
            axis=alt.Axis(format='%b %y', tickCount=6, labelOverlap='greedy', grid=False),
        ),
        y=alt.Y('Distance km:Q', title='km', axis=alt.Axis(tickCount=4)),
        tooltip=[
            alt.Tooltip('Month:T', format='%b %Y'),
            alt.Tooltip('Distance km:Q', title='Distance (km)', format=',.1f'),
        ],
    ).properties(height=180)
    return style_chart(chart, theme)


def format_clock(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f'{minutes}:{seconds:02d}'


def _elapsed_axis(duration):
    if duration < 60:
        step = max(1, math.ceil(duration / 5))
    elif duration < 300:
        step = 60
    elif duration <= 2700:
        step = 300
    else:
        step = 600 * math.ceil(duration / 7200)

    return alt.Axis(
        values=list(range(0, int(duration) + 1, step)),
        labelExpr="floor(datum.value / 60) + ':' + format(datum.value % 60, '02d')",
        labelPadding=8,
        grid=False,
    )


def _hr_zone_color(labels):
    return alt.Color(
        'Zone:N',
        scale=alt.Scale(domain=list(labels), range=list(config.HR_ZONE_COLORS)),
        legend=None,
    )


def _elapsed_activity(data):
    if {'Elapsed seconds', 'Elapsed'}.issubset(data.columns):
        return data

    field = 'timestamp' if 'timestamp' in data else 'graph_timestamp'
    timestamps = pd.to_datetime(data[field], format='mixed', errors='coerce', utc=True)
    activity = (
        data.assign(timestamp=timestamps)
        .dropna(subset=['timestamp'])
        .sort_values('timestamp', kind='stable')
    )
    seconds = (activity['timestamp'] - activity['timestamp'].min()).dt.total_seconds()
    return activity.assign(**{
        'Elapsed seconds': seconds,
        'Elapsed': seconds.map(format_clock),
    })


def _plot_activity_series(data, metric, unit, *, bands=None, theme=None):
    theme = get_theme() if theme is None else theme
    data = _elapsed_activity(data)
    duration = max(1, float(data['Elapsed seconds'].max()))
    x = alt.X(
        'Elapsed seconds:Q',
        title='Elapsed time (m:ss)',
        scale=alt.Scale(domain=[0, duration], nice=False),
        axis=_elapsed_axis(duration),
    )

    y_scale = alt.Scale(zero=False)
    if bands is not None:
        low = max(0, math.floor((data[metric].min() - 4) / 5) * 5)
        high = math.ceil((data[metric].max() + 4) / 5) * 5
        y_scale = alt.Scale(domain=[low, high], nice=False)
        visible = bands.assign(
            Lower=bands['Lower'].clip(lower=low),
            Upper=bands['Upper'].clip(upper=high),
        )
        visible = visible.loc[visible['Lower'] < visible['Upper']]

    line = alt.Chart(data[['Elapsed seconds', 'Elapsed', metric]]).mark_line(
        color=theme['text'], strokeWidth=2.1, clip=True,
    ).encode(
        x=x,
        y=alt.Y(
            f'{metric}:Q', title=unit, scale=y_scale,
            axis=alt.Axis(tickCount=5, grid=True),
        ),
        tooltip=[
            alt.Tooltip('Elapsed:N'),
            alt.Tooltip(f'{metric}:Q', title=unit, format='.0f'),
        ],
    )

    if bands is None:
        return style_chart(line.properties(height=SERIES_HEIGHT), theme)

    shading = alt.Chart(visible).mark_rect(opacity=0.15).encode(
        x=alt.value(0),
        x2=alt.value('width'),
        y=alt.Y('Lower:Q', scale=y_scale, title=unit),
        y2='Upper:Q',
        color=_hr_zone_color(config.HR_ZONE_LABELS),
    )
    return style_chart((shading + line).properties(height=SERIES_HEIGHT), theme)


def plot_hr_series(df, *, bands=None, theme=None):
    return _plot_activity_series(
        df, 'heart_rate', 'Heart rate (bpm)', bands=bands, theme=theme,
    )


def plot_ele_series(df, *, theme=None):
    return _plot_activity_series(df, 'elevation', 'Elevation (m)', theme=theme)


def plot_hr_zones(zone_counts, labels, *, theme=None):
    theme = get_theme() if theme is None else theme
    zones = zone_counts.assign(
        Label=zone_counts['Minutes'].map(lambda minutes: f'{minutes:.1f} min'),
    )
    base = alt.Chart(zones).encode(
        y=alt.Y(
            'Zone:N', sort=labels,
            axis=alt.Axis(title=None, ticks=False, domain=False, labelPadding=8),
        ),
        x=alt.X(
            'Minutes:Q', axis=None,
            scale=alt.Scale(domain=[0, max(1, zones['Minutes'].max() * 1.28)], nice=False),
        ),
    )
    bars = base.mark_bar(size=22, cornerRadiusEnd=4).encode(
        color=_hr_zone_color(labels),
        tooltip=[alt.Tooltip('Zone:N'), alt.Tooltip('Minutes:Q', format='.2f')],
    )
    values = base.mark_text(
        align='left', baseline='middle', dx=6, size=12, color=theme['text'],
    ).encode(text='Label:N')

    return style_chart(
        (bars + values).properties(height=max(150, 29 * len(labels))), theme,
    )


def plot_aero(df, *, theme=None):
    theme = get_theme() if theme is None else theme
    runs = df if 'cv_time' in df else df.rename(columns={'Activity Date': 'cv_time'})
    dates = pd.to_datetime(runs['cv_time'], format='mixed', errors='coerce', utc=True)
    runs = (
        runs.assign(cv_time=dates)
        .dropna(subset=['cv_time', 'aero_ratio'])
        .sort_values('cv_time', kind='stable')
    )
    if 'rolling_ratio' not in runs:
        runs['rolling_ratio'] = runs.rolling('42D', on='cv_time')['aero_ratio'].mean()

    trend = runs.drop_duplicates('cv_time', keep='last')
    series = alt.Color(
        'Series:N',
        scale=alt.Scale(
            domain=['Individual run', '42-day average'],
            range=[theme['muted'], config.ACCENT],
        ),
        legend=alt.Legend(title=None),
    )
    x = alt.X(
        'cv_time:T', title=None,
        axis=alt.Axis(format='%b %Y', tickCount=7, labelAngle=0, grid=False),
    )
    y_axis = alt.Axis(titleAngle=0, titleAlign='left', titleX=0, titleY=-10)
    scale = alt.Scale(zero=False)

    points = (
        alt.Chart(runs)
        .transform_calculate(Series="'Individual run'")
        .mark_circle(size=32, opacity=0.45)
        .encode(
            x=x,
            y=alt.Y('aero_ratio:Q', title='Efficiency ratio', scale=scale, axis=y_axis),
            color=series,
            tooltip=[
                alt.Tooltip('cv_time:T', title='Run date', format='%d %b %Y'),
                alt.Tooltip('aero_ratio:Q', title='Efficiency ratio', format='.2f'),
            ],
        )
    )
    average = (
        alt.Chart(trend)
        .transform_calculate(Series="'42-day average'")
        .mark_line(strokeWidth=2.8, point=alt.OverlayMarkDef(size=13, filled=True))
        .encode(
            x=x,
            y=alt.Y('rolling_ratio:Q', title='Efficiency ratio', scale=scale, axis=y_axis),
            color=series,
            tooltip=[
                alt.Tooltip('cv_time:T', title='Run date', format='%d %b %Y'),
                alt.Tooltip('rolling_ratio:Q', title='42-day average', format='.2f'),
            ],
        )
    )
    return style_chart((points + average).properties(height=360), theme)


def plot_prediction_error(mean_mae, *, theme=None):
    theme = get_theme() if theme is None else theme
    errors = mean_mae.rename_axis('Model').reset_index(name='MAE')
    errors['Label'] = errors['MAE'].map('{:.2f}'.format)
    displayed = errors['MAE'].round(2)
    errors['Lowest error'] = displayed.eq(displayed.min())

    base = alt.Chart(errors).encode(
        y=alt.Y('Model:N', sort=errors['Model'].tolist(), axis=alt.Axis(title=None)),
        x=alt.X(
            'MAE:Q', axis=None,
            scale=alt.Scale(domain=[0, max(1, errors['MAE'].max() * 1.18)], nice=False),
        ),
    )
    bars = base.mark_bar(size=23, cornerRadiusEnd=4).encode(
        color=alt.condition(
            alt.datum['Lowest error'], alt.value(config.ACCENT), alt.value(theme['muted']),
        ),
        tooltip=[
            alt.Tooltip('Model:N'),
            alt.Tooltip('MAE:Q', title='MAE (bpm)', format='.2f'),
        ],
    )
    labels = base.mark_text(
        align='left', dx=6, color=theme['text'], size=12,
    ).encode(text='Label:N')
    return style_chart((bars + labels).properties(height=205), theme)


def plot_heat_effect(heat_index, adjusted_bpm, fitted_bpm, *, theme=None):
    theme = get_theme() if theme is None else theme
    points = pd.DataFrame({
        'Heat index': heat_index.to_numpy(),
        'Heart rate': adjusted_bpm.to_numpy(),
    })
    fitted = pd.DataFrame({
        'Heat index': heat_index.to_numpy(),
        'Heart rate': fitted_bpm.to_numpy(),
    }).sort_values('Heat index')

    x = alt.X(
        'Heat index:Q', title='Heat index (°HI)', scale=alt.Scale(zero=False),
        axis=alt.Axis(tickCount=5, grid=False),
    )
    y = alt.Y('Heart rate:Q', title='Heart rate (bpm)', scale=alt.Scale(zero=False))

    dots = alt.Chart(points).mark_circle(
        color=theme['muted'], opacity=0.55, size=35,
    ).encode(
        x=x,
        y=y,
        tooltip=[
            alt.Tooltip('Heat index:Q', format='.2f'),
            alt.Tooltip('Heart rate:Q', title='Adjusted heart rate (bpm)', format='.2f'),
        ],
    )
    line = alt.Chart(fitted).mark_line(color=config.ACCENT, strokeWidth=2.5).encode(
        x=x, y=y,
    )
    return style_chart((dots + line).properties(height=205), theme)


def _load_date_axis():
    return alt.X(
        'Date:T', title=None,
        axis=alt.Axis(format='%d %b %y', tickCount=7, labelOverlap='greedy', grid=False),
    )


def _date_marker(selected_date, x, theme):
    return alt.Chart(pd.DataFrame({'Date': [pd.Timestamp(selected_date)]})).mark_rule(
        color=theme['text'], strokeDash=[4, 4], strokeWidth=1.5, opacity=0.7,
    ).encode(x=x)


def plot_fitness_fatigue(df, selected_date=None, *, theme=None, trim_to_last_activity=True):
    theme = get_theme() if theme is None else theme
    days = filter_to_last_activity(df) if trim_to_last_activity else df
    metrics = ['Fitness (CTL)', 'Fatigue (ATL)']
    data = (
        days[['Date', 'CTL', 'ATL']]
        .rename(columns={'CTL': metrics[0], 'ATL': metrics[1]})
        .melt(id_vars='Date', var_name='Metric', value_name='Load')
    )

    x = _load_date_axis()
    line = alt.Chart(data).mark_line(strokeWidth=2.4, point=len(days) == 1).encode(
        x=x,
        y=alt.Y('Load:Q', title='Training load'),
        color=alt.Color(
            'Metric:N', title=None,
            scale=alt.Scale(domain=metrics, range=list(config.LOAD_LINE_COLORS)),
        ),
        tooltip=[
            alt.Tooltip('Date:T', format='%d %b %Y'),
            alt.Tooltip('Metric:N'),
            alt.Tooltip('Load:Q', format='.2f'),
        ],
    )
    if selected_date is not None:
        line = line + _date_marker(selected_date, x, theme)

    return style_chart(line.properties(height=300), theme)


def plot_tsb_zones(df, selected_date=None, *, theme=None, trim_to_last_activity=True):
    theme = get_theme() if theme is None else theme
    days = filter_to_last_activity(df) if trim_to_last_activity else df
    data = days[['Date', 'TSB']]
    values = data['TSB'].dropna()

    balance = config.TSB_MAINTENANCE_UPPER
    low, high = balance, balance
    if not values.empty:
        low, high = min(values.min(), balance), max(values.max(), balance)

    padding = max(2, (high - low) * 0.08)
    low, high = low - padding, high + padding
    scale = alt.Scale(domain=[low, high], nice=False, zero=False)

    thresholds = [
        -inf, config.TSB_OVERTRAINING_UPPER, config.TSB_OPTIMAL_UPPER, balance, inf,
    ]
    zone_colors = config.TSB_ZONE_COLORS
    bands = pd.DataFrame({
        'Zone': list(reversed(zone_colors)),
        'Lower': [max(low, value) for value in thresholds[:-1]],
        'Upper': [min(high, value) for value in thresholds[1:]],
    })
    bands = bands.loc[bands['Lower'] < bands['Upper']]

    shading = alt.Chart(bands).mark_rect(opacity=0.30).encode(
        x=alt.value(0),
        x2=alt.value('width'),
        y=alt.Y('Lower:Q', title='Training stress balance', scale=scale),
        y2='Upper:Q',
        color=alt.Color(
            'Zone:N', title=None,
            scale=alt.Scale(domain=list(zone_colors), range=list(zone_colors.values())),
            legend=alt.Legend(symbolType='square', symbolOpacity=1, columns=2),
        ),
    )

    x = _load_date_axis()
    trace = alt.Chart(data).mark_line(
        color=theme['text'], strokeWidth=2.6, point=len(values) == 1,
    ).encode(
        x=x,
        y=alt.Y('TSB:Q', title='Training stress balance', scale=scale),
        tooltip=[
            alt.Tooltip('Date:T', format='%d %b %Y'),
            alt.Tooltip('TSB:Q', title='Form (TSB)', format='.2f'),
        ],
    )

    chart = shading
    if selected_date is not None:
        chart = chart + _date_marker(selected_date, x, theme)

    return style_chart((chart + trace).properties(height=300), theme)


def plot_cluster_efficiency(df, colors=None, *, theme=None):
    theme = get_theme() if theme is None else theme
    columns = {}
    if 'Date' not in df:
        columns['Activity Date'] = 'Date'
    if 'Efficiency ratio' not in df:
        columns['aero_ratio'] = 'Efficiency ratio'

    points = df.rename(columns=columns)
    dates = pd.to_datetime(points['Date'], format='mixed', errors='coerce', utc=True)
    points = points.assign(Date=dates).dropna(subset=['Date', CLUSTER, 'Efficiency ratio'])

    if colors is None:
        names = sorted(points[CLUSTER].unique())
        colors = dict(zip(names, cycle(config.CHART_CATEGORICAL_COLORS)))

    series = []
    for _, group in points.groupby(CLUSTER, sort=False):
        group = group.sort_values('Date', kind='stable').copy()
        group['42-day average'] = group.rolling('42D', on='Date')['Efficiency ratio'].mean()
        series.append(group.drop_duplicates('Date', keep='last'))

    trend = (
        pd.concat(series) if series
        else points.assign(**{'42-day average': float('nan')})
    )
    active_names = set(points[CLUSTER])
    names = [name for name in colors if name in active_names]
    color = alt.Color(
        f'{CLUSTER}:N', title=None,
        scale=alt.Scale(domain=names, range=[colors[name] for name in names]),
        legend=alt.Legend(symbolType='square', symbolOpacity=1),
    )
    x = alt.X(
        'Date:T', title=None,
        axis=alt.Axis(
            format='%b %Y', tickCount=7, labelAngle=0, labelOverlap='greedy', grid=False,
        ),
    )
    y_scale = alt.Scale(zero=False)

    dots = alt.Chart(points).mark_circle(size=33, opacity=0.40).encode(
        x=x,
        y=alt.Y('Efficiency ratio:Q', scale=y_scale),
        color=color,
        tooltip=[
            alt.Tooltip('Date:T', format='%d %b %Y'),
            alt.Tooltip(f'{CLUSTER}:N'),
            alt.Tooltip('Efficiency ratio:Q', format='.2f'),
        ],
    )
    lines = alt.Chart(trend).mark_line(strokeWidth=2.6, opacity=1).encode(
        x=x,
        y=alt.Y('42-day average:Q', title='Efficiency ratio', scale=y_scale),
        color=color,
        tooltip=[
            alt.Tooltip('Date:T', format='%d %b %Y'),
            alt.Tooltip(f'{CLUSTER}:N'),
            alt.Tooltip('42-day average:Q', format='.2f'),
        ],
    )
    return style_chart((dots + lines).properties(height=345), theme)


def plot_silhouette(scores_df, best_k=None, *, theme=None):
    theme = get_theme() if theme is None else theme
    scores = scores_df
    if best_k is None:
        best_k = int(scores.loc[scores['score'].idxmax(), 'k'])

    base = alt.Chart(scores).encode(
        x=alt.X(
            'k:Q', title='Number of clusters (k)',
            scale=alt.Scale(
                domain=[scores['k'].min() - 0.5, scores['k'].max() + 0.5],
                nice=False,
            ),
            axis=alt.Axis(values=scores['k'].tolist(), format='d', grid=False),
        ),
        y=alt.Y('score:Q', title='Silhouette score', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('k:Q', format='d'), alt.Tooltip('score:Q', format='.3f')],
    )
    line = base.mark_line(color=config.CHART_SECONDARY, point=True)
    best = base.transform_filter(alt.datum.k == best_k)
    point = best.mark_circle(color=config.ACCENT, size=120, opacity=1)
    label = best.mark_text(dy=-15, color=theme['text'], size=12).encode(
        text=alt.value(f'Best k = {best_k}'),
    )
    return style_chart((line + point + label).properties(height=215, padding={'top': 20}), theme)
