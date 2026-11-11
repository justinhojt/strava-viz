import altair as alt
import pandas as pd
import config

# Helper function to filter dataframe to the last recorded activity date
def filter_to_last_activity(df):
    if 'trimps' in df.columns:
        last_active_date = df[df['trimps'] > 0]['Date'].max()
        if pd.notna(last_active_date):
            return df[df['Date'] <= last_active_date]
    return df

# Plots aerobic efficiency chart with 42-day moving average
def plot_aero(df):
    df = df.sort_values('Activity Date')
    df['moving_avg'] = df.rolling('42D', on='Activity Date')['aero_ratio'].mean()
    
    df['graph_date'] = df['Activity Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    base = alt.Chart(df).encode(
        x=alt.X('graph_date:T', title='Date')
    )

    points = base.mark_circle(
        size=60, 
        fill=config.COLOR_PURE_WHITE, 
        stroke=config.COLOR_STRAVA_ORANGE, 
        strokeWidth=1.5, 
        opacity=0.7
    ).encode(
        y=alt.Y('aero_ratio:Q', title='Ratio (Grade-adjusted Pace/Heart Rate Reserve)', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('graph_date:T', title='Date', format='%Y-%m-%d'), 'aero_ratio:Q']
    )

    trend_line = base.mark_line(color=config.COLOR_STRAVA_ORANGE, size=3).encode(
        y=alt.Y('moving_avg:Q')
    )

    return alt.layer(points, trend_line).properties(height=400)
    
# Plots fitness (chronic training load) and fatigue (acute training load)
def plot_fitness_fatigue(df, selected_date=None):
    df = filter_to_last_activity(df)

    base = df.melt(id_vars=['Date'], value_vars=['CTL', 'ATL'], 
                   var_name='Metric', value_name='Value')

    base['Metric_Label'] = base['Metric'].map({'CTL': 'Fitness (CTL)', 'ATL': 'Fatigue (ATL)'})

    chart = alt.Chart(base).mark_line(strokeWidth=2).encode(
        x=alt.X('Date:T', title='Date', axis=alt.Axis(format="%b '%y")),
        y=alt.Y('Value:Q', title='Stress Units', scale=alt.Scale(zero=False)),
        color=alt.Color('Metric_Label:N', 
                        scale=alt.Scale(domain=['Fitness (CTL)', 'Fatigue (ATL)'], 
                                        range=[config.COLOR_FITNESS, config.COLOR_FATIGUE]),
                        title='Metric'),
        tooltip=[
            alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('Metric_Label:N', title='Metric'),
            alt.Tooltip('Value:Q', title='Stress Units', format='.2f')
        ]
    )

    # Add vertical sync line if a date is provided
    if selected_date:
        vline = alt.Chart(pd.DataFrame({'Date': [pd.Timestamp(selected_date)]})).mark_rule(
            color=config.COLOR_PURE_WHITE, strokeWidth=1
        ).encode(x='Date:T')
        return alt.layer(chart, vline).properties(height=350)

    return chart.properties(height=350)

# Plots training stress balance with training zones
def plot_tsb_zones(df, selected_date=None):
    df = filter_to_last_activity(df)

    max_tsb = float(max(df['TSB'].max(), 15) + 10)
    min_tsb = float(min(df['TSB'].min(), -35) - 10)

    zone_data = pd.DataFrame([
        {'y1': config.TSB_MAINTENANCE_UPPER, 'y2': max_tsb, 'color': config.ZONE_COLORS['Freshness']},        
        {'y1': config.TSB_OPTIMAL_UPPER, 'y2': config.TSB_MAINTENANCE_UPPER, 'color': config.ZONE_COLORS['Maintenance']},        
        {'y1': config.TSB_OVERTRAINING_UPPER, 'y2': config.TSB_OPTIMAL_UPPER, 'color': config.ZONE_COLORS['Optimal_Training']},   
        {'y1': min_tsb, 'y2': config.TSB_OVERTRAINING_UPPER, 'color': config.ZONE_COLORS['Overtraining']}     
    ])

    zones = alt.Chart(zone_data).mark_rect(opacity=0.25).encode(
        y=alt.Y('y1:Q', title='Form (TSB)'),
        y2='y2:Q',
        color=alt.Color('color:N', scale=None),
        tooltip=alt.value(None)
    )

    tsb_line = alt.Chart(df).mark_line(color=config.COLOR_PURE_WHITE, strokeWidth=2).encode(
        x=alt.X('Date:T', title='Date', axis=alt.Axis(format="%b '%y")),
        y=alt.Y('TSB:Q', title='Form (TSB)')
    )

    baseline = alt.Chart(pd.DataFrame([{'y': 0}])).mark_rule(
        color=config.COLOR_MUTED_GREY, strokeDash=[4, 4]
    ).encode(y='y:Q')

    layers = [zones, tsb_line, baseline]

    # Add vertical sync line if a date is provided
    if selected_date:
        vline = alt.Chart(pd.DataFrame({'Date': [pd.Timestamp(selected_date)]})).mark_rule(
            color=config.COLOR_PURE_WHITE, strokeWidth=1 
        ).encode(x='Date:T')
        layers.append(vline)

    return alt.layer(*layers).properties(height=350)

# Plots activity breakdown donut chart
def plot_donut(df):
    breakdown = df['Activity Type'].value_counts().reset_index()
    breakdown.columns = ['Activity', 'Count']
    
    # Align colours with the activities present in the current dataframe
    present_activities = breakdown['Activity'].tolist()
    chart_range = [config.ACTIVITY_COLORS.get(act, config.ACTIVITY_COLORS['Default']) for act in present_activities]
    
    donut_chart = alt.Chart(breakdown).mark_arc(innerRadius=60).encode(
        theta=alt.Theta(field='Count', type='quantitative'),
        color=alt.Color(
            field='Activity', 
            type='nominal', 
            scale=alt.Scale(domain=present_activities, range=chart_range),
            legend=alt.Legend(title='Activity Breakdown', orient='right')
        ),
        tooltip=['Activity', 'Count']
    ).properties(height=220)
    
    return donut_chart

# Plots a horizontal bar chart of time spent in each heart rate zone
def plot_hr_zones(zone_counts, labels):
    zone_colors = alt.Scale(
        domain=labels,
        range=config.HR_ZONE_COLORS 
    )
    
    zone_chart = alt.Chart(zone_counts).mark_bar(cornerRadiusEnd=2, height=18).encode(
        y=alt.Y('Zone:N', sort=labels, title=None, axis=alt.Axis(labelAngle=0, grid=False)),
        x=alt.X('Minutes:Q', title='Time (Minutes)'),
        color=alt.Color('Zone:N', scale=zone_colors, legend=None),
        tooltip=[
            alt.Tooltip('Zone:N', title='Zone'),
            alt.Tooltip('Minutes:Q', title='Minutes', format='.1f')
        ]
    ).properties(height=200)
    return zone_chart

# Plots a line chart of heart rate over time
def plot_hr_series(df):
    return alt.Chart(df).mark_line(color=config.COLOR_STRAVA_ORANGE).encode(
        x=alt.X('graph_timestamp:T', title='Time'),
        y=alt.Y('heart_rate:Q', title='Heart Rate (bpm)', scale=alt.Scale(zero=False))
    )

# Plots a line chart of elevation over time
def plot_ele_series(df):
    return alt.Chart(df).mark_line(color=config.COLOR_STRAVA_ORANGE).encode(
        x=alt.X('graph_timestamp:T', title='Time'),
        y=alt.Y('elevation:Q', title='Elevation (m)', scale=alt.Scale(zero=False))
    )

# Plots silhouette score for each candidate k
def plot_silhouette(scores_df):
    return alt.Chart(scores_df).mark_line(
        color=config.COLOR_STRAVA_ORANGE, point=True, strokeWidth=2
    ).encode(
        x=alt.X('k:O', title='Number of clusters (k)', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('score:Q', title='Silhouette score', scale=alt.Scale(zero=False)),
        tooltip=['k:O', alt.Tooltip('score:Q', format='.3f')]
    ).properties(height=250)

# Plots aerobic efficiency coloured by cluster, with a 42-day trend line per cluster
def plot_cluster_efficiency(df):
    df = df.sort_values('Activity Date').copy()
    df['moving_avg'] = df.groupby('Cluster Name', group_keys=False).apply(
        lambda g: g.rolling('42D', on='Activity Date')['aero_ratio'].mean()
    )
    df['graph_date'] = df['Activity Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    base = alt.Chart(df).encode(
        x=alt.X('graph_date:T', title='Date'),
        color=alt.Color('Cluster Name:N', title='Run type')
    )

    points = base.mark_circle(size=50, opacity=0.5).encode(
        y=alt.Y('aero_ratio:Q', title='Efficiency ratio', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('graph_date:T', title='Date', format='%Y-%m-%d'),
                 'Cluster Name:N', 'aero_ratio:Q']
    )
    lines = base.mark_line(size=3).encode(y='moving_avg:Q')

    return alt.layer(points, lines).properties(height=400)
