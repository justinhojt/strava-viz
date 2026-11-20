import streamlit as st

import config


def get_theme():
    base = st.get_option('theme.base') or 'light'
    mode = st.context.theme.type or base
    colors = config.CHART_NEUTRALS[mode]
    text = st.get_option(f'theme.{mode}.textColor')
    if text is None and mode == base:
        text = st.get_option('theme.textColor')
    return {**colors, 'text': text or colors['text']}


def style_chart(chart, theme):
    font = config.CHART_FONT
    return (
        chart.configure(font=font)
        .configure_axis(
            labelFont=font,
            titleFont=font,
            labelColor=theme['text'],
            titleColor=theme['text'],
            labelFontSize=12,
            titleFontSize=12,
            gridColor=theme['grid'],
            tickColor=theme['grid'],
            domain=False,
        )
        .configure_legend(
            labelFont=font,
            labelColor=theme['text'],
            labelFontSize=12,
            orient='bottom',
            direction='horizontal',
        )
        .configure_view(stroke=None)
    )
