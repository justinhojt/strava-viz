from contextlib import contextmanager

import streamlit as st


@contextmanager
def metric_card(parent, title=None):
    with parent.container(border=True):
        if title is not None:
            st.markdown(f'## {title}')
        yield


def metric_cards(items):
    columns = st.columns(len(items), gap='small')
    for column, (label, value) in zip(columns, items):
        with metric_card(column):
            st.metric(label, value)


def styled_table(frame, formats, theme, *, highlight_min=None):
    table = (
        frame.style.hide(axis='index')
        .format(formats, na_rep='—')
        .set_table_styles([
            {'selector': '', 'props': 'width: 100%; border-collapse: collapse;'},
            {'selector': 'th, td', 'props': (
                'text-align: right; padding: 0.55rem 0.7rem; '
                f"border: 1px solid {theme['table_border']}; white-space: nowrap;"
            )},
            {'selector': 'th', 'props': (
                f"background-color: {theme['table_header']}; font-weight: 600;"
            )},
            {'selector': 'th:first-child, td:first-child', 'props': 'text-align: left;'},
        ])
    )
    if highlight_min is not None:
        table = table.highlight_min(
            axis=1,
            subset=highlight_min,
            props=(
                f"background-color: {theme['table_highlight']}; "
                f"color: {theme['table_highlight_text']}; font-weight: 600;"
            ),
        )
    return f'<div style="overflow-x:auto">{table.to_html()}</div>'
