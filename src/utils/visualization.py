import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter


def to_count_ts(df, date_col: str = "created_at"):
    """Return a timeseries data with count of data for each date."""
    df[date_col] = pd.to_datetime(df[date_col], format="mixed", utc=True)
    dates = [t.to_pydatetime().date() for t in df[date_col].to_list()]
    date_counts = Counter(dates)
    timeseries = sorted(date_counts.items())
    count_ts = pd.DataFrame(timeseries, columns=["date", "count"])
    return count_ts


def plot_bar(
    df,
    col: str,
    show_freq: bool = True,
    config: dict = None,
    figsize: tuple = (1200, 400),
    font_size: int = 10,
    params: dict = dict(),
):
    df_copy = df.copy(deep=True)
    df_copy[col] = df_copy[col].dropna(axis=0)
    value_counts = df_copy[col].value_counts().reset_index()

    fig = px.bar(
        value_counts, x=col, y="count", text="count", template="simple_white", **params
    )
    fig.update_layout(
        font_family="Rockwell",
        font_size=font_size,
        autosize=True,
        margin=dict(l=50, r=50, b=0, t=0, pad=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.95),
        width=figsize[0],
        height=figsize[1],
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(
        showgrid=True,
    )
    fig.update_traces(
        textangle=0, textposition="outside", textfont_color="#a2a2a2", cliponaxis=False
    )
    fig.show(renderer="svg", config=config)


def plot_count_timeseries(
    data: list[pd.DataFrame],
    titles: list[str],
    modes: list[str],
    colors: list[str],
    start_date: str,
    end_date: str,
    width: int = 1500,
    height: int = 300,
):
    """Plot count timeseries of dataframe on the same figure."""
    # add the figure traces
    plot_data = []
    for i, df in enumerate(data):
        count_ts = to_count_ts(df)
        count_ts["timestamp"] = pd.to_datetime(count_ts["date"])
        count_ts = count_ts[
            (count_ts["timestamp"] >= start_date) & (count_ts["timestamp"] <= end_date)
        ]
        # add the time series as a line chart
        plot_data.append(
            go.Scatter(
                x=count_ts["date"],
                y=count_ts["count"],
                mode=modes[i],
                marker=dict(size=4, color=colors[i]),
                line=dict(width=1, color=colors[i]),
                showlegend=True,
                name=titles[i],
            )
        )
    layout = go.Layout(
        font_family="Rockwell",
        font_size=14,
        autosize=False,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(yanchor="top", y=-0.07, xanchor="left", x=0.68, orientation="h"),
        width=width,
        height=height,
        template="plotly_white",
    )
    # show the figure
    fig = go.Figure(data=plot_data, layout=layout)
    fig.show()
