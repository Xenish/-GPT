"""
Chart Creation Utilities.

Base utilities for creating interactive charts using Plotly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartType(str, Enum):
    """Chart types."""

    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    CANDLESTICK = "candlestick"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    AREA = "area"


@dataclass
class ChartConfig:
    """
    Chart configuration.

    Controls chart appearance and behavior.
    """

    title: str = ""
    width: int = 1200
    height: int = 600
    template: str = "plotly_white"  # plotly, plotly_white, plotly_dark, ggplot2
    show_legend: bool = True
    show_grid: bool = True

    # Axes
    xaxis_title: str = ""
    yaxis_title: str = ""
    xaxis_type: str = "linear"  # linear, log, date
    yaxis_type: str = "linear"

    # Colors
    color_scheme: Optional[List[str]] = None

    # Export
    export_format: str = "html"  # html, png, svg, pdf

    def __post_init__(self):
        """Set default color scheme if not provided."""
        if self.color_scheme is None:
            self.color_scheme = [
                "#1f77b4",  # Blue
                "#ff7f0e",  # Orange
                "#2ca02c",  # Green
                "#d62728",  # Red
                "#9467bd",  # Purple
                "#8c564b",  # Brown
                "#e377c2",  # Pink
                "#7f7f7f",  # Gray
                "#bcbd22",  # Olive
                "#17becf",  # Cyan
            ]


def create_chart(
    chart_type: ChartType,
    data: Union[pd.DataFrame, Dict],
    config: Optional[ChartConfig] = None,
    **kwargs,
) -> go.Figure:
    """
    Create chart from data.

    Args:
        chart_type: Type of chart to create
        data: Data for chart (DataFrame or dict)
        config: Chart configuration
        **kwargs: Additional chart-specific arguments

    Returns:
        Plotly Figure object
    """
    if config is None:
        config = ChartConfig()

    if chart_type == ChartType.LINE:
        fig = _create_line_chart(data, config, **kwargs)
    elif chart_type == ChartType.SCATTER:
        fig = _create_scatter_chart(data, config, **kwargs)
    elif chart_type == ChartType.BAR:
        fig = _create_bar_chart(data, config, **kwargs)
    elif chart_type == ChartType.CANDLESTICK:
        fig = _create_candlestick_chart(data, config, **kwargs)
    elif chart_type == ChartType.HEATMAP:
        fig = _create_heatmap(data, config, **kwargs)
    elif chart_type == ChartType.HISTOGRAM:
        fig = _create_histogram(data, config, **kwargs)
    elif chart_type == ChartType.BOX:
        fig = _create_box_plot(data, config, **kwargs)
    elif chart_type == ChartType.AREA:
        fig = _create_area_chart(data, config, **kwargs)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    # Apply common configuration
    fig.update_layout(
        title=config.title,
        width=config.width,
        height=config.height,
        template=config.template,
        showlegend=config.show_legend,
        xaxis_title=config.xaxis_title,
        yaxis_title=config.yaxis_title,
        xaxis_type=config.xaxis_type,
        yaxis_type=config.yaxis_type,
    )

    if config.show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def _create_line_chart(
    data: Union[pd.DataFrame, Dict],
    config: ChartConfig,
    x: Optional[str] = None,
    y: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> go.Figure:
    """Create line chart."""
    fig = go.Figure()

    if isinstance(data, pd.DataFrame):
        if x is None:
            x = data.index

        if y is None:
            # Plot all numeric columns
            y_cols = data.select_dtypes(include=['number']).columns.tolist()
        elif isinstance(y, str):
            y_cols = [y]
        else:
            y_cols = y

        for i, col in enumerate(y_cols):
            color = config.color_scheme[i % len(config.color_scheme)]
            fig.add_trace(go.Scatter(
                x=data[x] if x in data.columns else x,
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=color, width=2),
            ))

    elif isinstance(data, dict):
        for i, (name, values) in enumerate(data.items()):
            color = config.color_scheme[i % len(config.color_scheme)]
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
            ))

    return fig


def _create_scatter_chart(
    data: Union[pd.DataFrame, Dict],
    config: ChartConfig,
    x: str = None,
    y: str = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """Create scatter chart."""
    fig = go.Figure()

    if isinstance(data, pd.DataFrame):
        marker_dict = {}

        if size and size in data.columns:
            marker_dict['size'] = data[size]

        if color and color in data.columns:
            marker_dict['color'] = data[color]
            marker_dict['colorscale'] = 'Viridis'
            marker_dict['showscale'] = True

        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[y],
            mode='markers',
            marker=marker_dict,
        ))

    return fig


def _create_bar_chart(
    data: Union[pd.DataFrame, Dict],
    config: ChartConfig,
    x: Optional[str] = None,
    y: Optional[Union[str, List[str]]] = None,
    orientation: str = 'v',  # 'v' or 'h'
    **kwargs,
) -> go.Figure:
    """Create bar chart."""
    fig = go.Figure()

    if isinstance(data, pd.DataFrame):
        if y is None:
            y_cols = data.select_dtypes(include=['number']).columns.tolist()
        elif isinstance(y, str):
            y_cols = [y]
        else:
            y_cols = y

        for i, col in enumerate(y_cols):
            color = config.color_scheme[i % len(config.color_scheme)]

            if orientation == 'v':
                fig.add_trace(go.Bar(
                    x=data[x] if x else data.index,
                    y=data[col],
                    name=col,
                    marker_color=color,
                ))
            else:
                fig.add_trace(go.Bar(
                    y=data[x] if x else data.index,
                    x=data[col],
                    name=col,
                    orientation='h',
                    marker_color=color,
                ))

    return fig


def _create_candlestick_chart(
    data: pd.DataFrame,
    config: ChartConfig,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    **kwargs,
) -> go.Figure:
    """Create candlestick chart."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data[open_col],
        high=data[high_col],
        low=data[low_col],
        close=data[close_col],
        name="OHLC",
    ))

    return fig


def _create_heatmap(
    data: Union[pd.DataFrame, Dict],
    config: ChartConfig,
    x: Optional[List] = None,
    y: Optional[List] = None,
    z: Optional[List[List]] = None,
    colorscale: str = "RdYlGn",
    **kwargs,
) -> go.Figure:
    """Create heatmap."""
    fig = go.Figure()

    if isinstance(data, pd.DataFrame):
        # Assume DataFrame is already in pivot format
        fig.add_trace(go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=colorscale,
            text=data.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 10},
            hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>',
        ))
    else:
        fig.add_trace(go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale=colorscale,
        ))

    return fig


def _create_histogram(
    data: Union[pd.DataFrame, pd.Series, List],
    config: ChartConfig,
    column: Optional[str] = None,
    nbins: int = 30,
    **kwargs,
) -> go.Figure:
    """Create histogram."""
    fig = go.Figure()

    if isinstance(data, pd.DataFrame):
        if column:
            values = data[column]
        else:
            # Use first numeric column
            values = data.select_dtypes(include=['number']).iloc[:, 0]
    elif isinstance(data, pd.Series):
        values = data
    else:
        values = data

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=nbins,
        marker_color=config.color_scheme[0],
    ))

    return fig


def _create_box_plot(
    data: Union[pd.DataFrame, Dict],
    config: ChartConfig,
    y: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> go.Figure:
    """Create box plot."""
    fig = go.Figure()

    if isinstance(data, pd.DataFrame):
        if y is None:
            y_cols = data.select_dtypes(include=['number']).columns.tolist()
        elif isinstance(y, str):
            y_cols = [y]
        else:
            y_cols = y

        for i, col in enumerate(y_cols):
            color = config.color_scheme[i % len(config.color_scheme)]
            fig.add_trace(go.Box(
                y=data[col],
                name=col,
                marker_color=color,
            ))

    return fig


def _create_area_chart(
    data: Union[pd.DataFrame, Dict],
    config: ChartConfig,
    x: Optional[str] = None,
    y: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> go.Figure:
    """Create area chart."""
    fig = go.Figure()

    if isinstance(data, pd.DataFrame):
        if y is None:
            y_cols = data.select_dtypes(include=['number']).columns.tolist()
        elif isinstance(y, str):
            y_cols = [y]
        else:
            y_cols = y

        for i, col in enumerate(y_cols):
            color = config.color_scheme[i % len(config.color_scheme)]
            fig.add_trace(go.Scatter(
                x=data[x] if x in data.columns else data.index,
                y=data[col],
                mode='lines',
                name=col,
                fill='tonexty' if i > 0 else 'tozeroy',
                line=dict(color=color),
            ))

    return fig


def save_chart(
    fig: go.Figure,
    filename: str,
    format: str = "html",
) -> None:
    """
    Save chart to file.

    Args:
        fig: Plotly figure
        filename: Output filename
        format: Output format ('html', 'png', 'svg', 'pdf')
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "html":
        fig.write_html(str(path))
    elif format in ["png", "svg", "pdf"]:
        fig.write_image(str(path), format=format)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_subplot_grid(
    rows: int,
    cols: int,
    subplot_titles: Optional[List[str]] = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    vertical_spacing: float = 0.1,
    horizontal_spacing: float = 0.1,
) -> go.Figure:
    """
    Create subplot grid.

    Args:
        rows: Number of rows
        cols: Number of columns
        subplot_titles: Titles for subplots
        shared_xaxes: Share x-axes
        shared_yaxes: Share y-axes
        vertical_spacing: Vertical spacing between subplots
        horizontal_spacing: Horizontal spacing between subplots

    Returns:
        Figure with subplot grid
    """
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    return fig
