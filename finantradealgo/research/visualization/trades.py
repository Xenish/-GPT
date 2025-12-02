"""
Trade Analysis Visualization.

Detailed visualizations of individual trades and trade patterns.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finantradealgo.research.visualization.charts import ChartConfig, save_chart


class TradeAnalysisVisualizer:
    """
    Visualize trade analysis and patterns.

    Creates charts for trade distribution, duration, PnL patterns, etc.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize trade analysis visualizer.

        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()

    def plot_trade_overview(
        self,
        trades_df: pd.DataFrame,
    ) -> go.Figure:
        """
        Plot comprehensive trade overview.

        Args:
            trades_df: DataFrame with trades

        Returns:
            Plotly figure with multiple subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "PnL Distribution",
                "Win/Loss Count",
                "Trade Duration",
                "Cumulative PnL"
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "scatter"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
        )

        # 1. PnL Distribution
        fig.add_trace(
            go.Histogram(
                x=trades_df['pnl'],
                nbinsx=30,
                name='PnL',
                marker_color='steelblue',
            ),
            row=1, col=1,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)

        # 2. Win/Loss Count
        wins = (trades_df['pnl'] > 0).sum()
        losses = (trades_df['pnl'] <= 0).sum()

        fig.add_trace(
            go.Bar(
                x=['Wins', 'Losses'],
                y=[wins, losses],
                marker_color=['green', 'red'],
                text=[wins, losses],
                textposition='outside',
            ),
            row=1, col=2,
        )

        # 3. Trade Duration (if available)
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            duration = (pd.to_datetime(trades_df['exit_time']) -
                       pd.to_datetime(trades_df['entry_time'])).dt.total_seconds() / 3600  # hours

            fig.add_trace(
                go.Histogram(
                    x=duration,
                    nbinsx=20,
                    name='Duration',
                    marker_color='orange',
                ),
                row=2, col=1,
            )

        # 4. Cumulative PnL
        fig.add_trace(
            go.Scatter(
                y=trades_df['pnl'].cumsum(),
                mode='lines',
                name='Cumulative PnL',
                line=dict(color='blue', width=2),
            ),
            row=2, col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

        # Update layout
        fig.update_layout(
            title="Trade Analysis Overview",
            width=self.config.width,
            height=800,
            template=self.config.template,
            showlegend=False,
        )

        fig.update_xaxes(title_text="PnL ($)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (hours)", row=2, col=1)
        fig.update_xaxes(title_text="Trade #", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative PnL ($)", row=2, col=2)

        return fig

    def plot_pnl_by_time(
        self,
        trades_df: pd.DataFrame,
        group_by: str = "hour",  # hour, day, month
    ) -> go.Figure:
        """
        Plot PnL grouped by time.

        Args:
            trades_df: DataFrame with trades
            group_by: Grouping ('hour', 'day', 'month')

        Returns:
            Plotly figure
        """
        if 'exit_time' not in trades_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No timestamp data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
            )
            return fig

        trades_df = trades_df.copy()
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

        # Group by time
        if group_by == "hour":
            trades_df['time_group'] = trades_df['exit_time'].dt.hour
            x_title = "Hour of Day"
        elif group_by == "day":
            trades_df['time_group'] = trades_df['exit_time'].dt.dayofweek
            x_title = "Day of Week"
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        elif group_by == "month":
            trades_df['time_group'] = trades_df['exit_time'].dt.month
            x_title = "Month"
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

        # Aggregate
        grouped = trades_df.groupby('time_group')['pnl'].agg(['sum', 'mean', 'count'])

        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Total PnL", "Average PnL"),
        )

        # Total PnL
        x_values = grouped.index
        if group_by == "day":
            x_values = [day_names[i] for i in grouped.index]

        fig.add_trace(
            go.Bar(
                x=x_values,
                y=grouped['sum'],
                marker_color=['green' if x > 0 else 'red' for x in grouped['sum']],
                text=grouped['sum'],
                texttemplate='%{text:.2f}',
                hovertemplate=f'{x_title}: %{{x}}<br>Total PnL: $%{{y:.2f}}<br>Trades: %{{customdata}}<extra></extra>',
                customdata=grouped['count'],
            ),
            row=1, col=1,
        )

        # Average PnL
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=grouped['mean'],
                marker_color=['green' if x > 0 else 'red' for x in grouped['mean']],
                text=grouped['mean'],
                texttemplate='%{text:.2f}',
                hovertemplate=f'{x_title}: %{{x}}<br>Avg PnL: $%{{y:.2f}}<extra></extra>',
            ),
            row=1, col=2,
        )

        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

        # Update layout
        fig.update_layout(
            title=f"PnL by {x_title}",
            width=self.config.width,
            height=500,
            template=self.config.template,
            showlegend=False,
        )

        fig.update_xaxes(title_text=x_title, row=1, col=1)
        fig.update_xaxes(title_text=x_title, row=1, col=2)

        return fig

    def plot_win_loss_analysis(
        self,
        trades_df: pd.DataFrame,
    ) -> go.Figure:
        """
        Plot detailed win/loss analysis.

        Args:
            trades_df: DataFrame with trades

        Returns:
            Plotly figure
        """
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Win Distribution", "Loss Distribution"),
        )

        # Win distribution
        if not wins.empty:
            fig.add_trace(
                go.Histogram(
                    x=wins,
                    nbinsx=20,
                    name='Wins',
                    marker_color='green',
                    hovertemplate='PnL: $%{x:.2f}<br>Count: %{y}<extra></extra>',
                ),
                row=1, col=1,
            )

        # Loss distribution
        if not losses.empty:
            fig.add_trace(
                go.Histogram(
                    x=losses,
                    nbinsx=20,
                    name='Losses',
                    marker_color='red',
                    hovertemplate='PnL: $%{x:.2f}<br>Count: %{y}<extra></extra>',
                ),
                row=1, col=2,
            )

        # Update layout
        fig.update_layout(
            title="Win/Loss Distribution Analysis",
            width=self.config.width,
            height=500,
            template=self.config.template,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Win Amount ($)", row=1, col=1)
        fig.update_xaxes(title_text="Loss Amount ($)", row=1, col=2)

        return fig

    def plot_consecutive_analysis(
        self,
        trades_df: pd.DataFrame,
    ) -> go.Figure:
        """
        Plot consecutive wins/losses analysis.

        Args:
            trades_df: DataFrame with trades

        Returns:
            Plotly figure
        """
        # Calculate consecutive streaks
        streaks = []
        current_streak = 0
        streak_type = None

        for pnl in trades_df['pnl']:
            if pnl > 0:
                if streak_type == 'win':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append({'type': streak_type, 'length': current_streak})
                    current_streak = 1
                    streak_type = 'win'
            else:
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append({'type': streak_type, 'length': current_streak})
                    current_streak = 1
                    streak_type = 'loss'

        # Add last streak
        if current_streak > 0:
            streaks.append({'type': streak_type, 'length': current_streak})

        if not streaks:
            fig = go.Figure()
            fig.add_annotation(
                text="No streak data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
            )
            return fig

        streaks_df = pd.DataFrame(streaks)

        # Create figure
        fig = go.Figure()

        # Win streaks
        win_streaks = streaks_df[streaks_df['type'] == 'win']['length']
        if not win_streaks.empty:
            fig.add_trace(go.Histogram(
                x=win_streaks,
                name='Win Streaks',
                marker_color='green',
                opacity=0.7,
                nbinsx=max(win_streaks.max(), 10),
            ))

        # Loss streaks
        loss_streaks = streaks_df[streaks_df['type'] == 'loss']['length']
        if not loss_streaks.empty:
            fig.add_trace(go.Histogram(
                x=loss_streaks,
                name='Loss Streaks',
                marker_color='red',
                opacity=0.7,
                nbinsx=max(loss_streaks.max(), 10),
            ))

        # Update layout
        fig.update_layout(
            title="Consecutive Wins/Losses Analysis",
            width=self.config.width,
            height=500,
            template=self.config.template,
            xaxis_title="Streak Length",
            yaxis_title="Frequency",
            barmode='overlay',
        )

        return fig

    def plot_trade_sizes(
        self,
        trades_df: pd.DataFrame,
    ) -> go.Figure:
        """
        Plot trade size analysis.

        Args:
            trades_df: DataFrame with trades

        Returns:
            Plotly figure
        """
        # Separate wins and losses
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]

        fig = go.Figure()

        # Scatter plot: Trade index vs PnL size
        fig.add_trace(go.Scatter(
            x=wins.index,
            y=wins['pnl'].abs(),
            mode='markers',
            name='Wins',
            marker=dict(color='green', size=8, opacity=0.6),
            hovertemplate='Trade #%{x}<br>Win: $%{y:.2f}<extra></extra>',
        ))

        fig.add_trace(go.Scatter(
            x=losses.index,
            y=losses['pnl'].abs(),
            mode='markers',
            name='Losses',
            marker=dict(color='red', size=8, opacity=0.6),
            hovertemplate='Trade #%{x}<br>Loss: $%{y:.2f}<extra></extra>',
        ))

        # Update layout
        fig.update_layout(
            title="Trade Size Analysis",
            width=self.config.width,
            height=500,
            template=self.config.template,
            xaxis_title="Trade Number",
            yaxis_title="Absolute PnL ($)",
            yaxis_type="log",  # Log scale for better visualization
        )

        return fig

    def save(self, fig: go.Figure, filename: str, format: str = "html") -> None:
        """Save figure to file."""
        save_chart(fig, filename, format)
