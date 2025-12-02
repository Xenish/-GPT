"""
Equity Curve Visualization.

Visualizes strategy equity curves, drawdowns, and performance over time.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finantradealgo.research.visualization.charts import ChartConfig, save_chart


class EquityCurveVisualizer:
    """
    Visualize strategy equity curves and performance.

    Creates interactive charts showing equity, drawdown, returns, and metrics.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize equity curve visualizer.

        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()

    def plot_equity_curve(
        self,
        trades_df: pd.DataFrame,
        starting_capital: float = 10000.0,
        show_trades: bool = True,
        show_drawdown: bool = True,
    ) -> go.Figure:
        """
        Plot equity curve from trades.

        Args:
            trades_df: DataFrame with trades (must have 'pnl' and 'exit_time' columns)
            starting_capital: Starting capital
            show_trades: Show individual trade markers
            show_drawdown: Show drawdown subplot

        Returns:
            Plotly figure
        """
        # Calculate cumulative equity
        equity_curve = self._calculate_equity_curve(trades_df, starting_capital)

        # Create subplots
        if show_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=("Equity Curve", "Drawdown"),
                shared_xaxes=True,
                vertical_spacing=0.1,
            )
        else:
            fig = go.Figure()

        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{x}<br>Equity: $%{y:,.2f}<extra></extra>',
            ),
            row=1, col=1,
        )

        # Add trade markers if requested
        if show_trades and 'exit_time' in trades_df.columns:
            # Winning trades
            wins = trades_df[trades_df['pnl'] > 0]
            if not wins.empty:
                win_equity = equity_curve.loc[wins['exit_time']]['equity']
                fig.add_trace(
                    go.Scatter(
                        x=wins['exit_time'],
                        y=win_equity,
                        mode='markers',
                        name='Wins',
                        marker=dict(color='green', size=6, symbol='triangle-up'),
                        hovertemplate='Win: $%{customdata:,.2f}<extra></extra>',
                        customdata=wins['pnl'],
                    ),
                    row=1, col=1,
                )

            # Losing trades
            losses = trades_df[trades_df['pnl'] < 0]
            if not losses.empty:
                loss_equity = equity_curve.loc[losses['exit_time']]['equity']
                fig.add_trace(
                    go.Scatter(
                        x=losses['exit_time'],
                        y=loss_equity,
                        mode='markers',
                        name='Losses',
                        marker=dict(color='red', size=6, symbol='triangle-down'),
                        hovertemplate='Loss: $%{customdata:,.2f}<extra></extra>',
                        customdata=losses['pnl'],
                    ),
                    row=1, col=1,
                )

        # Plot drawdown if requested
        if show_drawdown:
            drawdown = self._calculate_drawdown(equity_curve['equity'])

            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='#d62728', width=1),
                    hovertemplate='%{x}<br>Drawdown: %{y:.2%}<extra></extra>',
                ),
                row=2, col=1,
            )

            # Update y-axis for drawdown
            fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

        # Update layout
        fig.update_layout(
            title=self.config.title or "Equity Curve",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            hovermode='x unified',
        )

        fig.update_xaxes(title_text="Date", row=2 if show_drawdown else 1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)

        return fig

    def plot_returns_distribution(
        self,
        trades_df: pd.DataFrame,
        bins: int = 50,
    ) -> go.Figure:
        """
        Plot distribution of trade returns.

        Args:
            trades_df: DataFrame with trades
            bins: Number of histogram bins

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Returns Distribution", "Returns Box Plot"),
            column_widths=[0.7, 0.3],
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=trades_df['pnl'],
                nbinsx=bins,
                name='Returns',
                marker_color='#1f77b4',
                hovertemplate='PnL: $%{x:,.2f}<br>Count: %{y}<extra></extra>',
            ),
            row=1, col=1,
        )

        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)

        # Box plot
        fig.add_trace(
            go.Box(
                y=trades_df['pnl'],
                name='Returns',
                marker_color='#1f77b4',
                boxmean='sd',  # Show mean and std dev
            ),
            row=1, col=2,
        )

        # Update layout
        fig.update_layout(
            title="Trade Returns Analysis",
            width=self.config.width,
            height=600,
            template=self.config.template,
            showlegend=False,
        )

        fig.update_xaxes(title_text="PnL ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="PnL ($)", row=1, col=2)

        return fig

    def plot_rolling_metrics(
        self,
        trades_df: pd.DataFrame,
        window: int = 20,
        metrics: list = None,
    ) -> go.Figure:
        """
        Plot rolling performance metrics.

        Args:
            trades_df: DataFrame with trades
            window: Rolling window size
            metrics: List of metrics to plot (default: ['sharpe', 'win_rate'])

        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ['sharpe', 'win_rate']

        # Calculate rolling metrics
        rolling = self._calculate_rolling_metrics(trades_df, window)

        # Create subplots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=n_metrics, cols=1,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            shared_xaxes=True,
            vertical_spacing=0.08,
        )

        # Plot each metric
        for i, metric in enumerate(metrics, 1):
            if metric in rolling.columns:
                fig.add_trace(
                    go.Scatter(
                        x=rolling.index,
                        y=rolling[metric],
                        mode='lines',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=2),
                    ),
                    row=i, col=1,
                )

                # Add reference lines
                if metric == 'win_rate':
                    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=i, col=1)
                elif metric == 'sharpe':
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i, col=1)

        # Update layout
        fig.update_layout(
            title=f"Rolling Metrics (Window: {window} trades)",
            width=self.config.width,
            height=300 * n_metrics,
            template=self.config.template,
            hovermode='x unified',
            showlegend=False,
        )

        fig.update_xaxes(title_text="Trade Index", row=n_metrics, col=1)

        return fig

    def plot_monthly_returns(
        self,
        trades_df: pd.DataFrame,
    ) -> go.Figure:
        """
        Plot monthly returns heatmap.

        Args:
            trades_df: DataFrame with trades

        Returns:
            Plotly figure
        """
        # Calculate monthly returns
        monthly = self._calculate_monthly_returns(trades_df)

        if monthly.empty:
            # No data
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for monthly returns",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
            )
            return fig

        # Create heatmap
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=monthly.values,
            x=monthly.columns,
            y=monthly.index,
            colorscale='RdYlGn',
            zmid=0,
            text=monthly.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            hovertemplate='%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>',
        ))

        # Update layout
        fig.update_layout(
            title="Monthly Returns Heatmap",
            width=self.config.width,
            height=400,
            template=self.config.template,
            xaxis_title="Month",
            yaxis_title="Year",
        )

        return fig

    def _calculate_equity_curve(
        self,
        trades_df: pd.DataFrame,
        starting_capital: float,
    ) -> pd.DataFrame:
        """Calculate equity curve from trades."""
        if 'exit_time' not in trades_df.columns:
            # No timestamps, use index
            equity = pd.DataFrame({
                'equity': starting_capital + trades_df['pnl'].cumsum()
            })
        else:
            # Sort by exit time
            trades_sorted = trades_df.sort_values('exit_time')

            # Calculate cumulative equity
            equity = pd.DataFrame({
                'equity': starting_capital + trades_sorted['pnl'].cumsum()
            }, index=pd.to_datetime(trades_sorted['exit_time']))

        return equity

    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown from equity curve."""
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        return drawdown

    def _calculate_rolling_metrics(
        self,
        trades_df: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        rolling = pd.DataFrame(index=trades_df.index)

        # Rolling Sharpe
        rolling['sharpe'] = trades_df['pnl'].rolling(window).apply(
            lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )

        # Rolling win rate
        rolling['win_rate'] = trades_df['pnl'].rolling(window).apply(
            lambda x: (x > 0).sum() / len(x)
        )

        # Rolling avg PnL
        rolling['avg_pnl'] = trades_df['pnl'].rolling(window).mean()

        return rolling

    def _calculate_monthly_returns(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns."""
        if 'exit_time' not in trades_df.columns:
            return pd.DataFrame()

        # Convert to datetime
        trades_df = trades_df.copy()
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

        # Group by year-month
        trades_df['year'] = trades_df['exit_time'].dt.year
        trades_df['month'] = trades_df['exit_time'].dt.month

        # Calculate monthly PnL
        monthly_pnl = trades_df.groupby(['year', 'month'])['pnl'].sum()

        # Convert to percentage (assuming starting capital)
        starting_capital = 10000.0
        monthly_returns = (monthly_pnl / starting_capital) * 100

        # Pivot to year x month format
        monthly_pivot = monthly_returns.unstack(fill_value=0)

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly_pivot.columns = [month_names[m-1] for m in monthly_pivot.columns]

        return monthly_pivot

    def save(self, fig: go.Figure, filename: str, format: str = "html") -> None:
        """
        Save figure to file.

        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format
        """
        save_chart(fig, filename, format)
