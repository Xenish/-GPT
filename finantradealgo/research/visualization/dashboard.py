"""
Performance Dashboard Components.

Combines multiple visualizations into comprehensive interactive dashboards.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finantradealgo.research.visualization.charts import ChartConfig, save_chart
from finantradealgo.research.visualization.equity import EquityCurveVisualizer
from finantradealgo.research.visualization.trades import TradeAnalysisVisualizer
from finantradealgo.research.visualization.heatmap import ParameterHeatmapVisualizer
from finantradealgo.research.performance.models import PerformanceMetrics


class StrategyDashboard:
    """
    Strategy performance dashboard.

    Combines equity curves, trade analysis, and metrics into a single view.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize strategy dashboard.

        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()
        self.equity_viz = EquityCurveVisualizer(config)
        self.trade_viz = TradeAnalysisVisualizer(config)

    def create_full_dashboard(
        self,
        trades_df: pd.DataFrame,
        metrics: Optional[PerformanceMetrics] = None,
        starting_capital: float = 10000.0,
    ) -> go.Figure:
        """
        Create comprehensive strategy dashboard.

        Args:
            trades_df: DataFrame with trades
            metrics: Performance metrics (optional)
            starting_capital: Starting capital

        Returns:
            Plotly figure with full dashboard
        """
        # Create 3x2 subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Equity Curve",
                "PnL Distribution",
                "Drawdown",
                "Win/Loss Analysis",
                "Cumulative PnL",
                "Trade Duration",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "histogram"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
        )

        # Calculate equity curve
        equity_curve = self.equity_viz._calculate_equity_curve(
            trades_df, starting_capital
        )

        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Equity: $%{y:,.2f}<extra></extra>',
            ),
            row=1, col=1,
        )

        # 2. PnL Distribution
        fig.add_trace(
            go.Histogram(
                x=trades_df['pnl'],
                nbinsx=30,
                name='PnL',
                marker_color='steelblue',
                hovertemplate='PnL: $%{x:.2f}<br>Count: %{y}<extra></extra>',
            ),
            row=1, col=2,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)

        # 3. Drawdown
        drawdown = self.equity_viz._calculate_drawdown(equity_curve['equity'])
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#d62728', width=1),
                hovertemplate='Drawdown: %{y:.2f}%<extra></extra>',
            ),
            row=2, col=1,
        )

        # 4. Win/Loss Analysis
        wins = (trades_df['pnl'] > 0).sum()
        losses = (trades_df['pnl'] <= 0).sum()
        fig.add_trace(
            go.Bar(
                x=['Wins', 'Losses'],
                y=[wins, losses],
                marker_color=['green', 'red'],
                text=[wins, losses],
                textposition='outside',
                hovertemplate='%{x}: %{y}<extra></extra>',
            ),
            row=2, col=2,
        )

        # 5. Cumulative PnL
        fig.add_trace(
            go.Scatter(
                y=trades_df['pnl'].cumsum(),
                mode='lines',
                name='Cumulative PnL',
                line=dict(color='blue', width=2),
                hovertemplate='Cumulative: $%{y:,.2f}<extra></extra>',
            ),
            row=3, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        # 6. Trade Duration (if available)
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            duration = (
                pd.to_datetime(trades_df['exit_time']) -
                pd.to_datetime(trades_df['entry_time'])
            ).dt.total_seconds() / 3600  # hours

            fig.add_trace(
                go.Histogram(
                    x=duration,
                    nbinsx=20,
                    name='Duration',
                    marker_color='orange',
                    hovertemplate='Duration: %{x:.1f}h<br>Count: %{y}<extra></extra>',
                ),
                row=3, col=2,
            )

        # Update layout
        title_text = "Strategy Performance Dashboard"
        if metrics:
            title_text += (
                f"<br><sub>Sharpe: {metrics.sharpe_ratio:.2f} | "
                f"Win Rate: {metrics.win_rate:.1%} | "
                f"Max DD: {metrics.max_drawdown:.2f}%</sub>"
            )

        fig.update_layout(
            title=title_text,
            width=1400,
            height=1000,
            template=self.config.template,
            showlegend=False,
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_xaxes(title_text="PnL ($)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_xaxes(title_text="Trade #", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative PnL ($)", row=3, col=1)
        fig.update_xaxes(title_text="Duration (hours)", row=3, col=2)
        fig.update_yaxes(title_text="Frequency", row=3, col=2)

        return fig

    def create_quick_summary(
        self,
        trades_df: pd.DataFrame,
        metrics: PerformanceMetrics,
    ) -> go.Figure:
        """
        Create quick summary dashboard with key metrics.

        Args:
            trades_df: DataFrame with trades
            metrics: Performance metrics

        Returns:
            Plotly figure with summary
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Equity Curve",
                "Key Metrics",
                "Monthly Returns",
                "Win Rate by Size",
            ),
            specs=[
                [{"type": "scatter", "colspan": 2}, None],
                [{"type": "table"}, {"type": "bar"}],
            ],
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5],
        )

        # 1. Equity Curve
        equity_curve = self.equity_viz._calculate_equity_curve(trades_df, 10000.0)
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['equity'],
                mode='lines',
                line=dict(color='#1f77b4', width=3),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                hovertemplate='Equity: $%{y:,.2f}<extra></extra>',
            ),
            row=1, col=1,
        )

        # 2. Key Metrics Table
        metrics_data = [
            ["Total PnL", f"${metrics.total_pnl:,.2f}"],
            ["Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}"],
            ["Max Drawdown", f"{metrics.max_drawdown:.2f}%"],
            ["Win Rate", f"{metrics.win_rate:.1%}"],
            ["Total Trades", f"{metrics.total_trades}"],
            ["Avg Win", f"${metrics.avg_win_amount:,.2f}"],
            ["Avg Loss", f"${metrics.avg_loss_amount:,.2f}"],
            ["Profit Factor", f"{metrics.profit_factor:.2f}"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color='#3498db',
                    font=dict(color='white', size=12),
                    align='left',
                ),
                cells=dict(
                    values=list(zip(*metrics_data)),
                    fill_color=[['#ecf0f1', '#ffffff'] * 4],
                    align='left',
                    font=dict(size=11),
                ),
            ),
            row=2, col=1,
        )

        # 3. Win Rate by Trade Size
        if not trades_df.empty:
            # Bin trades by size
            trades_df = trades_df.copy()
            trades_df['size_bin'] = pd.cut(
                trades_df['pnl'].abs(),
                bins=[0, 50, 100, 200, float('inf')],
                labels=['Small', 'Medium', 'Large', 'XLarge'],
            )

            win_rate_by_size = trades_df.groupby('size_bin')['pnl'].apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
            )

            fig.add_trace(
                go.Bar(
                    x=win_rate_by_size.index.astype(str),
                    y=win_rate_by_size.values * 100,
                    marker_color='#2ecc71',
                    text=[f"{v:.1f}%" for v in win_rate_by_size.values * 100],
                    textposition='outside',
                    hovertemplate='%{x}<br>Win Rate: %{y:.1f}%<extra></extra>',
                ),
                row=2, col=2,
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=2)

        # Update layout
        fig.update_layout(
            title="Strategy Quick Summary",
            width=1200,
            height=800,
            template=self.config.template,
            showlegend=False,
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)

        return fig

    def save(self, fig: go.Figure, filename: str, format: str = "html") -> None:
        """Save dashboard to file."""
        save_chart(fig, filename, format)


class ParameterSearchDashboard:
    """
    Parameter search results dashboard.

    Visualizes parameter optimization results and parameter space.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize parameter search dashboard.

        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()
        self.heatmap_viz = ParameterHeatmapVisualizer(config)

    def create_search_dashboard(
        self,
        results_df: pd.DataFrame,
        metric: str = "sharpe_ratio",
        param1: Optional[str] = None,
        param2: Optional[str] = None,
    ) -> go.Figure:
        """
        Create parameter search dashboard.

        Args:
            results_df: DataFrame with search results
            metric: Target metric
            param1: First parameter for heatmap (auto-detect if None)
            param2: Second parameter for heatmap (auto-detect if None)

        Returns:
            Plotly figure with search dashboard
        """
        # Auto-detect parameters
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        if param1 is None and len(param_cols) > 0:
            param1 = param_cols[0].replace('param_', '')
        if param2 is None and len(param_cols) > 1:
            param2 = param_cols[1].replace('param_', '')

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Parameter Space: {param1} vs {param2}",
                f"Top 20 Combinations",
                f"{metric.replace('_', ' ').title()} Distribution",
                "Parameter Correlation",
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "heatmap"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
        )

        # 1. Parameter Heatmap (if 2D)
        if param1 and param2:
            pivot = results_df.pivot_table(
                values=metric,
                index=f'param_{param2}',
                columns=f'param_{param1}',
                aggfunc='mean',
            )

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='RdYlGn',
                    hovertemplate=f'{param1}: %{{x}}<br>{param2}: %{{y}}<br>{metric}: %{{z:.3f}}<extra></extra>',
                ),
                row=1, col=1,
            )

        # 2. Top 20 Combinations
        top_20 = results_df.nlargest(20, metric)
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(top_20) + 1)),
                y=top_20[metric],
                marker_color='#2ecc71',
                text=top_20[metric].round(3),
                textposition='outside',
                hovertemplate='Rank %{x}<br>Score: %{y:.3f}<extra></extra>',
            ),
            row=1, col=2,
        )

        # 3. Metric Distribution
        fig.add_trace(
            go.Histogram(
                x=results_df[metric],
                nbinsx=30,
                marker_color='steelblue',
                hovertemplate=f'{metric}: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>',
            ),
            row=2, col=1,
        )

        # 4. Correlation Matrix
        corr_cols = param_cols + [metric]
        corr_matrix = results_df[corr_cols].corr()

        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=[col.replace('param_', '') for col in corr_matrix.columns],
                y=[col.replace('param_', '') for col in corr_matrix.index],
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
            ),
            row=2, col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"Parameter Search Results - {metric.replace('_', ' ').title()}",
            width=1400,
            height=1000,
            template=self.config.template,
            showlegend=False,
        )

        # Update axes
        if param1 and param2:
            fig.update_xaxes(title_text=param1, row=1, col=1)
            fig.update_yaxes(title_text=param2, row=1, col=1)
        fig.update_xaxes(title_text="Rank", row=1, col=2)
        fig.update_yaxes(title_text=metric, row=1, col=2)
        fig.update_xaxes(title_text=metric, row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        return fig

    def save(self, fig: go.Figure, filename: str, format: str = "html") -> None:
        """Save dashboard to file."""
        save_chart(fig, filename, format)


class ComparisonDashboard:
    """
    Strategy comparison dashboard.

    Compares multiple strategies side by side.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize comparison dashboard.

        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()

    def create_comparison_dashboard(
        self,
        strategies_data: Dict[str, pd.DataFrame],
        starting_capital: float = 10000.0,
    ) -> go.Figure:
        """
        Create strategy comparison dashboard.

        Args:
            strategies_data: Dict mapping strategy names to trades DataFrames
            starting_capital: Starting capital

        Returns:
            Plotly figure with comparison
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Equity Curves Comparison",
                "Sharpe Ratio Comparison",
                "Drawdown Comparison",
                "Win Rate & Profit Factor",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        colors = self.config.color_scheme
        metrics_summary = {}

        # Process each strategy
        for i, (name, trades_df) in enumerate(strategies_data.items()):
            color = colors[i % len(colors)]

            # Calculate equity curve
            equity = starting_capital + trades_df['pnl'].cumsum()

            # 1. Equity Curves
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(equity))),
                    y=equity,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    hovertemplate=f'{name}<br>Equity: $%{{y:,.2f}}<extra></extra>',
                ),
                row=1, col=1,
            )

            # 3. Drawdown
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(drawdown))),
                    y=drawdown,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    hovertemplate=f'{name}<br>DD: %{{y:.2f}}%<extra></extra>',
                    showlegend=False,
                ),
                row=2, col=1,
            )

            # Calculate metrics for summary
            returns = trades_df['pnl']
            sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
            wins = returns[returns > 0].sum() if (returns > 0).any() else 0
            losses = abs(returns[returns <= 0].sum()) if (returns <= 0).any() else 1
            profit_factor = wins / losses if losses > 0 else 0

            metrics_summary[name] = {
                'sharpe': sharpe,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
            }

        # 2. Sharpe Comparison
        sharpe_values = [m['sharpe'] for m in metrics_summary.values()]
        fig.add_trace(
            go.Bar(
                x=list(metrics_summary.keys()),
                y=sharpe_values,
                marker_color=colors[:len(metrics_summary)],
                text=[f"{v:.2f}" for v in sharpe_values],
                textposition='outside',
                hovertemplate='%{x}<br>Sharpe: %{y:.2f}<extra></extra>',
                showlegend=False,
            ),
            row=1, col=2,
        )

        # 4. Win Rate & Profit Factor
        win_rates = [m['win_rate'] * 100 for m in metrics_summary.values()]
        profit_factors = [m['profit_factor'] for m in metrics_summary.values()]

        fig.add_trace(
            go.Bar(
                x=list(metrics_summary.keys()),
                y=win_rates,
                name='Win Rate (%)',
                marker_color='#2ecc71',
                text=[f"{v:.1f}%" for v in win_rates],
                textposition='outside',
                hovertemplate='%{x}<br>Win Rate: %{y:.1f}%<extra></extra>',
            ),
            row=2, col=2,
        )

        fig.add_trace(
            go.Bar(
                x=list(metrics_summary.keys()),
                y=profit_factors,
                name='Profit Factor',
                marker_color='#3498db',
                text=[f"{v:.2f}" for v in profit_factors],
                textposition='outside',
                yaxis='y4',
                hovertemplate='%{x}<br>PF: %{y:.2f}<extra></extra>',
            ),
            row=2, col=2,
        )

        # Update layout
        fig.update_layout(
            title="Strategy Comparison Dashboard",
            width=1400,
            height=900,
            template=self.config.template,
            hovermode='x unified',
        )

        # Update axes
        fig.update_xaxes(title_text="Trade #", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Trade #", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        return fig

    def save(self, fig: go.Figure, filename: str, format: str = "html") -> None:
        """Save dashboard to file."""
        save_chart(fig, filename, format)
