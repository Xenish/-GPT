"""
Walk-Forward Visualization.

Charts and plots for walk-forward analysis results.
"""

from __future__ import annotations

from typing import Optional, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finantradealgo.research.walkforward.models import WalkForwardResult
from finantradealgo.research.visualization.charts import ChartConfig, save_chart


class WalkForwardVisualizer:
    """
    Visualize walk-forward optimization results.

    Creates charts showing IS/OOS performance, degradation, and robustness.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize visualizer.

        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()

    def plot_performance_comparison(self, result: WalkForwardResult) -> go.Figure:
        """
        Plot IS vs OOS performance across windows.

        Args:
            result: Walk-forward result

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Sharpe Ratio: IS vs OOS",
                "Total Return: IS vs OOS",
                "Degradation Over Time",
                "Win Rate: IS vs OOS",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        window_ids = [w.window_id for w in result.windows]

        # 1. Sharpe Ratio comparison
        fig.add_trace(
            go.Scatter(
                x=window_ids,
                y=[w.is_sharpe for w in result.windows],
                mode='lines+markers',
                name='IS Sharpe',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=window_ids,
                y=[w.oos_sharpe for w in result.windows],
                mode='lines+markers',
                name='OOS Sharpe',
                line=dict(color='red', width=2),
                marker=dict(size=8),
            ),
            row=1, col=1,
        )

        # 2. Total Return comparison
        fig.add_trace(
            go.Scatter(
                x=window_ids,
                y=[w.is_total_return for w in result.windows],
                mode='lines+markers',
                name='IS Return',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=window_ids,
                y=[w.oos_total_return for w in result.windows],
                mode='lines+markers',
                name='OOS Return',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=1, col=2,
        )

        # 3. Degradation over time
        fig.add_trace(
            go.Bar(
                x=window_ids,
                y=[w.sharpe_degradation * 100 for w in result.windows],
                name='Sharpe Degradation',
                marker_color=['red' if d > 0 else 'green' for d in [w.sharpe_degradation for w in result.windows]],
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # 4. Win Rate comparison
        fig.add_trace(
            go.Scatter(
                x=window_ids,
                y=[w.is_win_rate * 100 for w in result.windows],
                mode='lines+markers',
                name='IS Win Rate',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=window_ids,
                y=[w.oos_win_rate * 100 for w in result.windows],
                mode='lines+markers',
                name='OOS Win Rate',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=2)

        # Update layout
        fig.update_layout(
            title=f"Walk-Forward Performance Analysis - {result.strategy_id}",
            width=1400,
            height=900,
            template=self.config.template,
            hovermode='x unified',
        )

        # Update axes
        fig.update_xaxes(title_text="Window", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_xaxes(title_text="Window", row=1, col=2)
        fig.update_yaxes(title_text="Total Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Window", row=2, col=1)
        fig.update_yaxes(title_text="Degradation (%)", row=2, col=1)
        fig.update_xaxes(title_text="Window", row=2, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)

        return fig

    def plot_equity_curve(self, result: WalkForwardResult) -> go.Figure:
        """
        Plot combined OOS equity curve with window markers.

        Args:
            result: Walk-forward result

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        if result.combined_equity_curve is not None:
            # Plot equity curve
            fig.add_trace(
                go.Scatter(
                    y=result.combined_equity_curve,
                    mode='lines',
                    name='OOS Equity',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.2)',
                )
            )

            # Add window markers
            window_boundaries = []
            for i, window in enumerate(result.windows):
                # Add vertical line at window boundary
                window_boundaries.append(i)

            # Mark window transitions
            for boundary in window_boundaries[1:]:  # Skip first
                fig.add_vline(
                    x=boundary,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                )

        fig.update_layout(
            title=f"Combined Out-of-Sample Equity Curve - {result.strategy_id}",
            width=self.config.width,
            height=600,
            template=self.config.template,
            xaxis_title="Trade",
            yaxis_title="Cumulative PnL",
        )

        return fig

    def plot_parameter_stability(self, result: WalkForwardResult) -> go.Figure:
        """
        Plot parameter evolution across windows.

        Args:
            result: Walk-forward result

        Returns:
            Plotly figure
        """
        # Collect parameters
        param_names = set()
        for window in result.windows:
            param_names.update(window.best_params.keys())

        # Create subplots for each parameter
        n_params = len(param_names)
        if n_params == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No parameter data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
            )
            return fig

        rows = (n_params + 1) // 2
        cols = min(2, n_params)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(param_names),
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
        )

        window_ids = [w.window_id for w in result.windows]

        for i, param_name in enumerate(param_names, 1):
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1

            # Extract parameter values
            param_values = []
            for window in result.windows:
                value = window.best_params.get(param_name, None)
                if isinstance(value, (int, float)):
                    param_values.append(value)
                else:
                    param_values.append(None)

            fig.add_trace(
                go.Scatter(
                    x=window_ids,
                    y=param_values,
                    mode='lines+markers',
                    name=param_name,
                    line=dict(width=2),
                    marker=dict(size=10),
                    showlegend=False,
                ),
                row=row, col=col,
            )

        fig.update_layout(
            title=f"Parameter Stability Across Windows - {result.strategy_id}",
            width=1200,
            height=300 * rows,
            template=self.config.template,
        )

        return fig

    def plot_degradation_distribution(self, result: WalkForwardResult) -> go.Figure:
        """
        Plot distribution of IS-to-OOS degradation.

        Args:
            result: Walk-forward result

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Sharpe Degradation", "Return Degradation"),
        )

        # Sharpe degradation histogram
        sharpe_degradations = [w.sharpe_degradation * 100 for w in result.windows]
        fig.add_trace(
            go.Histogram(
                x=sharpe_degradations,
                nbinsx=15,
                name='Sharpe Degradation',
                marker_color='steelblue',
            ),
            row=1, col=1,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)

        # Return degradation histogram
        return_degradations = [w.return_degradation * 100 for w in result.windows]
        fig.add_trace(
            go.Histogram(
                x=return_degradations,
                nbinsx=15,
                name='Return Degradation',
                marker_color='orange',
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_layout(
            title=f"Degradation Distribution - {result.strategy_id}",
            width=1200,
            height=500,
            template=self.config.template,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Degradation (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Degradation (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        return fig

    def plot_robustness_dashboard(self, result: WalkForwardResult) -> go.Figure:
        """
        Create comprehensive robustness dashboard.

        Args:
            result: Walk-forward result

        Returns:
            Plotly figure with multiple panels
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "IS vs OOS Sharpe",
                "Consistency Score by Window",
                "Combined OOS Equity",
                "Parameter Stability Score",
                "Window Returns Distribution",
                "Efficiency Metrics",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "indicator"}],
                [{"type": "histogram"}, {"type": "table"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
        )

        window_ids = [w.window_id for w in result.windows]

        # 1. IS vs OOS Sharpe
        fig.add_trace(
            go.Scatter(
                x=window_ids,
                y=[w.is_sharpe for w in result.windows],
                mode='lines+markers',
                name='IS',
                line=dict(color='blue'),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=window_ids,
                y=[w.oos_sharpe for w in result.windows],
                mode='lines+markers',
                name='OOS',
                line=dict(color='red'),
            ),
            row=1, col=1,
        )

        # 2. Consistency by window (profitable OOS periods)
        oos_profitable = [1 if w.oos_total_return > 0 else 0 for w in result.windows]
        fig.add_trace(
            go.Bar(
                x=window_ids,
                y=oos_profitable,
                marker_color=['green' if x == 1 else 'red' for x in oos_profitable],
                showlegend=False,
            ),
            row=1, col=2,
        )

        # 3. Combined OOS Equity
        if result.combined_equity_curve is not None:
            fig.add_trace(
                go.Scatter(
                    y=result.combined_equity_curve,
                    mode='lines',
                    line=dict(color='#1f77b4', width=2),
                    showlegend=False,
                ),
                row=2, col=1,
            )

        # 4. Parameter Stability Indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=result.param_stability_score,
                title={'text': "Stability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"},
                    ],
                },
            ),
            row=2, col=2,
        )

        # 5. OOS Returns Distribution
        oos_returns = [w.oos_total_return for w in result.windows]
        fig.add_trace(
            go.Histogram(
                x=oos_returns,
                nbinsx=15,
                marker_color='steelblue',
                showlegend=False,
            ),
            row=3, col=1,
        )

        # 6. Summary Table
        summary_data = [
            ["Avg OOS Sharpe", f"{result.avg_oos_sharpe:.2f}"],
            ["Avg OOS Return", f"{result.avg_oos_return:.2f}%"],
            ["OOS Win Rate", f"{result.oos_win_rate:.1%}"],
            ["Avg Degradation", f"{result.avg_sharpe_degradation:.1%}"],
            ["Consistency Score", f"{result.consistency_score:.1f}/100"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color='#3498db',
                    font=dict(color='white'),
                ),
                cells=dict(
                    values=list(zip(*summary_data)),
                    fill_color='white',
                ),
            ),
            row=3, col=2,
        )

        fig.update_layout(
            title=f"Walk-Forward Robustness Dashboard - {result.strategy_id}",
            width=1400,
            height=1200,
            template=self.config.template,
            showlegend=True,
        )

        return fig

    def plot_comparison(self, results: List[WalkForwardResult]) -> go.Figure:
        """
        Compare multiple walk-forward results.

        Args:
            results: List of walk-forward results

        Returns:
            Plotly figure with comparison
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Avg OOS Sharpe Comparison",
                "Consistency Score Comparison",
                "Avg Degradation Comparison",
                "Parameter Stability Comparison",
            ),
        )

        strategy_names = [r.strategy_id for r in results]

        # 1. OOS Sharpe comparison
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=[r.avg_oos_sharpe for r in results],
                marker_color='#2ecc71',
                showlegend=False,
            ),
            row=1, col=1,
        )

        # 2. Consistency score
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=[r.consistency_score for r in results],
                marker_color='#3498db',
                showlegend=False,
            ),
            row=1, col=2,
        )

        # 3. Degradation
        degradations = [abs(r.avg_sharpe_degradation) * 100 for r in results]
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=degradations,
                marker_color='#e74c3c',
                showlegend=False,
            ),
            row=2, col=1,
        )

        # 4. Parameter stability
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=[r.param_stability_score for r in results],
                marker_color='#9b59b6',
                showlegend=False,
            ),
            row=2, col=2,
        )

        fig.update_layout(
            title="Walk-Forward Strategy Comparison",
            width=1400,
            height=900,
            template=self.config.template,
        )

        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Score (0-100)", row=1, col=2)
        fig.update_yaxes(title_text="Degradation (%)", row=2, col=1)
        fig.update_yaxes(title_text="Score (0-100)", row=2, col=2)

        return fig

    def save(self, fig: go.Figure, filename: str, format: str = "html") -> None:
        """Save figure to file."""
        save_chart(fig, filename, format)
