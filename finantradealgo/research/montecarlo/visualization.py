"""
Monte Carlo Visualization.

Charts for Monte Carlo simulation results.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finantradealgo.research.montecarlo.models import MonteCarloResult
from finantradealgo.research.visualization.charts import ChartConfig, save_chart


class MonteCarloVisualizer:
    """Visualize Monte Carlo simulation results."""

    def __init__(self, config: Optional[ChartConfig] = None):
        """Initialize visualizer."""
        self.config = config or ChartConfig()

    def plot_distribution(self, result: MonteCarloResult) -> go.Figure:
        """Plot return distribution with VaR/CVaR."""
        returns = [s.total_return for s in result.simulations]

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='steelblue',
        ))

        # VaR line
        fig.add_vline(x=result.value_at_risk, line_dash="dash",
                     line_color="red", annotation_text=f"VaR: {result.value_at_risk:.1f}%")

        # CVaR line
        fig.add_vline(x=result.conditional_var, line_dash="dot",
                     line_color="darkred", annotation_text=f"CVaR: {result.conditional_var:.1f}%")

        fig.update_layout(
            title="Monte Carlo Return Distribution",
            xaxis_title="Total Return (%)",
            yaxis_title="Frequency",
            width=self.config.width,
            height=600,
            template=self.config.template,
        )

        return fig

    def plot_risk_dashboard(self, result: MonteCarloResult) -> go.Figure:
        """Create comprehensive risk dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Return Distribution", "Confidence Intervals",
                          "Risk Metrics", "Percentiles"),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "bar"}]],
        )

        returns = [s.total_return for s in result.simulations]

        # 1. Distribution
        fig.add_trace(go.Histogram(x=returns, nbinsx=40, marker_color='steelblue'), row=1, col=1)

        # 2. Confidence Intervals
        fig.add_trace(go.Bar(
            x=['Lower', 'Mean', 'Upper'],
            y=[result.return_ci_lower, result.mean_return, result.return_ci_upper],
            marker_color=['red', 'blue', 'green'],
        ), row=1, col=2)

        # 3. Risk Metrics Table
        fig.add_trace(go.Table(
            header=dict(values=["Metric", "Value"], fill_color='#3498db', font=dict(color='white')),
            cells=dict(values=[
                ["VaR 95%", "CVaR 95%", "Prob Loss>10%", "Prob Profit"],
                [f"{result.value_at_risk:.1f}%", f"{result.conditional_var:.1f}%",
                 f"{result.prob_loss_exceeds_10pct:.1%}", f"{result.prob_profit:.1%}"]
            ]),
        ), row=2, col=1)

        # 4. Percentiles
        percentiles = [result.percentile_1, result.percentile_25, result.median_return,
                      result.percentile_75, result.percentile_99]
        fig.add_trace(go.Bar(
            x=['P1', 'P25', 'P50', 'P75', 'P99'],
            y=percentiles,
            marker_color=['darkred', 'orange', 'blue', 'lightgreen', 'green'],
        ), row=2, col=2)

        fig.update_layout(
            title=f"Monte Carlo Risk Dashboard - {result.strategy_id}",
            width=1400,
            height=900,
            template=self.config.template,
            showlegend=False,
        )

        return fig

    def save(self, fig: go.Figure, filename: str, format: str = "html"):
        """Save figure."""
        save_chart(fig, filename, format)
