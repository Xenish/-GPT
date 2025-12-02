"""
Parameter Heatmap Visualization.

Visualizes parameter search results as heatmaps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from finantradealgo.research.visualization.charts import ChartConfig, save_chart


class ParameterHeatmapVisualizer:
    """
    Visualize parameter search results as heatmaps.

    Shows how performance varies across parameter combinations.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize parameter heatmap visualizer.

        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()

    def plot_2d_heatmap(
        self,
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = "sharpe",
        aggregation: str = "mean",
    ) -> go.Figure:
        """
        Plot 2D parameter heatmap.

        Args:
            results_df: Parameter search results DataFrame
            param1: First parameter (x-axis)
            param2: Second parameter (y-axis)
            metric: Metric to visualize (default: sharpe)
            aggregation: Aggregation method if multiple values (mean, max, min)

        Returns:
            Plotly figure
        """
        # Pivot data
        pivot = results_df.pivot_table(
            values=metric,
            index=f"param_{param2}",
            columns=f"param_{param1}",
            aggfunc=aggregation,
        )

        # Create heatmap
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=pivot.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 10},
            hovertemplate=f'{param1}: %{{x}}<br>{param2}: %{{y}}<br>{metric}: %{{z:.4f}}<extra></extra>',
            colorbar=dict(title=metric.replace("_", " ").title()),
        ))

        # Update layout
        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Heatmap: {param1} vs {param2}",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
            xaxis_title=param1,
            yaxis_title=param2,
        )

        return fig

    def plot_parameter_sensitivity(
        self,
        results_df: pd.DataFrame,
        parameters: list,
        metric: str = "sharpe",
    ) -> go.Figure:
        """
        Plot parameter sensitivity analysis.

        Shows how each parameter affects performance.

        Args:
            results_df: Parameter search results
            parameters: List of parameter names
            metric: Metric to analyze

        Returns:
            Plotly figure
        """
        from plotly.subplots import make_subplots

        n_params = len(parameters)
        rows = (n_params + 1) // 2
        cols = 2 if n_params > 1 else 1

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[p.replace("param_", "") for p in parameters],
            vertical_spacing=0.12,
            horizontal_spacing=0.15,
        )

        for i, param in enumerate(parameters):
            row = (i // cols) + 1
            col = (i % cols) + 1

            param_col = f"param_{param}" if not param.startswith("param_") else param

            if param_col not in results_df.columns:
                continue

            # Group by parameter value
            grouped = results_df.groupby(param_col)[metric].agg(['mean', 'std', 'count'])

            fig.add_trace(
                go.Scatter(
                    x=grouped.index,
                    y=grouped['mean'],
                    mode='lines+markers',
                    name=param,
                    error_y=dict(type='data', array=grouped['std'], visible=True),
                    hovertemplate=f'{param}: %{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>',
                ),
                row=row,
                col=col,
            )

            # Add reference line at 0
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=row, col=col)

        # Update layout
        fig.update_layout(
            title=f"Parameter Sensitivity Analysis - {metric.replace('_', ' ').title()}",
            width=self.config.width,
            height=400 * rows,
            template=self.config.template,
            showlegend=False,
        )

        return fig

    def plot_top_combinations(
        self,
        results_df: pd.DataFrame,
        metric: str = "sharpe",
        top_n: int = 20,
    ) -> go.Figure:
        """
        Plot top parameter combinations.

        Args:
            results_df: Parameter search results
            metric: Metric to rank by
            top_n: Number of top combinations to show

        Returns:
            Plotly figure
        """
        # Get top N results
        top_results = results_df.nlargest(top_n, metric)

        # Create labels for each combination
        param_cols = [col for col in top_results.columns if col.startswith("param_")]

        labels = []
        for _, row in top_results.iterrows():
            label_parts = [f"{col.replace('param_', '')}: {row[col]}" for col in param_cols]
            labels.append("<br>".join(label_parts))

        # Create bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(range(1, len(top_results) + 1)),
            y=top_results[metric],
            text=top_results[metric],
            texttemplate='%{text:.4f}',
            textposition='outside',
            hovertemplate='Rank: %{x}<br>%{customdata}<br>' + f'{metric}: %{{y:.4f}}<extra></extra>',
            customdata=labels,
            marker_color=self.config.color_scheme[0],
        ))

        # Update layout
        fig.update_layout(
            title=f"Top {top_n} Parameter Combinations by {metric.replace('_', ' ').title()}",
            width=self.config.width,
            height=600,
            template=self.config.template,
            xaxis_title="Rank",
            yaxis_title=metric.replace("_", " ").title(),
            showlegend=False,
        )

        return fig

    def plot_correlation_matrix(
        self,
        results_df: pd.DataFrame,
        metric: str = "sharpe",
    ) -> go.Figure:
        """
        Plot correlation matrix between parameters and metric.

        Args:
            results_df: Parameter search results
            metric: Target metric

        Returns:
            Plotly figure
        """
        # Get parameter columns
        param_cols = [col for col in results_df.columns if col.startswith("param_")]

        # Calculate correlations with metric
        correlations = []
        for param in param_cols:
            try:
                # Convert to numeric
                param_values = pd.to_numeric(results_df[param], errors='coerce')
                metric_values = results_df[metric]

                # Remove NaN pairs
                mask = ~(param_values.isna() | metric_values.isna())

                if mask.sum() > 1:
                    corr = param_values[mask].corr(metric_values[mask])
                    correlations.append({
                        'parameter': param.replace('param_', ''),
                        'correlation': corr,
                    })
            except Exception:
                pass

        if not correlations:
            fig = go.Figure()
            fig.add_annotation(
                text="Unable to calculate correlations",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
            )
            return fig

        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

        # Create bar chart
        fig = go.Figure()

        colors = ['green' if x > 0 else 'red' for x in corr_df['correlation']]

        fig.add_trace(go.Bar(
            x=corr_df['parameter'],
            y=corr_df['correlation'],
            marker_color=colors,
            text=corr_df['correlation'],
            texttemplate='%{text:.3f}',
            textposition='outside',
            hovertemplate='%{x}<br>Correlation: %{y:.4f}<extra></extra>',
        ))

        # Add reference line at 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        # Update layout
        fig.update_layout(
            title=f"Parameter Correlation with {metric.replace('_', ' ').title()}",
            width=self.config.width,
            height=500,
            template=self.config.template,
            xaxis_title="Parameter",
            yaxis_title=f"Correlation with {metric}",
            showlegend=False,
        )

        return fig

    def save(self, fig: go.Figure, filename: str, format: str = "html") -> None:
        """Save figure to file."""
        save_chart(fig, filename, format)
