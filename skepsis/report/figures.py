"""Plotly figure builders. Figures are returned bare; render.py handles embedding."""

import numpy as np
import plotly.graph_objects as go

from skepsis.core.bootstrap import BootstrapResult
from skepsis.core.pbo import PboResult

_ACCENT = "#2563eb"
_MUTED = "#94a3b8"
_ALERT = "#dc2626"
_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=50, r=20, t=40, b=40),
    height=320,
    font=dict(family="system-ui, sans-serif", size=13),
)


def equity_curve_figure(returns: np.ndarray) -> go.Figure:
    equity = np.cumprod(1.0 + returns)
    fig = go.Figure(go.Scatter(y=equity, mode="lines", line=dict(color=_ACCENT, width=1.5)))
    fig.update_layout(title="Equity curve (chosen strategy)", **_LAYOUT)
    fig.update_yaxes(title="growth of 1.0")
    fig.update_xaxes(title="period")
    return fig


def bootstrap_figure(boot: BootstrapResult) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=boot.sharpe_distribution, nbinsx=60, marker_color=_MUTED,
                      name="bootstrap Sharpe")
    fig.add_vline(x=boot.sharpe_obs, line_color=_ALERT,
                  annotation_text=f"observed {boot.sharpe_obs:.2f}")
    for x in boot.sharpe_ci:
        fig.add_vline(x=x, line_dash="dot", line_color=_ACCENT)
    fig.update_layout(
        title=f"Bootstrap Sharpe distribution (no-skill p = {boot.p_value_no_skill:.4f})",
        showlegend=False, **_LAYOUT,
    )
    fig.update_xaxes(title="annualized Sharpe")
    return fig


def pbo_figure(pbo: PboResult) -> go.Figure:
    fig = go.Figure(go.Histogram(x=pbo.logits, nbinsx=40, marker_color=_MUTED))
    fig.add_vline(x=0.0, line_color=_ALERT, annotation_text="λ = 0")
    fig.update_layout(
        title=f"OOS rank logits across {pbo.n_combinations} CSCV splits (PBO = {pbo.value:.3f})",
        showlegend=False, **_LAYOUT,
    )
    fig.update_xaxes(title="logit λ (≤ 0 means IS winner in bottom half OOS)")
    return fig


def sensitivity_figure(
    params: np.ndarray, param_names: list[str], metrics: np.ndarray, chosen_index: int
) -> go.Figure:
    d = params.shape[1]
    chosen_marker = dict(size=14, color="rgba(0,0,0,0)",
                         line=dict(color=_ALERT, width=2), symbol="circle-open")
    if d == 1:
        order = np.argsort(params[:, 0])
        fig = go.Figure(go.Scatter(x=params[order, 0], y=metrics[order],
                                   mode="lines+markers", line=dict(color=_ACCENT)))
        fig.add_scatter(x=[params[chosen_index, 0]], y=[metrics[chosen_index]],
                        mode="markers", marker=chosen_marker, name="chosen")
        fig.update_xaxes(title=param_names[0])
        fig.update_yaxes(title="annualized Sharpe")
    else:
        fig = go.Figure(go.Scatter(
            x=params[:, 0], y=params[:, 1], mode="markers",
            marker=dict(size=16, symbol="square", color=metrics,
                        colorscale="RdYlGn", showscale=True,
                        colorbar=dict(title="ann. Sharpe")),
        ))
        fig.add_scatter(x=[params[chosen_index, 0]], y=[params[chosen_index, 1]],
                        mode="markers", marker=chosen_marker, name="chosen")
        fig.update_xaxes(title=param_names[0])
        fig.update_yaxes(title=param_names[1])
    title = "Parameter sensitivity"
    if d > 2:
        title += f" (first 2 of {d} params shown)"
    fig.update_layout(title=title, showlegend=False, **_LAYOUT)
    return fig
