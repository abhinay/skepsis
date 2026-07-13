"""Assemble the report: figures -> divs, template -> single self-contained HTML."""

from importlib import resources
from typing import TYPE_CHECKING, Any

import jinja2
from plotly.offline import get_plotlyjs

from skepsis.report import figures as figmod

if TYPE_CHECKING:
    from skepsis.evaluate import Result

_FIG_HTML_KW: dict[str, Any] = dict(
    full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
)


def render_html(result: "Result") -> str:
    figs: dict[str, str] = {
        "equity": figmod.equity_curve_figure(result.returns).to_html(**_FIG_HTML_KW)
    }
    if result.pbo is not None:
        figs["pbo"] = figmod.pbo_figure(result.pbo).to_html(**_FIG_HTML_KW)
    if result.bootstrap is not None:
        figs["bootstrap"] = figmod.bootstrap_figure(result.bootstrap).to_html(**_FIG_HTML_KW)
    if result.sensitivity is not None and result.params is not None:
        assert result.param_names is not None and result.trial_metrics is not None
        figs["sensitivity"] = figmod.sensitivity_figure(
            result.params, result.param_names, result.trial_metrics,
            result.sensitivity.chosen_index,
        ).to_html(**_FIG_HTML_KW)

    template_text = (
        resources.files("skepsis.report").joinpath("template.html.j2").read_text("utf-8")
    )
    env = jinja2.Environment(autoescape=True)
    return env.from_string(template_text).render(
        verdict=result.verdict,
        psr=result.psr,
        dsr=result.deflated_sharpe,
        pbo=result.pbo,
        bootstrap=result.bootstrap,
        sensitivity=result.sensitivity,
        skipped=result.skipped,
        warnings=result.warnings,
        meta=result.meta,
        figures=figs,
        plotly_js=get_plotlyjs(),
    )
