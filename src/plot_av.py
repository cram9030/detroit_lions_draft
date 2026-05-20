"""Plotly visualization functions for NFL draft pick AV analysis.

All plots use the Viridis color palette: dark blue (#440154) for primary lines
progressing through teal/green (#21908c) for shaded regions. The bright yellow
end of the Viridis palette is not used.

PNG and SVG static exports require the ``kaleido`` package to be installed.
Animated figures can only be exported as HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from scipy.stats import skewnorm

from src import curve_fitting as _cf

# Viridis color list (10 entries, dark purple → bright yellow)
_VIRIDIS = px.colors.sequential.Viridis

# Use dark blue for the primary line, mid-teal for the IQR band
_LINE_COLOR = _VIRIDIS[0]   # #440154  dark purple/blue
_BAND_COLOR = _VIRIDIS[5]   # ~#21908c teal/green


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string to an ``rgba(r, g, b, alpha)`` string.

    Args:
        hex_color: Hex color in ``#rrggbb`` format.
        alpha: Opacity in the range [0, 1].

    Returns:
        RGBA string suitable for Plotly ``fillcolor`` or ``line.color``.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_pick_av(
    stats_df: pl.DataFrame,
    title: str,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Create a line chart of pick number vs. mean rookie contract AV with IQR band.

    The mean is plotted as a solid dark-blue line. A shaded teal region spans
    the 25th–75th percentile range. Picks where the IQR is unavailable (fewer
    than 2 players) are excluded from the shaded region but retained in the
    mean line.

    Input DataFrame columns required:
        - ``Pick`` (Int64): Overall pick number, sorted ascending.
        - ``mean`` (Float64): Mean rookie contract AV at each pick.
        - ``25%`` (Float64): 25th percentile; may be null for sparse picks.
        - ``75%`` (Float64): 75th percentile; may be null for sparse picks.

    Args:
        stats_df: Output of :func:`annual_av_analysis.pick_based_stats` or a
            single window's DataFrame from
            :func:`annual_av_analysis.rolling_window_pick_stats`.
        title: Chart title displayed at the top of the figure.
        export_path: If provided, the figure is saved to this path.
        export_format: Required when ``export_path`` is set. One of
            ``"html"``, ``"png"``, or ``"svg"``. PNG and SVG require the
            ``kaleido`` package.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None,
            or if ``export_format`` is not one of the supported values.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    # Mean line — keep all picks up to 250, drop only where mean is null
    mean_df = stats_df.filter(pl.col("Pick") <= 250).select(["Pick", "mean"]).drop_nulls()
    picks_mean = mean_df["Pick"].to_list()
    means = mean_df["mean"].to_list()

    # IQR band — only picks where both quantiles are available
    iqr_df = stats_df.filter(pl.col("Pick") <= 250).select(["Pick", "25%", "75%"]).drop_nulls()
    picks_iqr = iqr_df["Pick"].to_list()
    q25 = iqr_df["25%"].to_list()
    q75 = iqr_df["75%"].to_list()

    band_fill = _hex_to_rgba(_BAND_COLOR, 0.3)

    fig = go.Figure()

    # IQR shaded region (added first so line renders on top)
    fig.add_trace(
        go.Scatter(
            x=picks_iqr + picks_iqr[::-1],
            y=q75 + q25[::-1],
            fill="toself",
            fillcolor=band_fill,
            line=dict(color="rgba(0,0,0,0)"),
            name="25%–75% IQR",
            showlegend=True,
        )
    )

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=picks_mean,
            y=means,
            mode="lines",
            line=dict(color=_LINE_COLOR, width=2),
            name="Mean Rookie Contract AV",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Pick Number", range=[1, 250]),
        yaxis_title="Rookie Contract AV",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


def plot_animated_rolling_window(
    rolling_fit_dict: dict[int, dict],
    export_path: str | Path | None = None,
) -> go.Figure:
    """Create an animated figure cycling through rolling-window exponential fit results.

    Each frame corresponds to one center year from ``rolling_fit_dict`` and
    mirrors the three-layer layout of :func:`plot_exponential_fit_means`:
    the 25th–75th percentile IQR band, the fitted exponential curve, and the
    per-pick mean scatter. A slider and Play button allow interactive scrubbing.

    Input dict values — keys required per fit result (output of
    :func:`annual_av_analysis.exponential_av_fit_means`):
        - ``picks`` (ndarray): Unique pick numbers used in the fit.
        - ``means`` (ndarray): Mean AV per pick, aligned with ``picks``.
        - ``x_fit`` (ndarray): Dense pick axis for the smooth fitted curve.
        - ``y_fit`` (ndarray): Fitted AV values at ``x_fit`` points.
        - ``iqr_picks`` (ndarray): Pick numbers where both percentiles are available.
        - ``q25`` (ndarray): 25th percentile AV per pick, aligned with ``iqr_picks``.
        - ``q75`` (ndarray): 75th percentile AV per pick, aligned with ``iqr_picks``.
        - ``popt`` (ndarray): Fitted parameters ``[a, b, c]`` shown in legend.

    Args:
        rolling_fit_dict: Mapping of ``center_year`` → fit result dict,
            produced by calling :func:`annual_av_analysis.exponential_av_fit_means`
            on each window from :func:`annual_av_analysis.rolling_window_pick_stats`.
        export_path: If provided, saves the figure as an HTML file. Must end
            with ``.html`` — animated figures cannot be exported to static
            image formats.

    Returns:
        Plotly Figure with animation frames and a year slider.

    Raises:
        ValueError: If ``export_path`` is provided but does not end with
            ``.html``.
    """
    if export_path is not None and not str(export_path).endswith(".html"):
        raise ValueError(
            "Animated figures can only be exported as HTML. "
            f"Got: {export_path!r}. Use a path ending with '.html'."
        )

    center_years = sorted(rolling_fit_dict.keys())
    band_fill = _hex_to_rgba(_BAND_COLOR, 0.25)
    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.8)

    # Compute global y range across all frames so the axis never jumps
    all_y = [
        v
        for fr in rolling_fit_dict.values()
        for v in (
            list(fr["q75"]) + list(fr["q25"]) + list(fr["y_fit"]) + list(fr["stat_values"])
        )
    ]
    y_min = min(all_y)
    y_max = max(all_y)
    y_pad = (y_max - y_min) * 0.05
    y_range = [y_min - y_pad, y_max + y_pad]

    def _build_traces(fit_result: dict) -> list[go.BaseTraceType]:
        iqr_picks = list(fit_result["iqr_picks"])
        q25 = list(fit_result["q25"])
        q75 = list(fit_result["q75"])
        x_fit = fit_result["x_fit"]
        y_fit = fit_result["y_fit"]
        picks = fit_result["picks"]
        stat_values = fit_result["stat_values"]
        a, b, c = fit_result["popt"]

        band = go.Scatter(
            x=iqr_picks + iqr_picks[::-1],
            y=q75 + q25[::-1],
            fill="toself",
            fillcolor=band_fill,
            line=dict(color="rgba(0,0,0,0)"),
            name="25%–75% IQR",
            showlegend=True,
        )
        curve = go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            line=dict(color=_LINE_COLOR, width=2),
            name=f"Fit: {a:.2f}·exp(−{b:.4f}·pick) + {c:.2f}",
        )
        scatter = go.Scatter(
            x=picks,
            y=stat_values,
            mode="markers",
            marker=dict(color=obs_color, size=5),
            name="Mean AV per pick",
        )
        return [band, curve, scatter]

    # Build frames
    frames = [
        go.Frame(data=_build_traces(rolling_fit_dict[yr]), name=str(yr))
        for yr in center_years
    ]

    # Initial display = first frame
    initial_traces = _build_traces(rolling_fit_dict[center_years[0]])

    fig = go.Figure(data=initial_traces, frames=frames)

    # Slider steps
    slider_steps = [
        dict(
            method="animate",
            args=[[str(yr)], dict(mode="immediate", frame=dict(duration=300, redraw=True))],
            label=str(yr),
        )
        for yr in center_years
    ]

    fig.update_layout(
        title=f"Exponential Fit — Rookie Contract AV by Draft Pick — Rolling {center_years[0]}–{center_years[-1]}",
        xaxis=dict(title="Pick Number", range=[1, 250]),
        yaxis=dict(title="Rookie Contract AV", range=y_range, dtick=5),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.05,
                x=0,
                xanchor="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=100),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Center Year: ", visible=True, xanchor="center"),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
    )

    if export_path is not None:
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))

    return fig


def _export_figure(
    fig: go.Figure,
    export_path: str | Path,
    export_format: str,
) -> None:
    """Save a static figure to disk in the requested format.

    Args:
        fig: Plotly Figure to export.
        export_path: Destination file path.
        export_format: One of ``"html"``, ``"png"``, or ``"svg"``.

    Raises:
        ValueError: If ``export_format`` is not supported.
    """
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "html":
        fig.write_html(str(path))
    elif export_format in ("png", "svg"):
        fig.write_image(str(path))
    else:
        raise ValueError(
            f"Unsupported export_format: {export_format!r}. Use 'html', 'png', or 'svg'."
        )


def plot_exponential_fit(
    fit_result: dict,
    title: str,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Plot an exponential decay fit to mean rookie contract AV by pick number.

    Displays three layers:
    1. Observed mean AV per pick as scatter markers (semi-transparent).
    2. The fitted ``f(pick) = a·exp(-b·pick) + c`` curve as a solid dark-blue line.
    3. The 1-sigma confidence band as a shaded teal region.

    Input ``fit_result`` keys required (output of
    :func:`annual_av_analysis.exponential_av_fit`):
        - ``picks`` (ndarray): Individual pick numbers, one per player.
        - ``av_values`` (ndarray): Individual rookie contract AV values, one per player.
        - ``x_fit`` (ndarray): Dense pick axis for the smooth fitted curve.
        - ``y_fit`` (ndarray): Fitted AV values at ``x_fit`` points.
        - ``y_upper`` (ndarray): Upper 1-sigma bound at ``x_fit`` points.
        - ``y_lower`` (ndarray): Lower 1-sigma bound at ``x_fit`` points.
        - ``popt`` (ndarray): Fitted parameters ``[a, b, c]`` shown in legend.

    Args:
        fit_result: Return value of :func:`annual_av_analysis.exponential_av_fit`.
        title: Chart title displayed at the top of the figure.
        export_path: If provided, the figure is saved to this path.
        export_format: Required when ``export_path`` is set. One of
            ``"html"``, ``"png"``, or ``"svg"``.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    picks = fit_result["picks"]
    av_values = fit_result["av_values"]
    x_fit = fit_result["x_fit"]
    y_fit = fit_result["y_fit"]
    y_upper = fit_result["y_upper"]
    y_lower = fit_result["y_lower"]
    a, b, c = fit_result["popt"]

    band_fill = _hex_to_rgba(_BAND_COLOR, 0.25)
    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.3)  # mid-viridis, lighter for dense scatter

    fig = go.Figure()

    # 1-sigma band (rendered first, behind everything)
    x_band = list(x_fit) + list(x_fit[::-1])
    y_band = list(y_upper) + list(y_lower[::-1])
    fig.add_trace(
        go.Scatter(
            x=x_band,
            y=y_band,
            fill="toself",
            fillcolor=band_fill,
            line=dict(color="rgba(0,0,0,0)"),
            name="±1σ confidence",
            showlegend=True,
        )
    )

    # Fitted curve
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            line=dict(color=_LINE_COLOR, width=2),
            name=f"Fit: {a:.2f}·exp(−{b:.4f}·pick) + {c:.2f}",
        )
    )

    # Individual player AV scatter (one point per player)
    fig.add_trace(
        go.Scatter(
            x=picks,
            y=av_values,
            mode="markers",
            marker=dict(color=obs_color, size=3),
            name="Individual player AV",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Pick Number", range=[1, 250]),
        yaxis_title="Rookie Contract AV",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


def plot_eavar(
    fit_result: dict,
    replacement_level: float,
    title: str,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Plot the Expected AV Above Replacement (EAVAR) curve by draft pick.

    Shifts the exponential fit down by ``replacement_level`` so the y-axis
    represents how much above replacement a given pick is expected to produce
    over the rookie contract window. A horizontal dashed line at y = 0 marks
    the replacement level baseline.

    Displays three layers:
    1. Individual player AV above replacement (scatter, semi-transparent).
    2. EAVAR fitted curve (solid dark-blue line).
    3. 1-sigma confidence band (shaded teal region).
    4. Horizontal dashed line at y = 0 (replacement level).

    Args:
        fit_result: Return value of :func:`annual_av_analysis.exponential_av_fit`.
            Must contain ``picks``, ``av_values``, ``x_fit``, ``y_fit``,
            ``y_upper``, ``y_lower``, and ``popt`` keys.
        replacement_level: Replacement-level AV baseline to subtract (mean
            per-season AV × 4 for the rookie contract window).
        title: Chart title displayed at the top of the figure.
        export_path: If provided, the figure is saved to this path.
        export_format: Required when ``export_path`` is set. One of
            ``"html"``, ``"png"``, or ``"svg"``.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    picks = fit_result["picks"]
    av_values = fit_result["av_values"] - replacement_level
    x_fit = fit_result["x_fit"]
    y_fit = fit_result["y_fit"] - replacement_level
    y_upper = fit_result["y_upper"] - replacement_level
    y_lower = fit_result["y_lower"] - replacement_level
    a, b, c = fit_result["popt"]

    band_fill = _hex_to_rgba(_BAND_COLOR, 0.25)
    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.3)

    fig = go.Figure()

    # 1-sigma band (rendered first, behind everything)
    x_band = list(x_fit) + list(x_fit[::-1])
    y_band = list(y_upper) + list(y_lower[::-1])
    fig.add_trace(
        go.Scatter(
            x=x_band,
            y=y_band,
            fill="toself",
            fillcolor=band_fill,
            line=dict(color="rgba(0,0,0,0)"),
            name="±1σ confidence",
            showlegend=True,
        )
    )

    # EAVAR fitted curve
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            line=dict(color=_LINE_COLOR, width=2),
            name=f"EAVAR: {a:.2f}·exp(−{b:.4f}·pick) + {c:.2f} − {replacement_level:.1f}",
        )
    )

    # Individual player EAVAR scatter
    fig.add_trace(
        go.Scatter(
            x=picks,
            y=av_values,
            mode="markers",
            marker=dict(color=obs_color, size=3),
            name="Individual player EAVAR",
        )
    )

    # Replacement level baseline at y = 0
    fig.add_hline(
        y=0,
        line=dict(color="black", width=1.5, dash="dash"),
        annotation_text=f"Replacement Level ({replacement_level:.1f} AV)",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Pick Number", range=[1, 250]),
        yaxis_title="Expected AV Above Replacement (Rookie Contract Window)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


def plot_exp_and_log_fit(
    exp_fit_result: dict,
    log_fit_result: dict,
    title: str,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Plot exponential and logarithmic decay fits together on one figure.

    Displays four layers:
    1. Individual player AV scatter (semi-transparent).
    2. Exponential fit curve + 1-sigma band (dark blue / teal).
    3. Logarithmic fit curve + 1-sigma band (green / light green).

    Args:
        exp_fit_result: Return value of :func:`annual_av_analysis.exponential_av_fit`.
        log_fit_result: Return value of :func:`annual_av_analysis.logarithmic_av_fit`.
        title: Chart title.
        export_path: If provided, the figure is saved to this path.
        export_format: Required when ``export_path`` is set.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    _LOG_LINE_COLOR = _VIRIDIS[7]   # green-yellow for log fit line
    _LOG_BAND_COLOR = _VIRIDIS[6]   # mid-green for log fit band

    picks = exp_fit_result["picks"]
    av_values = exp_fit_result["av_values"]
    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.3)

    exp_band_fill = _hex_to_rgba(_BAND_COLOR, 0.25)
    log_band_fill = _hex_to_rgba(_LOG_BAND_COLOR, 0.25)

    a_e, b_e, c_e = exp_fit_result["popt"]
    a_l, b_l = log_fit_result["popt"]

    fig = go.Figure()

    # Exponential 1-sigma band
    x_exp = exp_fit_result["x_fit"]
    x_exp_band = list(x_exp) + list(x_exp[::-1])
    y_exp_band = list(exp_fit_result["y_upper"]) + list(exp_fit_result["y_lower"][::-1])
    fig.add_trace(go.Scatter(
        x=x_exp_band, y=y_exp_band,
        fill="toself", fillcolor=exp_band_fill,
        line=dict(color="rgba(0,0,0,0)"),
        name="Exp ±1σ", showlegend=True,
    ))

    # Exponential fit curve
    fig.add_trace(go.Scatter(
        x=x_exp, y=exp_fit_result["y_fit"],
        mode="lines",
        line=dict(color=_LINE_COLOR, width=2),
        name=f"Exp: {a_e:.2f}·exp(−{b_e:.4f}·pick) + {c_e:.2f}",
    ))

    # Logarithmic 1-sigma band
    x_log = log_fit_result["x_fit"]
    x_log_band = list(x_log) + list(x_log[::-1])
    y_log_band = list(log_fit_result["y_upper"]) + list(log_fit_result["y_lower"][::-1])
    fig.add_trace(go.Scatter(
        x=x_log_band, y=y_log_band,
        fill="toself", fillcolor=log_band_fill,
        line=dict(color="rgba(0,0,0,0)"),
        name="Log ±1σ", showlegend=True,
    ))

    # Logarithmic fit curve
    fig.add_trace(go.Scatter(
        x=x_log, y=log_fit_result["y_fit"],
        mode="lines",
        line=dict(color=_LOG_LINE_COLOR, width=2),
        name=f"Log: {a_l:.2f}·ln(pick) + {b_l:.2f}",
    ))

    # Individual player AV scatter
    fig.add_trace(go.Scatter(
        x=picks, y=av_values,
        mode="markers",
        marker=dict(color=obs_color, size=3),
        name="Individual player AV",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Pick Number", range=[1, 250]),
        yaxis_title="Rookie Contract AV",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


def plot_exponential_fit_means(
    fit_result: dict,
    title: str,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Plot an exponential decay fit to per-pick mean rookie contract AV.

    Displays three layers:
    1. Per-pick mean AV as scatter markers (one point per pick position).
    2. The fitted ``f(pick) = a·exp(-b·pick) + c`` curve as a solid dark-blue line.
    3. The 25th–75th percentile IQR as a shaded teal region.

    Input ``fit_result`` keys required (output of
    :func:`annual_av_analysis.exponential_av_fit_means`):
        - ``picks`` (ndarray): Unique pick numbers from the stats data.
        - ``means`` (ndarray): Mean AV per pick, aligned with ``picks``.
        - ``x_fit`` (ndarray): Dense pick axis for the smooth fitted curve.
        - ``y_fit`` (ndarray): Fitted AV values at ``x_fit`` points.
        - ``iqr_picks`` (ndarray): Pick numbers where both percentiles are available.
        - ``q25`` (ndarray): 25th percentile AV per pick, aligned with ``iqr_picks``.
        - ``q75`` (ndarray): 75th percentile AV per pick, aligned with ``iqr_picks``.
        - ``popt`` (ndarray): Fitted parameters ``[a, b, c]`` shown in legend.

    Args:
        fit_result: Return value of
            :func:`annual_av_analysis.exponential_av_fit_means`.
        title: Chart title displayed at the top of the figure.
        export_path: If provided, the figure is saved to this path.
        export_format: Required when ``export_path`` is set. One of
            ``"html"``, ``"png"``, or ``"svg"``.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    picks = fit_result["picks"]
    stat_values = fit_result["stat_values"]
    x_fit = fit_result["x_fit"]
    y_fit = fit_result["y_fit"]
    iqr_picks = list(fit_result["iqr_picks"])
    q25 = list(fit_result["q25"])
    q75 = list(fit_result["q75"])
    a, b, c = fit_result["popt"]

    band_fill = _hex_to_rgba(_BAND_COLOR, 0.25)
    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.8)  # more opaque — one point per pick

    fig = go.Figure()

    # IQR band (25th–75th percentile)
    fig.add_trace(
        go.Scatter(
            x=iqr_picks + iqr_picks[::-1],
            y=q75 + q25[::-1],
            fill="toself",
            fillcolor=band_fill,
            line=dict(color="rgba(0,0,0,0)"),
            name="25%–75% IQR",
            showlegend=True,
        )
    )

    # Fitted curve
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            line=dict(color=_LINE_COLOR, width=2),
            name=f"Fit: {a:.2f}·exp(−{b:.4f}·pick) + {c:.2f}",
        )
    )

    # Per-pick mean scatter (one point per unique pick)
    fig.add_trace(
        go.Scatter(
            x=picks,
            y=stat_values,
            mode="markers",
            marker=dict(color=obs_color, size=5),
            name="Mean AV per pick",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Pick Number", range=[1, 250]),
        yaxis_title="Rookie Contract AV",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


# Colors for trade-chart lines — visually distinct from Viridis
_TRADE_COLORS = ["#e15759", "#f28e2b", "#76b7b2", "#59a14f", "#b07aa1"]


def plot_normalized_pick_value_comparison(
    fits: dict[str, dict],
    trade_charts: dict[str, pl.DataFrame],
    title: str = "Pick Value Comparison — Normalized to Pick 1",
    max_pick: int = 224,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Plot multiple pick-value models on a common normalized scale.

    Evaluates each exponential fit at integer picks 1–``max_pick`` and
    normalizes so that pick 1 = 1.0. Trade charts are normalized the same way
    using the value at ``Pick == 1``. All series are overlaid on one figure for
    direct comparison.

    Args:
        fits: Mapping of label → exponential fit result dict (output of
            :func:`annual_av_analysis.exponential_av_fit` or
            :func:`annual_av_analysis.exponential_av_fit_means`). Must contain
            a ``popt`` key with fitted parameters ``[a, b, c]``.
        trade_charts: Mapping of label → DataFrame with exactly two columns:
            ``Pick`` (Int64) and ``Value`` (Float64/Int64), already sorted by
            ``Pick`` ascending. The value at ``Pick == 1`` is used as the
            normalization reference. Duplicate pick numbers must be removed
            before passing.
        title: Chart title.
        max_pick: Last pick number to show on the x-axis. Default ``224``
            (the upper bound of the Jimmy Johnson chart).
        export_path: If provided, saves the figure to this path.
        export_format: Required when ``export_path`` is set.

    Returns:
        Plotly Figure with all AV models (solid lines) and trade charts
        (dashed lines) normalized to pick 1 = 1.0.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    pick_axis = np.arange(1, max_pick + 1, dtype=float)

    n_fits = len(fits)
    fit_colors = [_VIRIDIS[0]] if n_fits == 1 else px.colors.sample_colorscale("Viridis", n_fits)

    fig = go.Figure()

    for i, (label, fit_result) in enumerate(fits.items()):
        a, b, c = fit_result["popt"]
        y = a * np.exp(-b * pick_axis) + c
        y_norm = y / y[0]
        color = fit_colors[i]

        fig.add_trace(go.Scatter(
            x=pick_axis.tolist(),
            y=y_norm.tolist(),
            mode="lines",
            line=dict(color=color, width=2),
            name=label,
            legendgroup="av_models",
            legendgrouptitle=dict(text="AV Models") if i == 0 else {},
        ))

    for i, (label, df) in enumerate(trade_charts.items()):
        color = _TRADE_COLORS[i % len(_TRADE_COLORS)]
        val_at_1 = df.filter(pl.col("Pick") == 1)["Value"][0]
        chart_df = df.filter(pl.col("Pick") <= max_pick)
        x = chart_df["Pick"].to_list()
        y_norm = [v / val_at_1 for v in chart_df["Value"].to_list()]

        fig.add_trace(go.Scatter(
            x=x,
            y=y_norm,
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            name=label,
            legendgroup="trade_charts",
            legendgrouptitle=dict(text="Trade Charts") if i == 0 else {},
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Overall Pick", range=[1, max_pick]),
        yaxis=dict(title="Normalized Value (Pick 1 = 1.0)"),
        template="plotly_white",
        legend=dict(groupclick="toggleitem"),
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


def plot_multi_fit_comparison(
    fits: dict[str, tuple[dict, "_cf.ModelDescriptor"]],
    picks: np.ndarray,
    av_values: np.ndarray,
    title: str,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Overlay multiple curve-fit models on one figure with individual player scatter.

    Each model in ``fits`` is evaluated at every integer pick from 1 to the
    maximum pick in the fit result and plotted as a solid line. Individual
    player AV observations are shown as a semi-transparent scatter underneath.

    Args:
        fits: Mapping of label → ``(fit_result, model)`` pairs. ``fit_result``
            is the output of :func:`curve_fitting.fit_individuals` or
            :func:`curve_fitting.fit_stats`; ``model`` is the same
            :data:`~curve_fitting.ModelDescriptor` used to produce it.
        picks: Individual player pick numbers for the scatter layer.
        av_values: Individual player AV values aligned with ``picks``.
        title: Chart title.
        export_path: If provided, the figure is saved to this path.
        export_format: Required when ``export_path`` is set.

    Returns:
        Plotly Figure with one line per model plus the scatter layer.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    n = len(fits)
    if n == 1:
        line_colors = [_VIRIDIS[0]]
    else:
        line_colors = px.colors.sample_colorscale("Viridis", n)

    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.2)

    fig = go.Figure()

    # Scatter first so all fit lines render on top
    fig.add_trace(go.Scatter(
        x=picks.tolist(),
        y=av_values.tolist(),
        mode="markers",
        marker=dict(color=obs_color, size=3),
        name="Individual player AV",
        legendgroup="scatter",
    ))

    for i, (label, (fit_result, model)) in enumerate(fits.items()):
        df = _cf.fit_result_to_dataframe(fit_result, model)
        color = line_colors[i]

        fig.add_trace(go.Scatter(
            x=df["pick"].to_list(),
            y=df["y_fit"].to_list(),
            mode="lines",
            line=dict(color=color, width=2),
            name=label,
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Pick Number", range=[1, 250]),
        yaxis_title="Rookie Contract AV",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


def plot_position_career_av(
    stats_df: pl.DataFrame,
    title: str,
    positions: list | None = None,
    show_percentile: bool = False,
    group_col: str = "Pos",
    max_years: int = 10,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Create a multi-line chart of annual AV by career year, one line per group.

    Groups are color-coded using the Viridis palette spread across the number
    plotted. The 25th–75th percentile band is off by default; enable it with
    ``show_percentile=True`` (most useful when plotting a single group for a
    detailed view).

    Works for any grouping column — pass ``group_col="Pos"`` (default) for
    position data from :func:`annual_av_analysis.position_career_stats` or
    ``group_col="Round"`` for round data from
    :func:`annual_av_analysis.round_career_stats`.

    Input DataFrame columns required:
        - Column named ``group_col``: Group label (position, round, etc.).
        - ``years_from_draft`` (Int64): Career year (0 = rookie season).
        - ``mean`` (Float64): Mean annual AV at this group/year.
        - ``25%`` (Float64): 25th percentile; used when ``show_percentile=True``.
        - ``75%`` (Float64): 75th percentile; used when ``show_percentile=True``.

    Args:
        stats_df: Output of :func:`annual_av_analysis.position_career_stats`
            or :func:`annual_av_analysis.round_career_stats`.
        title: Chart title displayed at the top of the figure.
        positions: Optional list of group values to include. If ``None``, all
            values present in ``stats_df[group_col]`` are plotted (sorted).
            When provided, groups are plotted in the order given, which also
            controls color assignment.
        show_percentile: If ``True``, draw a shaded 25th–75th percentile band
            behind each mean line. Default ``False``.
        group_col: Name of the column in ``stats_df`` that identifies each
            group (line). Default ``"Pos"``.
        max_years: Maximum ``years_from_draft`` value to plot (inclusive).
            Default ``10`` shows career years 0–9. Pass ``None`` to include
            all available years.
        export_path: If provided, the figure is saved to this path.
        export_format: Required when ``export_path`` is set. One of
            ``"html"``, ``"png"``, or ``"svg"``.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None,
            or if ``export_format`` is not one of the supported values.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    if positions is None:
        positions = sorted(stats_df[group_col].unique().to_list())

    # sample_colorscale requires at least 2 points; fall back to a single color for n=1
    if len(positions) == 1:
        colors = [_VIRIDIS[0]]
    else:
        colors = px.colors.sample_colorscale("Viridis", len(positions))

    fig = go.Figure()

    for group_val, color in zip(positions, colors):
        mask = pl.col(group_col) == group_val
        if max_years is not None:
            mask = mask & (pl.col("years_from_draft") < max_years)
        group_df = stats_df.filter(mask).sort("years_from_draft")
        if group_df.is_empty():
            continue

        x = group_df["years_from_draft"].to_list()
        y_mean = group_df["mean"].to_list()

        # Normalise color to rgba: sample_colorscale → "rgb(...)", _VIRIDIS → "#rrggbb"
        if color.startswith("#"):
            band_color = _hex_to_rgba(color, 0.2)
            line_color = color
        else:
            rgb_vals = color.replace("rgb(", "").replace(")", "").split(",")
            band_color = f"rgba({rgb_vals[0].strip()},{rgb_vals[1].strip()},{rgb_vals[2].strip()},0.2)"
            line_color = color

        if show_percentile:
            y_q25 = group_df["25%"].to_list()
            y_q75 = group_df["75%"].to_list()
            x_band = x + x[::-1]
            y_band = y_q75 + y_q25[::-1]
            fig.add_trace(
                go.Scatter(
                    x=x_band,
                    y=y_band,
                    fill="toself",
                    fillcolor=band_color,
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_mean,
                mode="lines",
                line=dict(color=line_color, width=2),
                name=str(group_val),
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Years from Draft", dtick=1),
        yaxis_title="Annual AV",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


def plot_skew_3d(
    skew_params_df: pl.DataFrame,
    player_av_df: pl.DataFrame,
    title: str = "Skew-Normal AV Distributions by Draft Pick",
    max_pick: int | None = None,
    av_range: tuple[float, float] = (0.0, 100.0),
    n_av_points: int = 300,
    degeneracy_threshold: float = 1e6,
    export_path: str | Path | None = None,
    export_format: Literal["html", "png", "svg"] | None = None,
) -> go.Figure:
    """Create a static 3D surface of skew-normal AV distributions by draft pick.

    For each pick with a valid skew-normal fit, evaluates the PDF across an AV
    range and stacks the results into a surface. Picks where ``|a| >
    degeneracy_threshold`` are skipped — roughly 85% of picks produce degenerate
    MLE estimates and are excluded.

    Displays two layers:
    1. ``go.Surface`` — probability density surface (X=pick, Y=AV, Z=density).
    2. ``go.Scatter3d`` — individual player AV values as dots on the z=0 floor.

    Args:
        skew_params_df: DataFrame with columns ``Pick (Int64)``, ``a (Float64)``,
            ``loc (Float64)``, ``scale (Float64)`` — the full ``skew_params.csv``
            contents. Degenerate picks are filtered internally.
        player_av_df: DataFrame with columns ``Player``, ``Pick``,
            ``Draft Year``, ``Draft Team``, ``rookie_contract_av`` — output of
            ``aggregate_player_av().collect()``.
        title: Chart title.
        max_pick: If set, restrict both surface and scatter to picks ≤ this value.
        av_range: ``(lo, hi)`` range for the AV axis. Default ``(0.0, 100.0)``.
        n_av_points: Number of AV sample points for PDF evaluation. Default 300.
        degeneracy_threshold: Picks with ``|a|`` above this are excluded.
        export_path: If provided, the figure is saved to this path.
        export_format: Required when ``export_path`` is set.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If ``export_path`` is set but ``export_format`` is None.
    """
    if export_path is not None and export_format is None:
        raise ValueError("export_format must be specified when export_path is provided.")

    valid_df = skew_params_df.filter(pl.col("a").abs() <= degeneracy_threshold)
    if max_pick is not None:
        valid_df = valid_df.filter(pl.col("Pick") <= max_pick)
    valid_df = valid_df.sort("Pick")

    av_arr = np.linspace(av_range[0], av_range[1], n_av_points)
    picks_list = valid_df["Pick"].to_list()
    n_picks = len(picks_list)

    z_matrix = np.zeros((n_av_points, n_picks))
    for j, row in enumerate(valid_df.iter_rows(named=True)):
        z_matrix[:, j] = skewnorm.pdf(av_arr, row["a"], row["loc"], row["scale"])

    scatter_df = player_av_df.filter(pl.col("Pick").is_in(picks_list))
    if max_pick is not None:
        scatter_df = scatter_df.filter(pl.col("Pick") <= max_pick)

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=picks_list,
            y=av_arr.tolist(),
            z=z_matrix.tolist(),
            colorscale="Viridis",
            opacity=0.85,
            showscale=True,
            colorbar=dict(title="Probability Density"),
            name="PDF Surface",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=scatter_df["Pick"].to_list(),
            y=scatter_df["rookie_contract_av"].to_list(),
            z=[0.0] * len(scatter_df),
            mode="markers",
            marker=dict(
                size=2,
                color=_hex_to_rgba(_VIRIDIS[3], 0.4),
                opacity=0.4,
            ),
            name="Individual Players",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Pick Number"),
            yaxis=dict(title="Rookie Contract AV", range=list(av_range)),
            zaxis=dict(title="Probability Density"),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        template="plotly_white",
    )

    if export_path is not None:
        _export_figure(fig, export_path, export_format)

    return fig


def plot_animated_skew_by_pick(
    skew_params_df: pl.DataFrame,
    player_av_df: pl.DataFrame,
    title: str = "Skew-Normal AV Distribution by Pick Number",
    max_pick: int | None = None,
    av_range: tuple[float, float] = (0.0, 100.0),
    n_av_points: int = 300,
    degeneracy_threshold: float = 1e6,
    export_path: str | Path | None = None,
) -> go.Figure:
    """Create an animated 2D figure cycling through pick numbers.

    Each frame shows the fitted skew-normal PDF for one pick plus a rug plot
    of the actual player AV values at that pick. Only picks with valid fits
    (``|a| <= degeneracy_threshold``) are included — typically ~49 of 329
    picks from the full-dataset ``skew_params.csv``.

    The y-axis (probability density) is fixed across all frames so the shape
    of each distribution is visually comparable.

    Args:
        skew_params_df: DataFrame with columns ``Pick``, ``a``, ``loc``,
            ``scale`` — the full ``skew_params.csv`` contents.
        player_av_df: DataFrame with columns ``Player``, ``Pick``,
            ``Draft Year``, ``Draft Team``, ``rookie_contract_av``.
        title: Chart title.
        max_pick: If set, restrict frames and player scatter to picks ≤ this value.
        av_range: ``(lo, hi)`` range for the AV axis. Kept fixed across frames.
        n_av_points: AV sample points for PDF evaluation.
        degeneracy_threshold: Picks with ``|a|`` above this are excluded.
        export_path: If provided, saves the figure as HTML. Must end with
            ``.html`` — animated figures cannot be exported as static images.

    Returns:
        Plotly Figure with animation frames and a pick-number slider.

    Raises:
        ValueError: If ``export_path`` is provided but does not end with ``.html``.
    """
    if export_path is not None and not str(export_path).endswith(".html"):
        raise ValueError(
            "Animated figures can only be exported as HTML. "
            f"Got: {export_path!r}. Use a path ending with '.html'."
        )

    valid_df = skew_params_df.filter(pl.col("a").abs() <= degeneracy_threshold)
    if max_pick is not None:
        valid_df = valid_df.filter(pl.col("Pick") <= max_pick)
    valid_df = valid_df.sort("Pick")
    valid_picks = valid_df["Pick"].to_list()
    rows_by_pick = {row["Pick"]: row for row in valid_df.iter_rows(named=True)}

    av_arr = np.linspace(av_range[0], av_range[1], n_av_points)

    # Pre-compute fixed y-axis range from peak PDF values across all valid picks
    max_pdf = max(
        float(skewnorm.pdf(av_arr, row["a"], row["loc"], row["scale"]).max())
        for row in rows_by_pick.values()
    )
    y_range = [-(max_pdf * 0.05), max_pdf * 1.1]

    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.7)

    def _build_traces(pick_num: int, a: float, loc: float, scale: float) -> list:
        pdf = skewnorm.pdf(av_arr, a, loc, scale)
        pick_players = player_av_df.filter(pl.col("Pick") == pick_num)
        curve = go.Scatter(
            x=av_arr.tolist(),
            y=pdf.tolist(),
            mode="lines",
            line=dict(color=_LINE_COLOR, width=2),
            name=f"Pick {pick_num} PDF",
        )
        rug = go.Scatter(
            x=pick_players["rookie_contract_av"].to_list(),
            y=[0.0] * len(pick_players),
            mode="markers",
            marker=dict(color=obs_color, size=8, symbol="line-ns-open"),
            name="Player AV",
            text=pick_players["Player"].to_list(),
            hovertemplate="%{text}: %{x:.1f} AV<extra></extra>",
        )
        return [curve, rug]

    frames = [
        go.Frame(
            data=_build_traces(
                p, rows_by_pick[p]["a"], rows_by_pick[p]["loc"], rows_by_pick[p]["scale"]
            ),
            name=str(p),
        )
        for p in valid_picks
    ]

    first_row = rows_by_pick[valid_picks[0]]
    initial_traces = _build_traces(
        valid_picks[0], first_row["a"], first_row["loc"], first_row["scale"]
    )
    fig = go.Figure(data=initial_traces, frames=frames)

    slider_steps = [
        dict(
            method="animate",
            args=[[str(p)], dict(mode="immediate", frame=dict(duration=400, redraw=True))],
            label=str(p),
        )
        for p in valid_picks
    ]

    fig.update_layout(
        title=title,
        xaxis=dict(title="Rookie Contract AV", range=list(av_range)),
        yaxis=dict(title="Probability Density", range=y_range),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.05,
                x=0,
                xanchor="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=400, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=150),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Pick: ", visible=True, xanchor="center"),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
    )

    if export_path is not None:
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))

    return fig


def plot_animated_skew_by_year(
    rolling_skew_df: pl.DataFrame,
    player_av_df: pl.DataFrame | None = None,
    title: str = "Skew-Normal AV Distributions by Pick — Rolling Window",
    max_pick: int | None = None,
    av_range: tuple[float, float] = (0.0, 100.0),
    n_av_points: int = 300,
    degeneracy_threshold: float = 1e6,
    window_half: int = 5,
    export_path: str | Path | None = None,
) -> go.Figure:
    """Create an animated 3D surface cycling through rolling-window center years.

    Each frame shows the skew-normal PDF surface for one center year from the
    rolling-window fit. The pick axis is the union of all valid picks across
    all years so the surface dimensions are constant across frames. Picks that
    are degenerate or absent in a given year appear as zero-density columns.

    Displays two layers per frame:
    1. ``go.Surface`` — PDF surface (X=pick, Y=AV, Z=density).
    2. ``go.Scatter3d`` — players drafted within the window at z=0 (optional).

    Args:
        rolling_skew_df: DataFrame with columns ``Pick``, ``a``, ``loc``,
            ``scale``, ``center_year`` — the full ``rolling_skew_params.parquet``
            contents.
        player_av_df: If provided, draws a scatter of players whose ``Draft Year``
            falls in ``[center_year - window_half, center_year + window_half]``
            for each frame. If ``None``, no scatter layer is drawn.
        title: Chart title.
        max_pick: If set, restrict surface and scatter to picks ≤ this value.
        av_range: ``(lo, hi)`` range for the AV axis. Fixed across all frames.
        n_av_points: AV sample points for PDF evaluation.
        degeneracy_threshold: Picks with ``|a|`` above this are excluded.
        window_half: Half-width of the rolling window used when filtering
            players per frame. Default 5, matching ``WINDOW_LENGTH=11``.
        export_path: If provided, saves the figure as HTML. Must end with
            ``.html`` — animated 3D figures cannot be exported as static images.

    Returns:
        Plotly Figure with animation frames and a center-year slider.

    Raises:
        ValueError: If ``export_path`` is provided but does not end with ``.html``.
    """
    if export_path is not None and not str(export_path).endswith(".html"):
        raise ValueError(
            "Animated figures can only be exported as HTML. "
            f"Got: {export_path!r}. Use a path ending with '.html'."
        )

    valid_rolling = rolling_skew_df.filter(pl.col("a").abs() <= degeneracy_threshold)
    if max_pick is not None:
        valid_rolling = valid_rolling.filter(pl.col("Pick") <= max_pick)

    all_valid_picks = sorted(valid_rolling["Pick"].unique().to_list())
    pick_to_idx = {p: i for i, p in enumerate(all_valid_picks)}
    center_years = sorted(rolling_skew_df["center_year"].unique().to_list())
    n_picks = len(all_valid_picks)

    av_arr = np.linspace(av_range[0], av_range[1], n_av_points)

    # Pre-compute global z_max so the z-axis stays fixed across frames
    global_zmax = 0.0
    for row in valid_rolling.iter_rows(named=True):
        peak = float(skewnorm.pdf(row["loc"], row["a"], row["loc"], row["scale"]))
        if peak > global_zmax:
            global_zmax = peak
    z_axis_max = global_zmax * 1.1

    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.35)

    def _build_surface_data(yr: int) -> list:
        yr_df = valid_rolling.filter(pl.col("center_year") == yr).sort("Pick")
        z_matrix = np.zeros((n_av_points, n_picks))
        for row in yr_df.iter_rows(named=True):
            col_idx = pick_to_idx[row["Pick"]]
            z_matrix[:, col_idx] = skewnorm.pdf(av_arr, row["a"], row["loc"], row["scale"])

        traces: list = [
            go.Surface(
                x=all_valid_picks,
                y=av_arr.tolist(),
                z=z_matrix.tolist(),
                colorscale="Viridis",
                opacity=0.85,
                showscale=True,
                colorbar=dict(title="Probability Density"),
                cmin=0.0,
                cmax=global_zmax,
                name=f"Year {yr}",
            )
        ]

        if player_av_df is not None:
            window_players = player_av_df.filter(
                (pl.col("Draft Year") >= yr - window_half)
                & (pl.col("Draft Year") <= yr + window_half)
                & pl.col("Pick").is_in(all_valid_picks)
            )
            if max_pick is not None:
                window_players = window_players.filter(pl.col("Pick") <= max_pick)
            if len(window_players) > 0:
                traces.append(
                    go.Scatter3d(
                        x=window_players["Pick"].to_list(),
                        y=window_players["rookie_contract_av"].to_list(),
                        z=[0.0] * len(window_players),
                        mode="markers",
                        marker=dict(size=2, color=obs_color, opacity=0.35),
                        name="Players in Window",
                        text=window_players["Player"].to_list(),
                        hovertemplate="%{text}<br>Pick %{x} — %{y:.1f} AV<extra></extra>",
                    )
                )

        return traces

    frames = [
        go.Frame(data=_build_surface_data(yr), name=str(yr))
        for yr in center_years
    ]

    initial_traces = _build_surface_data(center_years[0])
    fig = go.Figure(data=initial_traces, frames=frames)

    slider_steps = [
        dict(
            method="animate",
            args=[[str(yr)], dict(mode="immediate", frame=dict(duration=300, redraw=True))],
            label=str(yr),
        )
        for yr in center_years
    ]

    fig.update_layout(
        title=f"{title} — {center_years[0]}–{center_years[-1]}",
        scene=dict(
            xaxis=dict(title="Pick Number"),
            yaxis=dict(title="Rookie Contract AV", range=list(av_range)),
            zaxis=dict(title="Probability Density", range=[0, z_axis_max]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.05,
                x=0,
                xanchor="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=100),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Center Year: ", visible=True, xanchor="center"),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
    )

    if export_path is not None:
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))

    return fig
