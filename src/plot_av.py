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
    rolling_stats_dict: dict[int, pl.DataFrame],
    export_path: str | Path | None = None,
) -> go.Figure:
    """Create an animated figure cycling through rolling-window pick AV statistics.

    Each frame corresponds to one center year from ``rolling_stats_dict``.
    The animation shows how the mean rookie contract AV and IQR band change
    over time as the draft year window advances. A slider and Play button
    allow interactive scrubbing.

    The x-axis range is fixed to the global maximum pick number across all
    frames so the axis does not jump between frames.

    Input dict values — columns required per DataFrame:
        - ``Pick`` (Int64): Overall pick number.
        - ``mean`` (Float64): Mean rookie contract AV.
        - ``25%`` (Float64): 25th percentile; may be null for sparse picks.
        - ``75%`` (Float64): 75th percentile; may be null for sparse picks.

    Args:
        rolling_stats_dict: Mapping of ``center_year`` → stats DataFrame,
            output of :func:`annual_av_analysis.rolling_window_pick_stats`.
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

    center_years = sorted(rolling_stats_dict.keys())
    band_fill = _hex_to_rgba(_BAND_COLOR, 0.3)

    def _build_traces(df: pl.DataFrame) -> list[go.BaseTraceType]:
        mean_df = df.filter(pl.col("Pick") <= 250).select(["Pick", "mean"]).drop_nulls()
        iqr_df = df.filter(pl.col("Pick") <= 250).select(["Pick", "25%", "75%"]).drop_nulls()

        picks_iqr = iqr_df["Pick"].to_list()
        q25 = iqr_df["25%"].to_list()
        q75 = iqr_df["75%"].to_list()

        picks_mean = mean_df["Pick"].to_list()
        means = mean_df["mean"].to_list()

        band = go.Scatter(
            x=picks_iqr + picks_iqr[::-1],
            y=q75 + q25[::-1],
            fill="toself",
            fillcolor=band_fill,
            line=dict(color="rgba(0,0,0,0)"),
            name="25%–75% IQR",
            showlegend=True,
        )
        line = go.Scatter(
            x=picks_mean,
            y=means,
            mode="lines",
            line=dict(color=_LINE_COLOR, width=2),
            name="Mean Rookie Contract AV",
        )
        return [band, line]

    # Build frames
    frames = [
        go.Frame(data=_build_traces(rolling_stats_dict[yr]), name=str(yr))
        for yr in center_years
    ]

    # Initial display = first frame
    initial_traces = _build_traces(rolling_stats_dict[center_years[0]])

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
        title=f"Rookie Contract AV by Draft Pick — Rolling {center_years[0]}–{center_years[-1]}",
        xaxis=dict(title="Pick Number", range=[1, 250]),
        yaxis_title="Rookie Contract AV",
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
    3. The 1-sigma confidence band as a shaded teal region.

    Input ``fit_result`` keys required (output of
    :func:`annual_av_analysis.exponential_av_fit_means`):
        - ``picks`` (ndarray): Unique pick numbers from the stats data.
        - ``means`` (ndarray): Mean AV per pick, aligned with ``picks``.
        - ``x_fit`` (ndarray): Dense pick axis for the smooth fitted curve.
        - ``y_fit`` (ndarray): Fitted AV values at ``x_fit`` points.
        - ``y_upper`` (ndarray): Upper 1-sigma bound at ``x_fit`` points.
        - ``y_lower`` (ndarray): Lower 1-sigma bound at ``x_fit`` points.
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
    means = fit_result["means"]
    x_fit = fit_result["x_fit"]
    y_fit = fit_result["y_fit"]
    y_upper = fit_result["y_upper"]
    y_lower = fit_result["y_lower"]
    a, b, c = fit_result["popt"]

    band_fill = _hex_to_rgba(_BAND_COLOR, 0.25)
    obs_color = _hex_to_rgba(_VIRIDIS[3], 0.8)  # more opaque — one point per pick

    fig = go.Figure()

    # 1-sigma band
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

    # Per-pick mean scatter (one point per unique pick)
    fig.add_trace(
        go.Scatter(
            x=picks,
            y=means,
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
