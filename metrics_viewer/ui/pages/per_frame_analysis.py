"""Per-Frame Analysis page — frame-by-frame metric visualization."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from ...domain.enums import ComparisonAxis, ResultSource
from ...domain.models import MetricRecord
from ...domain.services import MetricsService
from ..components.filters import axis_selector
from ..components.per_frame_chart import per_frame_chart, per_frame_heatmap


def _get_records_with_perframe(service: MetricsService):
    """Return records that have per-frame data."""
    results = []
    for r in service.get_all_records():
        # Check all axes
        has_pf = r.per_frame_metrics is not None
        if not has_pf:
            for axis_pf in r.per_frame_axes.values():
                if axis_pf is not None:
                    has_pf = True
                    break
        if has_pf:
            results.append(r)
    return results


def build_per_frame_analysis(service: MetricsService) -> html.Div:
    """Construct the Per-Frame Analysis page."""
    records = _get_records_with_perframe(service)

    if not records:
        return html.Div([
            html.H4("Per-Frame Analysis", className="section-title"),
            html.P("No per-frame data available.", className="text-muted"),
        ])

    # Separate benchmark (with quality per-frame) vs VMAF-only
    benchmark_records = [r for r in records if r.source == ResultSource.BENCHMARK_JSON]
    vmaf_records = [r for r in records if r.source == ResultSource.VMAF_JSON]

    # Build metric selector
    available_metrics = set()
    for r in records:
        for pf in [r.per_frame_metrics] + list(r.per_frame_axes.values()):
            if pf is None:
                continue
            if pf.psnr:
                available_metrics.add("psnr")
            if pf.ssim:
                available_metrics.add("ssim")
            if pf.lpips:
                available_metrics.add("lpips")
            if pf.vmaf:
                available_metrics.add("vmaf")

    metric_options = [{"label": m.upper(), "value": m} for m in sorted(available_metrics)]

    # Strategy selector
    strategy_options = [{"label": r.display_name, "value": r.id} for r in records]

    # ── Charts ──────────────────────────────────────────────

    # PSNR per-frame for benchmark records (compression fidelity)
    charts = []

    if benchmark_records:
        charts.append(html.H5("Benchmark — Compression Fidelity", className="section-title mt-3"))
        charts.append(per_frame_chart(
            benchmark_records, "psnr",
            axis=ComparisonAxis.COMPRESSION_FIDELITY,
            chart_id="pf-benchmark-psnr",
            title="PSNR per Frame (Compression Fidelity)",
        ))

        charts.append(per_frame_chart(
            benchmark_records, "ssim",
            axis=ComparisonAxis.COMPRESSION_FIDELITY,
            chart_id="pf-benchmark-ssim",
            title="SSIM per Frame (Compression Fidelity)",
        ))

        charts.append(per_frame_chart(
            benchmark_records, "lpips",
            axis=ComparisonAxis.COMPRESSION_FIDELITY,
            chart_id="pf-benchmark-lpips",
            title="LPIPS per Frame (Compression Fidelity)",
        ))

        # End-to-end
        charts.append(html.H5("Benchmark — End-to-End", className="section-title mt-4"))
        charts.append(per_frame_chart(
            benchmark_records, "psnr",
            axis=ComparisonAxis.END_TO_END,
            chart_id="pf-e2e-psnr",
            title="PSNR per Frame (End-to-End)",
        ))

    # VMAF per-frame — split by axis (vs original / vs ground truth)
    if vmaf_records:
        vmaf_vs_original = [
            r for r in vmaf_records
            if r.tags.get("axis") == ComparisonAxis.COMPRESSION_FIDELITY.value
        ]
        vmaf_vs_gt = [
            r for r in vmaf_records
            if r.tags.get("axis") == ComparisonAxis.END_TO_END.value
        ]

        if vmaf_vs_original:
            charts.append(html.H5(
                "VMAF per Frame — vs Original Renders",
                className="section-title mt-4",
            ))
            charts.append(per_frame_chart(
                vmaf_vs_original, "vmaf",
                axis=ComparisonAxis.COMPRESSION_FIDELITY,
                chart_id="pf-vmaf-vs-orig-line",
                title="VMAF per Frame (Decompressed vs Original Renders)",
            ))
            charts.append(per_frame_heatmap(
                vmaf_vs_original, "vmaf",
                axis=ComparisonAxis.COMPRESSION_FIDELITY,
                chart_id="pf-vmaf-vs-orig-heatmap",
                title="VMAF Heatmap — vs Original Renders",
            ))

        if vmaf_vs_gt:
            charts.append(html.H5(
                "VMAF per Frame — vs Ground Truth",
                className="section-title mt-4",
            ))
            charts.append(per_frame_chart(
                vmaf_vs_gt, "vmaf",
                axis=ComparisonAxis.END_TO_END,
                chart_id="pf-vmaf-vs-gt-line",
                title="VMAF per Frame (Decompressed vs Ground Truth)",
            ))
            charts.append(per_frame_heatmap(
                vmaf_vs_gt, "vmaf",
                axis=ComparisonAxis.END_TO_END,
                chart_id="pf-vmaf-vs-gt-heatmap",
                title="VMAF Heatmap — vs Ground Truth",
            ))

    # Heatmap for benchmarks
    if benchmark_records:
        charts.append(html.H5("Heatmaps", className="section-title mt-4"))
        charts.append(per_frame_heatmap(
            benchmark_records, "psnr",
            axis=ComparisonAxis.COMPRESSION_FIDELITY,
            chart_id="pf-heatmap-psnr",
            title="PSNR Heatmap (Compression Fidelity)",
        ))

    return html.Div([
        html.H4("Per-Frame Analysis", className="section-title"),
        html.P("Frame-by-frame quality metric visualization. Compare how quality varies across frames for each strategy.",
               className="section-subtitle"),

        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Metric", htmlFor="pf-metric-select",
                           style={"fontSize": "0.8rem", "color": "#a0b4d0", "textTransform": "uppercase"}),
                dcc.Dropdown(
                    id="pf-metric-select",
                    options=metric_options,
                    value="psnr" if "psnr" in available_metrics else list(available_metrics)[0],
                    clearable=False,
                    style={"backgroundColor": "#0b1e3d", "color": "#e0e0e0"},
                ),
            ], md=3),
            dbc.Col([
                html.Label("Strategies", htmlFor="pf-strategy-select",
                           style={"fontSize": "0.8rem", "color": "#a0b4d0", "textTransform": "uppercase"}),
                dcc.Dropdown(
                    id="pf-strategy-select",
                    options=strategy_options,
                    value=[r.id for r in records[:6]],
                    multi=True,
                    placeholder="Select strategies\u2026",
                    style={"backgroundColor": "#0b1e3d", "color": "#e0e0e0"},
                ),
            ], md=5),
            dbc.Col(axis_selector(selector_id="pf-axis", default="compression_fidelity"), md=4),
        ], className="filter-panel mb-4 p-3"),

        # Charts
        html.Div(id="pf-charts-container", children=charts),
    ])
