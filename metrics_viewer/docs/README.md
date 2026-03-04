# 4DGaussians Metrics Viewer

Interactive dashboard for visualizing, comparing, and analyzing metrics from the 4DGaussians pipeline — benchmarks, training evaluations, video quality (VMAF), and streaming QoE.

## Quick Start

```bash
# From the project root
python -m metrics_viewer.app

# Custom options
python -m metrics_viewer.app --data-dir . --port 8050 --refresh-interval 30
```

Then open **http://localhost:8050** in your browser.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `.` | Project root directory to scan for data |
| `--host` | `127.0.0.1` | Server bind address |
| `--port` | `8050` | Server port |
| `--no-debug` | — | Disable Dash debug/hot-reload mode |
| `--refresh-interval` | `30` | Auto-refresh check interval in seconds (0 = disabled) |

## Architecture

Hexagonal (Ports & Adapters) — the domain core has zero I/O dependencies:

```
Frontend (Dash/Plotly)
    ↓ driving port
Domain Core (MetricsService, MetricRecord, enums)
    ↓ driven ports (IDataSourceReader, IResultExporter)
Adapters (BenchmarkJsonReader, CsvExporter, etc.)
```

### Adding a new data format

1. Create a reader in `adapters/readers/` implementing `IDataSourceReader`
2. Register it in `app.py:build_service()`
3. The UI auto-discovers new records — no UI changes needed

### Adding a new page

1. Create `ui/pages/my_page.py` with a `build_my_page(service)` function
2. Register the nav item in `ui/layout.py:NAV_ITEMS`
3. Add callbacks in `ui/callbacks.py`

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Overview | `/` | Summary cards, filters, full comparison table |
| Benchmark Compare | `/benchmark` | Bar/radar/scatter charts, Pareto frontier, axis toggle |
| Per-Frame Analysis | `/per-frame` | Frame-by-frame line charts + VMAF heatmaps |
| Training Eval | `/training` | Quality vs iteration, delta improvement, timing |
| Streaming QoE | `/streaming` | QoE gauges, buffering metrics, throughput charts |
| Compression Detail | `/compression` | Pipeline waterfall, per-axis quality, model info |

## Data Sources

The viewer auto-discovers these files:

| File | Reader | What it provides |
|------|--------|-----------------|
| `benchmark_results/benchmark_results.json` | BenchmarkJsonReader | 3-axis quality, compression, timing, per-frame, pipeline |
| `benchmark_results/benchmark_summary.csv` | BenchmarkCsvReader | Summary quality metrics (fallback) |
| `results_json/*.json` | TrainingJsonReader | Training quality, timing, model info |
| `benchmark_results/*/vmaf*.json` | VmafJsonReader | Per-frame VMAF scores |

## Export

Every page supports CSV and PNG export via the export buttons. Data is exported using the domain's `IResultExporter` protocol.

## Accessibility

The UI follows Vercel Web Interface Guidelines:

- Skip-to-content link
- All form controls have associated `<label>` elements with `htmlFor`
- Icon-only buttons have `aria-label`
- Decorative icons use `aria-hidden="true"`
- Visible `:focus-visible` ring on all interactive elements
- `prefers-reduced-motion` respected — animations disabled
- `aria-live` region for async data refresh status
- Semantic HTML: `<nav>`, `<main>`, `<button>`, `<a>`
- Dark mode: `color-scheme: dark` + `meta theme-color`
- `font-variant-numeric: tabular-nums` on metric values
- `text-wrap: balance` on headings

## Dependencies

```
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=5.18.0
pandas>=2.0.0
```

## Project Structure

```
viewers/metrics_viewer/
├── app.py                    # Entry point + CLI
├── config.py                 # ViewerConfig
├── domain/                   # Domain core (no I/O deps)
│   ├── models.py             # MetricRecord, value objects
│   ├── enums.py              # MetricType, ResultSource, etc.
│   ├── ports.py              # IDataSourceReader, IResultExporter
│   └── services.py           # MetricsService
├── adapters/
│   ├── readers/              # JSON/CSV readers + directory scanner
│   └── exporters/            # CSV + PNG exporters
├── ui/
│   ├── layout.py             # Dash app + sidebar + routing
│   ├── callbacks.py          # All dynamic callbacks
│   ├── components/           # Reusable: metric_card, bar_chart, etc.
│   └── pages/                # 6 page modules
├── assets/styles.css         # Dark theme CSS
└── tests/test_integration.py # 27 integration tests
```
