"""Reader for VMAF JSON output files (ffmpeg libvmaf)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from ...domain.enums import ComparisonAxis, ResultCategory, ResultSource
from ...domain.models import (
    MetricRecord,
    PerFrameMetrics,
    QualityMetrics,
)


class VmafJsonReader:
    """Adapter that reads ``vmaf.json`` and ``vmaf_vs_gt.json`` files.

    These are produced by ffmpeg's libvmaf and contain per-frame VMAF
    scores plus sub-metrics (ADM, VIF, motion).  They live inside
    ``benchmark_results/<strategy>/`` directories.
    """

    @property
    def source_type(self) -> ResultSource:
        return ResultSource.VMAF_JSON

    def supports(self, path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name.lower()
        return name in ("vmaf.json", "vmaf_vs_gt.json")

    def read(self, path: Path) -> List[MetricRecord]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        record = self._parse(data, path)
        return [record]

    # ── Internals ───────────────────────────────────────────────

    @staticmethod
    def _detect_axis(path: Path) -> ComparisonAxis:
        """vmaf_vs_gt → END_TO_END, vmaf → COMPRESSION_FIDELITY."""
        if "vs_gt" in path.name.lower():
            return ComparisonAxis.END_TO_END
        return ComparisonAxis.COMPRESSION_FIDELITY

    @staticmethod
    def _detect_strategy(path: Path) -> str:
        """Strategy name from parent directory."""
        return path.parent.name

    @staticmethod
    def _make_id(strategy: str, axis: ComparisonAxis, path: Path) -> str:
        raw = f"vmaf_json:{strategy}:{axis.value}:{path.name}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _parse(self, data: dict, path: Path) -> MetricRecord:
        strategy = self._detect_strategy(path)
        axis = self._detect_axis(path)
        record_id = self._make_id(strategy, axis, path)

        # Per-frame VMAF
        frames = data.get("frames", [])
        frame_indices = []
        vmaf_scores: List[float] = []

        for fr in frames:
            frame_indices.append(fr.get("frameNum", 0))
            metrics = fr.get("metrics", {})
            vmaf_scores.append(metrics.get("vmaf", 0.0))

        per_frame = PerFrameMetrics(
            frame_indices=frame_indices,
            vmaf=vmaf_scores,
        )

        # Pooled mean
        pooled = data.get("pooled_metrics", {})
        vmaf_mean: Optional[float] = None
        if pooled:
            vmaf_block = pooled.get("vmaf", {})
            if isinstance(vmaf_block, dict):
                vmaf_mean = vmaf_block.get("mean")
            elif isinstance(vmaf_block, (int, float)):
                vmaf_mean = float(vmaf_block)

        quality = QualityMetrics(vmaf=vmaf_mean)

        # Determine category
        category = (
            ResultCategory.END_TO_END
            if axis == ComparisonAxis.END_TO_END
            else ResultCategory.COMPRESSION
        )

        # Distinguish the two VMAF comparisons in the display name
        if axis == ComparisonAxis.END_TO_END:
            display_suffix = "vmaf_vs_gt"
        else:
            display_suffix = "vmaf_vs_original"

        return MetricRecord(
            id=record_id,
            source=ResultSource.VMAF_JSON,
            category=category,
            name=f"{strategy}_{display_suffix}",
            tags={
                "strategy": strategy,
                "axis": axis.value,
                "source_path": str(path),
            },
            quality_axes={axis: quality},
            per_frame_axes={axis: per_frame},
            quality_metrics=quality,
            per_frame_metrics=per_frame,
        )
