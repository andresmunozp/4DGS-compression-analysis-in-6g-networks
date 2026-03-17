#!/usr/bin/env python3
"""
compress.py — Post-training 4DGS model compressor.

Loads a trained 4DGS model, applies a configurable compression pipeline,
and writes the compressed output as network-ready chunks.

Usage
-----
    python compress.py \
        --model_path output/dynerf/coffee_martini \
        --iteration 14000 \
        --config compression/configs/balanced.yaml \
        --output compressed_output/ \
        --chunk_size 1048576 \
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
import yaml

# ── Project imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compression.base import DeformationData, GaussianData
from compression.chunker import ModelChunker
from compression.pipeline import CompressionPipeline


def load_gaussian_data(model_path: str, iteration: int) -> GaussianData:
    """Load Gaussian parameters from PLY + deformation auxiliary files."""
    from plyfile import PlyData

    iter_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    ply_path = os.path.join(iter_dir, "point_cloud.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]

    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1)
    opacity = np.asarray(el["opacity"])[..., np.newaxis]

    # DC features
    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.asarray(el["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(el["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(el["f_dc_2"])

    # Rest features
    rest_names = sorted(
        [p.name for p in el.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if rest_names:
        features_extra = np.zeros((xyz.shape[0], len(rest_names)), dtype=np.float32)
        for idx, name in enumerate(rest_names):
            features_extra[:, idx] = np.asarray(el[name])
        n_coeffs = len(rest_names) // 3
        features_extra = features_extra.reshape(xyz.shape[0], 3, n_coeffs)
    else:
        features_extra = np.zeros((xyz.shape[0], 3, 0), dtype=np.float32)

    # Determine SH degree from coefficient count
    total_coeffs = 1 + (features_extra.shape[2] if features_extra.ndim == 3 else 0)
    import math
    sh_degree = int(math.sqrt(total_coeffs)) - 1

    # Transpose to match internal format: (N, K, 3)
    features_dc_t = np.ascontiguousarray(features_dc.transpose(0, 2, 1))  # (N, 1, 3)
    features_rest_t = np.ascontiguousarray(features_extra.transpose(0, 2, 1))  # (N, K, 3)

    # Scaling
    scale_names = sorted(
        [p.name for p in el.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scaling = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, name in enumerate(scale_names):
        scaling[:, idx] = np.asarray(el[name])

    # Rotation
    rot_names = sorted(
        [p.name for p in el.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rotation = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, name in enumerate(rot_names):
        rotation[:, idx] = np.asarray(el[name])

    # Auxiliary deformation data
    deformation_table = None
    deformation_accum = None
    dt_path = os.path.join(iter_dir, "deformation_table.pth")
    da_path = os.path.join(iter_dir, "deformation_accum.pth")
    if os.path.exists(dt_path):
        deformation_table = torch.load(dt_path, map_location="cpu", weights_only=False).numpy()
    if os.path.exists(da_path):
        deformation_accum = torch.load(da_path, map_location="cpu", weights_only=False).numpy()

    return GaussianData(
        xyz=xyz.astype(np.float32),
        features_dc=features_dc_t.astype(np.float32),
        features_rest=features_rest_t.astype(np.float32),
        opacity=opacity.astype(np.float32),
        scaling=scaling.astype(np.float32),
        rotation=rotation.astype(np.float32),
        sh_degree=sh_degree,
        active_sh_degree=sh_degree,
        deformation_table=deformation_table,
        deformation_accum=deformation_accum,
    )


def load_deformation_data(model_path: str, iteration: int) -> DeformationData:
    """Load the deformation network state dict."""
    iter_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    deform_path = os.path.join(iter_dir, "deformation.pth")

    if not os.path.exists(deform_path):
        raise FileNotFoundError(f"Deformation weights not found: {deform_path}")

    state_dict = torch.load(deform_path, map_location="cpu", weights_only=False)
    return DeformationData(state_dict=state_dict)


def main():
    parser = ArgumentParser(description="Compress a trained 4DGS model")
    parser.add_argument("--model_path", required=True, help="Path to trained model output dir")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load (-1 = latest)")
    parser.add_argument("--config", required=True, help="YAML compression config file")
    parser.add_argument("--output", default="compressed_output", help="Output directory for chunks")
    parser.add_argument("--chunk_size", type=int, default=1_048_576, help="Max bytes per chunk")
    parser.add_argument("--no_chunks", action="store_true", help="Write single .4dgs file instead of chunks")
    args = parser.parse_args()

    # ── Find iteration ────────────────────────────────────────────────
    if args.iteration == -1:
        from utils.system_utils import searchForMaxIteration
        args.iteration = searchForMaxIteration(
            os.path.join(args.model_path, "point_cloud")
        )
    print(f"Loading model from {args.model_path} at iteration {args.iteration}")

    # ── Load data ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    gaussian = load_gaussian_data(args.model_path, args.iteration)
    deformation = load_deformation_data(args.model_path, args.iteration)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s  |  {gaussian.num_gaussians} Gaussians")

    # ── Load pipeline config ──────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pipeline = CompressionPipeline.from_config(config)

    # ── Compress ──────────────────────────────────────────────────────
    original_g_bytes = gaussian.total_bytes
    original_d_bytes = deformation.total_bytes
    original_total = original_g_bytes + original_d_bytes

    print(f"Original size: {original_total / 1e6:.2f} MB "
          f"(Gaussians: {original_g_bytes / 1e6:.2f} MB, "
          f"Deformation: {original_d_bytes / 1e6:.2f} MB)")

    t0 = time.perf_counter()
    archive = pipeline.compress_to_archive(gaussian, deformation)
    compress_time = time.perf_counter() - t0

    compressed_size = len(archive)
    ratio = original_total / compressed_size if compressed_size > 0 else 0

    print(f"Compressed:    {compressed_size / 1e6:.2f} MB  "
          f"(ratio {ratio:.2f}x, savings {(1 - compressed_size / original_total) * 100:.1f}%)")
    print(f"Compression time: {compress_time:.3f}s")

    pipeline.print_stats()

    # ── Output ────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    if args.no_chunks:
        archive_path = os.path.join(args.output, "model.4dgs")
        with open(archive_path, "wb") as f:
            f.write(archive)
        print(f"Archive written to {archive_path}")
    else:
        chunker = ModelChunker(chunk_size=args.chunk_size)
        paths = chunker.split_and_write(archive, args.output)
        print(f"Written {len(paths)} chunks to {args.output}/")

    # ── Write compression report ──────────────────────────────────────
    report = {
        "model_path": args.model_path,
        "iteration": args.iteration,
        "config_file": args.config,
        "pipeline_config": config,
        "original_size_bytes": original_total,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": round(ratio, 4),
        "savings_pct": round((1 - compressed_size / original_total) * 100, 2),
        "compression_time_s": round(compress_time, 4),
        "num_gaussians": gaussian.num_gaussians,
        "num_chunks": 1 if args.no_chunks else len(paths),
        "chunk_size": args.chunk_size,
    }
    report_path = os.path.join(args.output, "compression_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
