#!/usr/bin/env python3
"""
benchmark_compression.py — Compare compression strategies on a trained 4DGS model.

Evaluates each compression config by:
  1. Compressing the model
  2. Decompressing it
  3. Rendering test frames from both original and reconstructed models
  4. Computing quality metrics (PSNR, SSIM, LPIPS) per frame:
     a) **Compression fidelity**: decompressed renders vs original-model renders
     b) **End-to-end quality**: decompressed renders vs dataset ground-truth images
     c) **Training baseline**: original-model renders vs dataset GT (computed once)
  5. Generating mp4 videos and computing VMAF (vs original renders AND vs GT)
  6. Recording compression ratio, sizes, and decode/export timing
  7. Computing streaming QoE metrics (startup delay, rebuffer events, stalls)

The --gt_dir option specifies a directory containing ground-truth images named
``00000.png``, ``00001.png``, … (the format produced by ``render.py``).  When
omitted, it defaults to ``<model_path>/test/ours_<iteration>/gt/``.

Usage
-----
    python benchmark_compression.py \
        --model_path output/dynerf/coffee_martini_sirvio \
        --iteration 14000 \
        --source_path data/dynerf/coffee_martini_funciono \
        --configs arguments/dynerf/coffee_martini.py \
        --compression_configs \
            compression/configs/lossless.yaml \
            compression/configs/balanced.yaml \
            compression/configs/aggressive.yaml \
        --output_dir benchmark_results/ \
        --num_frames 50 \
        --bandwidth_mbps 1000
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compression.base import DeformationData, GaussianData
from compression.chunker import ModelChunker
from compression.pipeline import CompressionPipeline


# ── Quality metrics ───────────────────────────────────────────────────────

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    from utils.loss_utils import ssim
    return ssim(img1.unsqueeze(0), img2.unsqueeze(0)).item()


def compute_lpips_metric(img1: torch.Tensor, img2: torch.Tensor, net_type: str = "vgg") -> float:
    from lpipsPyTorch import lpips
    return lpips(img1.unsqueeze(0), img2.unsqueeze(0), net_type=net_type).item()


def compute_vmaf(
    reference_video: str,
    distorted_video: str,
    output_json: Optional[str] = None,
) -> Optional[float]:
    """Compute VMAF using ffmpeg.  Returns the mean VMAF score or None on failure."""
    cmd = [
        "ffmpeg", "-i", distorted_video, "-i", reference_video,
        "-lavfi", f"libvmaf=log_fmt=json" +
                  (f":log_path={output_json}" if output_json else ""),
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        # Parse VMAF from log file
        if output_json and os.path.exists(output_json):
            with open(output_json, "r") as f:
                vmaf_data = json.load(f)
            return vmaf_data.get("pooled_metrics", {}).get("vmaf", {}).get("mean")
        # Try parsing from stderr
        for line in result.stderr.split("\n"):
            if "VMAF score" in line:
                parts = line.split(":")
                return float(parts[-1].strip())
    except Exception as e:
        print(f"  VMAF computation failed: {e}")
    return None


# ── Ground-truth image loader ─────────────────────────────────────────────

def load_gt_frames(
    gt_dir: str,
    num_frames: int,
) -> List[torch.Tensor]:
    """Load ground-truth PNG images from *gt_dir* as float32 [C,H,W] tensors.

    Expects files named ``00000.png``, ``00001.png``, …
    Returns at most *num_frames* tensors.  Prints a warning if the requested
    count exceeds the available images.
    """
    import torchvision

    frames: List[torch.Tensor] = []
    for idx in range(num_frames):
        path = os.path.join(gt_dir, f"{idx:05d}.png")
        if not os.path.exists(path):
            if idx == 0:
                print(f"  WARNING: GT directory has no images ({gt_dir})")
            elif idx < num_frames:
                print(f"  WARNING: only {idx} GT images found in {gt_dir} "
                      f"(requested {num_frames})")
            break
        img = torchvision.io.read_image(path).float().div(255.0).cuda()
        frames.append(img)
    return frames


# ── Streaming QoE metrics ────────────────────────────────────────────────

def compute_streaming_qoe(
    chunk_sizes: List[int],
    decode_time_s: float,
    export_time_per_frame_s: float,
    num_frames: int,
    bandwidth_mbps: float,
    target_fps: float = 30.0,
) -> Dict[str, Any]:
    """Compute streaming Quality of Experience (QoE) metrics.

    Simulates transmission of chunks over a link with *bandwidth_mbps* and
    computes startup delay, rebuffer events, stall duration, and overall
    QoE score.

    Parameters
    ----------
    chunk_sizes : list[int]
        Sizes of each chunk in bytes.
    decode_time_s : float
        Time to decode (decompress) the full archive.
    export_time_per_frame_s : float
        Average time to deform + write a single PLY frame.
    num_frames : int
        Total frames to export.
    bandwidth_mbps : float
        Simulated link bandwidth in Mbps.
    target_fps : float
        Target playback frame rate.

    Returns
    -------
    dict with QoE metrics.
    """
    bandwidth_bps = bandwidth_mbps * 1e6  # bits per second
    bandwidth_Bps = bandwidth_bps / 8     # bytes per second

    # ── Chunk transmission times ──
    chunk_arrival_times = []
    cumulative_t = 0.0
    for size in chunk_sizes:
        tx_time = size / bandwidth_Bps
        cumulative_t += tx_time
        chunk_arrival_times.append(cumulative_t)

    total_transmission_time = cumulative_t
    total_payload = sum(chunk_sizes)

    # ── Startup delay ──
    # Time until first chunk arrives + decode time (approximate: decode
    # can't start until all chunks arrive for the single-archive model)
    startup_delay = total_transmission_time + decode_time_s
    

    # ── Rebuffer analysis ──
    # After decode, frames are exported sequentially.  A rebuffer happens
    # when the renderer (SuperSplat at target_fps) consumes frames faster
    # than they are produced.
    frame_interval = 1.0 / target_fps
    rebuffer_events = 0
    total_stall_duration = 0.0
    playback_time = 0.0
    production_time = startup_delay  # decoding is complete, export starts

    for frame in range(num_frames):
        production_time += export_time_per_frame_s
        playback_time = startup_delay + frame * frame_interval

        if production_time > playback_time:
            stall = production_time - playback_time
            if stall > frame_interval * 0.1:  # threshold: >10% of frame interval
                rebuffer_events += 1
                total_stall_duration += stall

    # ── End-to-end latency ──
    e2e_latency = total_transmission_time + decode_time_s + (
        num_frames * export_time_per_frame_s
    )

    # ── Effective throughput ──
    effective_throughput_Bps = total_payload / total_transmission_time if total_transmission_time > 0 else 0

    # ── QoE score (ITU-T P.1203 inspired, simplified) ──
    # Higher is better; penalises startup delay, rebuffers, and stalls
    qoe_base = 5.0
    qoe_startup_penalty = min(startup_delay / 10.0, 2.0)  # up to -2 for >10s startup
    qoe_rebuffer_penalty = min(rebuffer_events * 0.3, 2.0)  # up to -2 for many rebuffers
    qoe_stall_penalty = min(total_stall_duration / 5.0, 1.0)  # up to -1 for >5s stalls
    qoe_score = max(1.0, qoe_base - qoe_startup_penalty - qoe_rebuffer_penalty - qoe_stall_penalty)

    return {
        "total_payload_bytes": total_payload,
        "num_chunks": len(chunk_sizes),
        "bandwidth_mbps": bandwidth_mbps,
        "total_transmission_time_s": round(total_transmission_time, 4),
        "startup_delay_s": round(startup_delay, 4),
        "rebuffer_events": rebuffer_events,
        "total_stall_duration_s": round(total_stall_duration, 4),
        "e2e_latency_s": round(e2e_latency, 4),
        "effective_throughput_MBps": round(effective_throughput_Bps / 1e6, 4),
        "qoe_score": round(qoe_score, 2),
        "target_fps": target_fps,
        "export_time_per_frame_s": round(export_time_per_frame_s, 6),
    }


# ── Rendering helpers ────────────────────────────────────────────────────

def render_frames_from_model(
    model_path: str,
    iteration: int,
    source_path: str,
    configs: Optional[str],
    output_dir: str,
    num_frames: int,
) -> Tuple[List[str], Any, str]:
    """Render reference frames using the original (uncompressed) model.

    Returns (list of rendered image paths, test_cameras, dataset_type).
    """
    from argparse import Namespace
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render as gs_render
    import torchvision

    # Build args & infer hyperparams from checkpoint for safety
    hyper = _load_hyperparams_for_render(configs)
    deform_pth = os.path.join(
        model_path, "point_cloud", f"iteration_{iteration}", "deformation.pth"
    )
    if os.path.exists(deform_pth):
        sd = torch.load(deform_pth, map_location="cpu")
        _infer_hyperparams_from_state_dict(sd, hyper)
        del sd

    render_args = Namespace(
        source_path=source_path,
        model_path=model_path,
        sh_degree=3,
        images="images",
        resolution=-1,
        white_background=True,
        eval=True,
        add_points=False,
        extension=".png",
        llffhold=8,
    )

    with torch.no_grad():
        gaussians = GaussianModel(render_args.sh_degree, hyper)
        scene = Scene(render_args, gaussians, load_iteration=iteration, shuffle=False)

        bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        cam_type = scene.dataset_type
        test_cams = scene.getTestCameras()

        os.makedirs(output_dir, exist_ok=True)
        paths = []
        n = min(num_frames, len(test_cams))

        for idx in range(n):
            cam = test_cams[idx]
            rendering = gs_render(cam, gaussians, _make_pipe(), bg, stage="fine", cam_type=cam_type)["render"]
            img_path = os.path.join(output_dir, f"{idx:05d}.png")
            torchvision.utils.save_image(rendering, img_path)
            paths.append(img_path)

    return paths, test_cams, cam_type


def render_frames_from_decompressed(
    gaussian: GaussianData,
    deformation: DeformationData,
    configs: Optional[str],
    output_dir: str,
    num_frames: int,
    test_cameras=None,
    cam_type: str = "dynerf",
) -> Tuple[List[str], float]:
    """Render frames from decompressed model using the gaussian rasteriser.

    When *test_cameras* are provided, renders actual images via ``gs_render``
    so that per-frame quality metrics (PSNR / SSIM / LPIPS) can be computed
    against the reference frames.  Falls back to PLY export when cameras are
    not available.

    Returns (list of rendered image paths, render_time).
    """
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render as gs_render
    import torchvision

    hyper = _load_hyperparams_for_render(configs)
    _infer_hyperparams_from_state_dict(deformation.state_dict, hyper)

    # ── Build a GaussianModel populated with decompressed data ──
    N = gaussian.num_gaussians
    gaussians = GaussianModel(gaussian.sh_degree, hyper)

    gaussians._xyz = torch.nn.Parameter(
        torch.tensor(gaussian.xyz, dtype=torch.float32, device="cuda"))
    gaussians._features_dc = torch.nn.Parameter(
        torch.tensor(gaussian.features_dc, dtype=torch.float32, device="cuda"))
    gaussians._features_rest = torch.nn.Parameter(
        torch.tensor(gaussian.features_rest, dtype=torch.float32, device="cuda"))
    gaussians._scaling = torch.nn.Parameter(
        torch.tensor(gaussian.scaling, dtype=torch.float32, device="cuda"))
    gaussians._rotation = torch.nn.Parameter(
        torch.tensor(gaussian.rotation, dtype=torch.float32, device="cuda"))
    gaussians._opacity = torch.nn.Parameter(
        torch.tensor(gaussian.opacity, dtype=torch.float32, device="cuda"))
    gaussians.active_sh_degree = gaussian.active_sh_degree
    gaussians._deformation_table = torch.ones(N, device="cuda").bool()
    gaussians.max_radii2D = torch.zeros(N, device="cuda")

    # Load deformation network weights (AABB is stored in the state_dict)
    gaussians._deformation.load_state_dict(deformation.state_dict)
    gaussians._deformation = gaussians._deformation.to("cuda")
    gaussians._deformation.eval()

    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    pipe = _make_pipe()

    t_render_start = time.perf_counter()
    with torch.no_grad():
        if test_cameras is not None and len(test_cameras) > 0:
            # ── Render actual images with the gaussian rasteriser ──
            n = min(num_frames, len(test_cameras))
            for idx in range(n):
                cam = test_cameras[idx]
                rendering = gs_render(
                    cam, gaussians, pipe, bg, stage="fine", cam_type=cam_type
                )["render"]
                img_path = os.path.join(output_dir, f"{idx:05d}.png")
                torchvision.utils.save_image(rendering, img_path)
                paths.append(img_path)
        else:
            # ── Fallback: export PLY files (no cameras available) ──
            from decompress import _write_ply
            means3D = gaussians._xyz
            scaling = gaussians._scaling
            rotation = gaussians._rotation
            opacity = gaussians._opacity
            shs = gaussians.get_features
            n_dc = gaussian.features_dc.shape[1]
            for frame_idx in range(num_frames):
                t = frame_idx / max(num_frames - 1, 1)
                time_tensor = torch.tensor([[t]], device="cuda").expand(N, 1)
                means3D_d, scales_d, rots_d, opacity_d, shs_d = gaussians._deformation(
                    means3D, scaling, rotation, opacity, shs, time_tensor
                )
                ply_path = os.path.join(output_dir, f"time_{frame_idx:05d}.ply")
                _write_ply(
                    ply_path,
                    means3D_d.cpu().numpy(),
                    shs_d[:, :n_dc, :].cpu().numpy(),
                    shs_d[:, n_dc:, :].cpu().numpy(),
                    opacity.detach().cpu().numpy(),
                    scales_d.cpu().numpy(),
                    rots_d.cpu().numpy(),
                )
                paths.append(ply_path)

    t_render = time.perf_counter() - t_render_start
    return paths, t_render


def _frames_to_video(frame_dir: str, output_path: str, fps: int = 30) -> bool:
    """Convert sequentially named PNG frames to mp4 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        return True
    except Exception as e:
        print(f"  Video encoding failed: {e}")
        return False


# ── Config helpers ────────────────────────────────────────────────────────

def _infer_hyperparams_from_state_dict(state_dict: dict, hyper_args) -> None:
    """Override hyper_args with values inferred directly from state_dict shapes."""
    import re as _re

    sd = state_dict

    # net_width
    key = "deformation_net.feature_out.0.weight"
    if key in sd:
        inferred_w = sd[key].shape[0]
        if inferred_w != hyper_args.net_width:
            print(f"  [infer] net_width: {hyper_args.net_width} → {inferred_w}")
            hyper_args.net_width = inferred_w

    # defor_depth
    feat_out_keys = sorted(
        [k for k in sd if k.startswith("deformation_net.feature_out.") and k.endswith(".weight")]
    )
    n_linears = len(feat_out_keys)
    if n_linears > 1:
        inferred_depth = n_linears
        if inferred_depth != hyper_args.defor_depth:
            print(f"  [infer] defor_depth: {hyper_args.defor_depth} → {inferred_depth}")
            hyper_args.defor_depth = inferred_depth

    # time resolution
    time_res = None
    for k, v in sd.items():
        m = _re.match(r"deformation_net\.grid\.grids\.(\d+)\.(\d+)", k)
        if m and int(m.group(2)) in (2, 4, 5):
            time_res = v.shape[2]
            break
    if time_res is not None:
        current_res = hyper_args.kplanes_config.get("resolution", [64, 64, 64, 25])
        if current_res[3] != time_res:
            print(f"  [infer] kplanes resolution time: {current_res[3]} → {time_res}")
            current_res[3] = time_res
            hyper_args.kplanes_config["resolution"] = current_res

    # multires
    res_indices = set()
    for k in sd:
        m = _re.match(r"deformation_net\.grid\.grids\.(\d+)\.", k)
        if m:
            res_indices.add(int(m.group(1)))
    if res_indices:
        n_res = len(res_indices)
        if n_res != len(hyper_args.multires):
            inferred_multires = list(range(1, n_res + 1))
            print(f"  [infer] multires: {hyper_args.multires} → {inferred_multires}")
            hyper_args.multires = inferred_multires

    # no_do, no_ds, no_dr, no_dshs
    has_opacity = any("opacity_deform" in k for k in sd)
    has_scales = any("scales_deform" in k for k in sd)
    has_rotations = any("rotations_deform" in k for k in sd)
    has_shs = any("shs_deform" in k for k in sd)
    for attr, has_head, label in [
        ("no_do", not has_opacity, "opacity"),
        ("no_ds", not has_scales, "scales"),
        ("no_dr", not has_rotations, "rotations"),
        ("no_dshs", not has_shs, "shs"),
    ]:
        current = getattr(hyper_args, attr, False)
        if current != (not has_head):
            inferred = not has_head
            print(f"  [infer] {attr}: {current} → {inferred}")
            setattr(hyper_args, attr, inferred)


def _read_py_config(config_path: str) -> dict:
    """Read a .py config file with _base_ inheritance (no mmcv needed)."""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        return {}
    ns: dict = {}
    with open(config_path, "r") as f:
        code = f.read()
    base_ns: dict = {}
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("_base_"):
            try:
                _, rhs = stripped.split("=", 1)
                base_rel = rhs.strip().strip("'\"")
                base_path = os.path.normpath(
                    os.path.join(os.path.dirname(config_path), base_rel)
                )
                base_ns = _read_py_config(base_path)
            except Exception:
                pass
            break
    try:
        exec(compile(code, config_path, "exec"), ns)
    except Exception:
        return base_ns
    merged: dict = {}
    for key in ("ModelHiddenParams", "OptimizationParams"):
        base_dict = base_ns.get(key, {})
        cur_dict = ns.get(key, {})
        if isinstance(base_dict, dict) and isinstance(cur_dict, dict):
            merged[key] = {**base_dict, **cur_dict}
        elif cur_dict:
            merged[key] = cur_dict
        elif base_dict:
            merged[key] = base_dict
    return merged


def _load_hyperparams_for_render(configs_path: Optional[str]):
    from argparse import Namespace

    defaults = Namespace(
        net_width=64, timebase_pe=4, defor_depth=1, posebase_pe=10,
        scale_rotation_pe=2, opacity_pe=2, timenet_width=64,
        timenet_output=32, bounds=1.6, plane_tv_weight=0.0001,
        time_smoothness_weight=0.01, l1_time_planes=0.0001,
        kplanes_config={"grid_dimensions": 2, "input_coordinate_dim": 4,
                        "output_coordinate_dim": 16, "resolution": [64, 64, 64, 25]},
        multires=[1, 2], no_grid=False, no_dx=False, no_ds=False,
        no_dr=False, no_do=False, no_dshs=False, apply_rotation=False,
        empty_voxel=False, grid_pe=0, static_mlp=False,
    )
    if configs_path:
        loaded = False
        try:
            import mmcv
            config = mmcv.Config.fromfile(configs_path)
            # Config files define ModelHiddenParams as a top-level dict;
            # extract from it so we set the actual hyperparams on defaults.
            model_params = config.get("ModelHiddenParams", None)
            if model_params and isinstance(model_params, dict):
                for k, v in model_params.items():
                    setattr(defaults, k, v)
            else:
                # Fallback: try flat attribute matching
                for k, v in config.items():
                    if hasattr(defaults, k):
                        setattr(defaults, k, v)
            loaded = True
        except Exception:
            pass
        if not loaded:
            try:
                cfg = _read_py_config(configs_path)
                for k, v in cfg.get("ModelHiddenParams", {}).items():
                    setattr(defaults, k, v)
            except Exception:
                pass
    return defaults


def _make_pipe():
    from argparse import Namespace
    return Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)


# ── Main benchmark ────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="Benchmark 4DGS compression strategies")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--configs", type=str, default=None,
                        help="Model config (e.g. arguments/dynerf/default.py)")
    parser.add_argument("--compression_configs", nargs="+", required=True,
                        help="List of YAML compression configs to compare")
    parser.add_argument("--output_dir", default="benchmark_results")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--bandwidth_mbps", type=float, default=10.0,
                        help="Simulated link bandwidth for QoE (Mbps)")
    parser.add_argument("--chunk_size", type=int, default=1_048_576)
    parser.add_argument("--skip_vmaf", action="store_true")
    parser.add_argument("--skip_render", action="store_true",
                        help="Skip rendering, only compute compression metrics")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Directory with GT images (00000.png …). "
                             "Defaults to <model_path>/test/ours_<iteration>/gt/")
    args = parser.parse_args()

    # Resolve iteration
    if args.iteration == -1:
        from utils.system_utils import searchForMaxIteration
        args.iteration = searchForMaxIteration(
            os.path.join(args.model_path, "point_cloud")
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load original model data
    from compress import load_gaussian_data, load_deformation_data

    print("Loading original model...")
    orig_gaussian = load_gaussian_data(args.model_path, args.iteration)
    orig_deformation = load_deformation_data(args.model_path, args.iteration)
    orig_total_bytes = orig_gaussian.total_bytes + orig_deformation.total_bytes

    print(f"Original: {orig_gaussian.num_gaussians} Gaussians, "
          f"{orig_total_bytes / 1e6:.2f} MB\n")

    # ── Pre-render reference frames & obtain test cameras ─────────────
    ref_dir = os.path.join(args.output_dir, "reference")
    ref_paths: List[str] = []
    test_cameras = None
    cam_type = "dynerf"

    if not args.skip_render:
        print("Rendering reference frames (original model)...")
        try:
            ref_paths, test_cameras, cam_type = render_frames_from_model(
                args.model_path, args.iteration, args.source_path,
                args.configs, ref_dir, args.num_frames,
            )
            print(f"  Rendered {len(ref_paths)} reference frames\n")
        except Exception as e:
            print(f"  Warning: reference rendering failed: {e}\n")

    # ── Load ground-truth images ──────────────────────────────────────
    gt_dir = args.gt_dir or os.path.join(
        args.model_path, "test", f"ours_{args.iteration}", "gt"
    )
    gt_frames: List[torch.Tensor] = []
    gt_available = False
    if os.path.isdir(gt_dir):
        print(f"Loading GT images from {gt_dir} ...")
        gt_frames = load_gt_frames(gt_dir, args.num_frames)
        gt_available = len(gt_frames) > 0
        print(f"  Loaded {len(gt_frames)} GT frames\n")
    else:
        print(f"  GT directory not found: {gt_dir}  — skipping GT comparisons\n")

    # ── Baseline: original model renders vs ground truth ──────────────
    baseline_vs_gt: Dict[str, Any] = {}
    if gt_available and ref_paths:
        import torchvision
        print("Computing baseline metrics (original model vs GT)...")
        b_psnr, b_ssim, b_lpips = [], [], []
        n_bl = min(len(ref_paths), len(gt_frames))
        for idx in range(n_bl):
            ref_img = torchvision.io.read_image(ref_paths[idx]).float().div(255.0).cuda()
            gt_img = gt_frames[idx]
            b_psnr.append(compute_psnr(ref_img, gt_img))
            b_ssim.append(compute_ssim(ref_img, gt_img))
            b_lpips.append(compute_lpips_metric(ref_img, gt_img))
        baseline_vs_gt = {
            "psnr_mean": round(float(np.mean(b_psnr)), 4),
            "ssim_mean": round(float(np.mean(b_ssim)), 4),
            "lpips_mean": round(float(np.mean(b_lpips)), 4),
            "psnr_per_frame": [round(v, 4) for v in b_psnr],
            "ssim_per_frame": [round(v, 4) for v in b_ssim],
            "lpips_per_frame": [round(v, 4) for v in b_lpips],
        }
        print(f"    PSNR:  {baseline_vs_gt['psnr_mean']:.2f} dB")
        print(f"    SSIM:  {baseline_vs_gt['ssim_mean']:.4f}")
        print(f"    LPIPS: {baseline_vs_gt['lpips_mean']:.4f}\n")

    results: List[Dict[str, Any]] = []

    for config_path in args.compression_configs:
        config_name = Path(config_path).stem
        print("=" * 70)
        print(f"BENCHMARK: {config_name}")
        print("=" * 70)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # ── Compress ──
        gaussian_copy = orig_gaussian.copy()
        deformation_copy = DeformationData(
            state_dict={k: v.clone() if hasattr(v, "clone") else v
                        for k, v in orig_deformation.state_dict.items()}
        )

        pipeline = CompressionPipeline.from_config(config)

        t0 = time.perf_counter()
        archive = pipeline.compress_to_archive(gaussian_copy, deformation_copy)
        t_compress = time.perf_counter() - t0

        compressed_bytes = len(archive)
        ratio = orig_total_bytes / compressed_bytes if compressed_bytes > 0 else 0

        print(f"  Compressed: {compressed_bytes / 1e6:.2f} MB "
              f"(ratio {ratio:.2f}x, {(1 - compressed_bytes / orig_total_bytes) * 100:.1f}% savings)")
        print(f"  Compress time: {t_compress:.3f}s")
        pipeline.print_stats()

        # ── Decompress ──
        pipeline2 = CompressionPipeline.from_config(config)
        t0 = time.perf_counter()
        dec_gaussian, dec_deformation, manifest = pipeline2.decompress_from_archive(archive)
        t_decode = time.perf_counter() - t0
        print(f"  Decode time: {t_decode:.3f}s")

        # ── Chunking ──
        chunker = ModelChunker(chunk_size=args.chunk_size)
        chunks = chunker.split(archive)
        chunk_sizes = [len(c) for c in chunks]

        # ── Render decompressed frames + quality metrics ──
        quality_metrics: Dict[str, Any] = {}
        e2e_metrics: Dict[str, Any] = {}
        export_time_per_frame = 0.05  # fallback estimate
        exp_dir = os.path.join(args.output_dir, config_name, "decompressed")

        if not args.skip_render and test_cameras is not None:
            print("\n  Rendering decompressed frames...")
            dec_paths, t_render = render_frames_from_decompressed(
                dec_gaussian, dec_deformation, args.configs,
                exp_dir, args.num_frames,
                test_cameras=test_cameras,
                cam_type=cam_type,
            )
            export_time_per_frame = t_render / max(len(dec_paths), 1)
            print(f"  Render time: {t_render:.3f}s ({export_time_per_frame * 1000:.1f}ms/frame)")

            # ── Per-frame PSNR / SSIM / LPIPS ──
            if ref_paths and dec_paths:
                print("  Computing quality metrics (PSNR, SSIM, LPIPS)...")
                import torchvision
                psnr_list, ssim_list, lpips_list = [], [], []
                n_compare = min(len(ref_paths), len(dec_paths))
                for idx in range(n_compare):
                    ref_img = torchvision.io.read_image(ref_paths[idx]).float().div(255.0).cuda()
                    dec_img = torchvision.io.read_image(dec_paths[idx]).float().div(255.0).cuda()
                    psnr_list.append(compute_psnr(ref_img, dec_img))
                    ssim_list.append(compute_ssim(ref_img, dec_img))
                    lpips_list.append(compute_lpips_metric(ref_img, dec_img))

                if psnr_list:
                    quality_metrics["psnr_mean"] = round(float(np.mean(psnr_list)), 4)
                    quality_metrics["ssim_mean"] = round(float(np.mean(ssim_list)), 4)
                    quality_metrics["lpips_mean"] = round(float(np.mean(lpips_list)), 4)
                    quality_metrics["psnr_per_frame"] = [round(v, 4) for v in psnr_list]
                    quality_metrics["ssim_per_frame"] = [round(v, 4) for v in ssim_list]
                    quality_metrics["lpips_per_frame"] = [round(v, 4) for v in lpips_list]
                    print(f"    PSNR:  {quality_metrics['psnr_mean']:.2f} dB")
                    print(f"    SSIM:  {quality_metrics['ssim_mean']:.4f}")
                    print(f"    LPIPS: {quality_metrics['lpips_mean']:.4f}")

            # ── End-to-end: decompressed vs GT ──
            if gt_available and dec_paths:
                print("  Computing end-to-end metrics (decompressed vs GT)...")
                e_psnr, e_ssim, e_lpips = [], [], []
                n_e2e = min(len(dec_paths), len(gt_frames))
                for idx in range(n_e2e):
                    dec_img = torchvision.io.read_image(dec_paths[idx]).float().div(255.0).cuda()
                    gt_img = gt_frames[idx]
                    e_psnr.append(compute_psnr(dec_img, gt_img))
                    e_ssim.append(compute_ssim(dec_img, gt_img))
                    e_lpips.append(compute_lpips_metric(dec_img, gt_img))
                e2e_metrics = {
                    "psnr_mean": round(float(np.mean(e_psnr)), 4),
                    "ssim_mean": round(float(np.mean(e_ssim)), 4),
                    "lpips_mean": round(float(np.mean(e_lpips)), 4),
                    "psnr_per_frame": [round(v, 4) for v in e_psnr],
                    "ssim_per_frame": [round(v, 4) for v in e_ssim],
                    "lpips_per_frame": [round(v, 4) for v in e_lpips],
                }
                print(f"    e2e PSNR:  {e2e_metrics['psnr_mean']:.2f} dB")
                print(f"    e2e SSIM:  {e2e_metrics['ssim_mean']:.4f}")
                print(f"    e2e LPIPS: {e2e_metrics['lpips_mean']:.4f}")

            # ── VMAF (optional) ──
            if not args.skip_vmaf and ref_paths and dec_paths:
                ref_video = os.path.join(args.output_dir, "reference.mp4")
                dec_video = os.path.join(args.output_dir, config_name, "decompressed.mp4")
                vmaf_json = os.path.join(args.output_dir, config_name, "vmaf.json")
                _frames_to_video(ref_dir, ref_video)
                _frames_to_video(exp_dir, dec_video)
                if os.path.exists(ref_video) and os.path.exists(dec_video):
                    vmaf_score = compute_vmaf(ref_video, dec_video, vmaf_json)
                    if vmaf_score is not None:
                        print(f"    VMAF:  {vmaf_score:.2f}")
                        quality_metrics["vmaf"] = vmaf_score

            # ── VMAF vs GT (optional) ──
            if not args.skip_vmaf and gt_available and dec_paths:
                gt_video = os.path.join(args.output_dir, "gt_reference.mp4")
                dec_video_e2e = os.path.join(args.output_dir, config_name, "decompressed.mp4")
                vmaf_gt_json = os.path.join(args.output_dir, config_name, "vmaf_vs_gt.json")
                _frames_to_video(gt_dir, gt_video)
                if not os.path.exists(dec_video_e2e):
                    _frames_to_video(exp_dir, dec_video_e2e)
                if os.path.exists(gt_video) and os.path.exists(dec_video_e2e):
                    vmaf_gt = compute_vmaf(gt_video, dec_video_e2e, vmaf_gt_json)
                    if vmaf_gt is not None:
                        print(f"    VMAF vs GT: {vmaf_gt:.2f}")
                        e2e_metrics["vmaf"] = vmaf_gt
        elif not args.skip_render:
            # No cameras but rendering requested — PLY fallback
            print("\n  Exporting decompressed PLYs (no cameras for image rendering)...")
            dec_paths, t_render = render_frames_from_decompressed(
                dec_gaussian, dec_deformation, args.configs,
                exp_dir, args.num_frames,
            )
            export_time_per_frame = t_render / max(len(dec_paths), 1)
            print(f"  Export time: {t_render:.3f}s ({export_time_per_frame * 1000:.1f}ms/frame)")

        # ── Streaming QoE ──
        qoe = compute_streaming_qoe(
            chunk_sizes=chunk_sizes,
            decode_time_s=t_decode,
            export_time_per_frame_s=export_time_per_frame,
            num_frames=args.num_frames,
            bandwidth_mbps=args.bandwidth_mbps,
        )

        print(f"\n  Streaming QoE @ {args.bandwidth_mbps} Mbps:")
        print(f"    Startup delay:     {qoe['startup_delay_s']:.2f}s")
        print(f"    Rebuffer events:   {qoe['rebuffer_events']}")
        print(f"    Stall duration:    {qoe['total_stall_duration_s']:.2f}s")
        print(f"    QoE score:         {qoe['qoe_score']:.2f}/5.0")
        print(f"    E2E latency:       {qoe['e2e_latency_s']:.2f}s")

        # ── Collect results ──
        entry = {
            "config_name": config_name,
            "config_file": config_path,
            "original_size_bytes": orig_total_bytes,
            "compressed_size_bytes": compressed_bytes,
            "compression_ratio": round(ratio, 4),
            "savings_pct": round((1 - compressed_bytes / orig_total_bytes) * 100, 2),
            "compress_time_s": round(t_compress, 4),
            "decode_time_s": round(t_decode, 4),
            "num_chunks": len(chunks),
            "num_gaussians_original": orig_gaussian.num_gaussians,
            "num_gaussians_compressed": dec_gaussian.num_gaussians,
            "sh_degree_original": orig_gaussian.sh_degree,
            "sh_degree_compressed": dec_gaussian.sh_degree,
            "compression_fidelity": quality_metrics,
            "end_to_end_quality": e2e_metrics,
            "training_baseline": baseline_vs_gt,
            "streaming_qoe": qoe,
            "pipeline_stats": [
                {
                    "strategy": s.strategy_name,
                    "ratio": round(s.ratio, 4),
                    "savings_pct": round(s.savings_pct, 2),
                    "compress_time_s": round(s.compression_time_s, 4),
                    "decompress_time_s": round(s.decompression_time_s, 4),
                    "extra": s.extra,
                }
                for s in pipeline.stats
            ],
        }
        results.append(entry)
        print()

    # ── Write summary (merge with existing) ───────────────────────────
    summary_path = os.path.join(args.output_dir, "benchmark_results.json")

    # Load previous results and merge by config_name (new runs update old)
    existing_results: List[Dict[str, Any]] = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = []
            print(f"  Loaded {len(existing_results)} existing entries from {summary_path}")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: could not read existing JSON ({e}), starting fresh")
            existing_results = []

    # Build lookup of new results by config_name
    new_by_name = {r["config_name"]: r for r in results}

    # Merge: update existing entries, keep old ones not in this run
    merged_results: List[Dict[str, Any]] = []
    seen_names: set = set()
    for existing in existing_results:
        name = existing.get("config_name", "")
        if name in new_by_name:
            # Replace with updated result
            merged_results.append(new_by_name[name])
            seen_names.add(name)
        else:
            # Keep previous entry
            merged_results.append(existing)
            seen_names.add(name)
    # Append brand-new configs that weren't in the old file
    for r in results:
        if r["config_name"] not in seen_names:
            merged_results.append(r)

    with open(summary_path, "w") as f:
        json.dump(merged_results, f, indent=2, default=str)
    print(f"  Wrote {len(merged_results)} entries to {summary_path} "
          f"({len(merged_results) - len(existing_results)} new, "
          f"{len(existing_results)} previously existing)")

    # CSV summary (also merged)
    csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    headers = [
        "config", "orig_MB", "comp_MB", "ratio", "savings%",
        "compress_s", "decode_s", "chunks",
        "gaussians_orig", "gaussians_comp",
        "sh_orig", "sh_comp",
        "cf_psnr_db", "cf_ssim", "cf_lpips",
        "e2e_psnr_db", "e2e_ssim", "e2e_lpips",
        "bl_psnr_db", "bl_ssim", "bl_lpips",
        "startup_s", "rebuffers", "stall_s", "qoe",
    ]

    def _result_to_csv_row(r: Dict[str, Any]) -> List[str]:
        cf = r.get("compression_fidelity", {})
        e2e = r.get("end_to_end_quality", {})
        bl = r.get("training_baseline", {})
        return [
            r["config_name"],
            f"{r['original_size_bytes'] / 1e6:.2f}",
            f"{r['compressed_size_bytes'] / 1e6:.2f}",
            f"{r['compression_ratio']:.2f}",
            f"{r['savings_pct']:.1f}",
            f"{r['compress_time_s']:.3f}",
            f"{r['decode_time_s']:.3f}",
            str(r["num_chunks"]),
            str(r["num_gaussians_original"]),
            str(r["num_gaussians_compressed"]),
            str(r["sh_degree_original"]),
            str(r["sh_degree_compressed"]),
            f"{cf.get('psnr_mean', 0):.2f}",
            f"{cf.get('ssim_mean', 0):.4f}",
            f"{cf.get('lpips_mean', 0):.4f}",
            f"{e2e.get('psnr_mean', 0):.2f}",
            f"{e2e.get('ssim_mean', 0):.4f}",
            f"{e2e.get('lpips_mean', 0):.4f}",
            f"{bl.get('psnr_mean', 0):.2f}",
            f"{bl.get('ssim_mean', 0):.4f}",
            f"{bl.get('lpips_mean', 0):.4f}",
            f"{r['streaming_qoe']['startup_delay_s']:.2f}",
            str(r["streaming_qoe"]["rebuffer_events"]),
            f"{r['streaming_qoe']['total_stall_duration_s']:.2f}",
            f"{r['streaming_qoe']['qoe_score']:.2f}",
        ]

    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in merged_results:
            try:
                row = _result_to_csv_row(r)
                f.write(",".join(row) + "\n")
            except (KeyError, TypeError) as e:
                print(f"  Warning: skipping CSV row for '{r.get('config_name', '?')}': {e}")

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print(f"  Results:  {summary_path}")
    print(f"  CSV:      {csv_path}")
    print("=" * 70)

    # Print comparison table (all merged entries)
    print(f"\n{'Config':20s} {'Size MB':>8s} {'Ratio':>6s} {'Save%':>6s} "
          f"{'cf_PSNR':>8s} {'e2e_PSNR':>9s} {'bl_PSNR':>8s} "
          f"{'cf_SSIM':>8s} {'e2e_SSIM':>9s} "
          f"{'Decode':>7s} {'QoE':>4s}")
    print("-" * 120)
    bl = merged_results[0].get("training_baseline", {}) if merged_results else {}
    bl_psnr = f"{bl['psnr_mean']:.2f}" if 'psnr_mean' in bl else "  n/a"
    for r in merged_results:
        cf = r.get("compression_fidelity", {})
        e2e = r.get("end_to_end_quality", {})
        cf_p = f"{cf['psnr_mean']:.2f}" if 'psnr_mean' in cf else "  n/a"
        e2e_p = f"{e2e['psnr_mean']:.2f}" if 'psnr_mean' in e2e else "   n/a"
        cf_s = f"{cf['ssim_mean']:.4f}" if 'ssim_mean' in cf else "  n/a"
        e2e_s = f"{e2e['ssim_mean']:.4f}" if 'ssim_mean' in e2e else "   n/a"
        print(f"{r['config_name']:20s} "
              f"{r['compressed_size_bytes'] / 1e6:8.2f} "
              f"{r['compression_ratio']:6.2f} "
              f"{r['savings_pct']:5.1f}% "
              f"{cf_p:>8s} "
              f"{e2e_p:>9s} "
              f"{bl_psnr:>8s} "
              f"{cf_s:>8s} "
              f"{e2e_s:>9s} "
              f"{r['decode_time_s']:6.3f}s "
              f"{r['streaming_qoe']['qoe_score']:4.1f}")


if __name__ == "__main__":
    main()
