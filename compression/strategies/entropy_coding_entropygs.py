"""
EntropyGS — Factorized, parameterized entropy coding for Gaussian Splatting.

Implements the core ideas from:
  Huang et al., "EntropyGS: An Efficient Entropy Coding on 3D Gaussian
  Splatting", arXiv 2508.10227, Aug 2025.

Key improvements over the generic ``EntropyCodingStrategy`` (zlib/zstd/lz4):

1. **Adaptive quantization per attribute group**
   - Geometry  (xyz)            → high bit-depth  (15-17 bits)
   - Rotation / Scaling / Opacity → medium          (7-8 bits)
   - SH-DC    (features_dc)     → medium           (8 bits)
   - SH-AC    (features_rest)   → low              (2-5 bits)

2. **Distribution-aware probability models**
   - SH-AC   → Laplace distribution  (MLE: median + mean-abs-deviation)
   - Rotation / Scaling / Opacity → Gaussian Mixture Model (EM, ≤4 components)
   - Geometry & SH-DC            → generic (histogram-based PMF)

3. **PMF-reordered symbol packing + zstd** for near-optimal byte encoding.
   Symbols are reordered by their PMF rank (most probable first), which
   clusters low-valued bytes together and dramatically improves the
   effectiveness of the backend compressor (zstd).

4. **Factorized design**: each attribute *channel* is encoded independently,
   allowing embarrassingly parallel encoding/decoding.

This version is fully vectorized with NumPy — no per-symbol Python loops —
making it practical for real models with 100K+ Gaussians.

Dependencies
------------
- ``scipy``       — Laplace distribution CDF
- ``numpy``       — array ops, histogram
- ``zstandard``   — (optional, falls back to zlib) high-performance backend
- ``scikit-learn`` (optional) — sklearn.mixture.GaussianMixture for GMM
  If not installed, a lightweight single-Gaussian fallback is used.

Usage inside a compression pipeline config YAML::

    strategies:
      - name: EntropyGSStrategy
        params:
          profile: medium          # "large" | "medium" | "small"
          gmm_max_components: 4
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import laplace as _laplace_dist
from scipy.stats import norm as _norm_dist

from compression.base import CompressionStrategy, DeformationData, GaussianData

# Optional GMM from sklearn — fall back to single Gaussian if missing
try:
    from sklearn.mixture import GaussianMixture as _GMM
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional high-performance backend
try:
    import zstandard as _zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# ---------------------------------------------------------------------------
# Quantization depth profiles (bits) — Table 2 of the paper
# ---------------------------------------------------------------------------

_QUANT_PROFILES: Dict[str, Dict[str, int]] = {
    "large": {
        "xyz": 17,
        "rotation": 8,
        "scaling": 8,
        "opacity": 8,
        "features_dc": 8,
        "features_rest": 5,
    },
    "medium": {
        "xyz": 16,
        "rotation": 8,
        "scaling": 8,
        "opacity": 8,
        "features_dc": 8,
        "features_rest": 4,
    },
    "small": {
        "xyz": 15,
        "rotation": 7,
        "scaling": 7,
        "opacity": 7,
        "features_dc": 8,
        "features_rest": 3,
    },
}


# =====================================================================
# Backend byte-level compressor (zstd preferred, zlib fallback)
# =====================================================================

def _backend_compress(data: bytes, level: int = 12) -> bytes:
    """Compress raw bytes with the best available backend."""
    if HAS_ZSTD:
        return _zstd.ZstdCompressor(level=level).compress(data)
    return zlib.compress(data, min(level, 9))


def _backend_decompress(data: bytes) -> bytes:
    """Decompress bytes; auto-detects zstd vs zlib."""
    if len(data) >= 4 and data[:4] == b'\x28\xb5\x2f\xfd':
        # zstd magic number
        if not HAS_ZSTD:
            raise ImportError("Data was compressed with zstd but zstandard is not installed")
        return _zstd.ZstdDecompressor().decompress(data)
    return zlib.decompress(data)


# =====================================================================
# Distribution estimation helpers
# =====================================================================

def _estimate_laplace(values: np.ndarray) -> Tuple[float, float]:
    """MLE for Laplace distribution: location = median, scale = MAD."""
    mu = float(np.median(values))
    b = float(np.mean(np.abs(values - mu)))
    if b < 1e-12:
        b = 1e-12
    return mu, b


def _estimate_gmm(values: np.ndarray, max_components: int = 4
                   ) -> List[Tuple[float, float, float]]:
    """Fit a Gaussian Mixture Model with ≤ *max_components* via BIC.

    Returns list of (weight, mean, std) tuples.
    Falls back to single Gaussian if sklearn is unavailable.
    """
    if values.size < 10:
        mu = float(np.mean(values))
        std = float(np.std(values)) or 1e-6
        return [(1.0, mu, std)]

    if HAS_SKLEARN:
        best_bic = np.inf
        best_gmm = None
        data = values.reshape(-1, 1)
        # Sub-sample for speed when N is large
        if data.shape[0] > 50_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(data.shape[0], 50_000, replace=False)
            data_fit = data[idx]
        else:
            data_fit = data
        for n in range(1, max_components + 1):
            try:
                gmm = _GMM(n_components=n, max_iter=80,
                            covariance_type="full", random_state=42)
                gmm.fit(data_fit)
                bic = gmm.bic(data_fit)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception:
                continue
        if best_gmm is not None:
            components = []
            for k in range(best_gmm.n_components):
                w = float(best_gmm.weights_[k])
                m = float(best_gmm.means_[k, 0])
                s = float(np.sqrt(best_gmm.covariances_[k, 0, 0]))
                if s < 1e-12:
                    s = 1e-12
                components.append((w, m, s))
            return components

    # Fallback: single Gaussian
    mu = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        std = 1e-12
    return [(1.0, mu, std)]


# =====================================================================
# PMF construction from estimated distributions
# =====================================================================

def _build_pmf_laplace(num_levels: int, vmin: float, vmax: float,
                       mu: float, b: float) -> np.ndarray:
    """Build a PMF over *num_levels* uniform bins using a Laplace CDF."""
    edges = np.linspace(vmin, vmax, num_levels + 1)
    cdf_vals = _laplace_dist.cdf(edges, loc=mu, scale=b)
    pmf = np.diff(cdf_vals)
    pmf = np.clip(pmf, 1e-10, None)
    pmf /= pmf.sum()
    return pmf


def _build_pmf_gmm(num_levels: int, vmin: float, vmax: float,
                    components: List[Tuple[float, float, float]]) -> np.ndarray:
    """Build a PMF using a Gaussian Mixture CDF."""
    edges = np.linspace(vmin, vmax, num_levels + 1)
    cdf_vals = np.zeros(num_levels + 1, dtype=np.float64)
    for w, m, s in components:
        cdf_vals += w * _norm_dist.cdf(edges, loc=m, scale=s)
    pmf = np.diff(cdf_vals)
    pmf = np.clip(pmf, 1e-10, None)
    pmf /= pmf.sum()
    return pmf


def _build_pmf_histogram(values_quantized: np.ndarray, num_levels: int
                         ) -> np.ndarray:
    """Build an empirical PMF from quantized integer histogram."""
    counts = np.bincount(values_quantized.astype(np.int64).ravel(),
                         minlength=num_levels)
    pmf = counts.astype(np.float64)
    pmf = np.clip(pmf, 1e-10, None)
    pmf /= pmf.sum()
    return pmf


# =====================================================================
# Uniform min-max quantization  (vectorized, operates on full arrays)
# =====================================================================

def _minmax_quantize(arr: np.ndarray, num_bits: int
                     ) -> Tuple[np.ndarray, float, float]:
    """Uniform min-max quantization (Eq. 2 of the paper).

    Returns (quantized_int_array, vmin, vmax).
    """
    num_levels = 1 << num_bits
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    q = np.round((arr - vmin) * (num_levels - 1) / (vmax - vmin)).astype(np.int32)
    np.clip(q, 0, num_levels - 1, out=q)
    return q, vmin, vmax


def _minmax_dequantize(q: np.ndarray, vmin: float, vmax: float,
                       num_bits: int) -> np.ndarray:
    """Inverse of ``_minmax_quantize``."""
    num_levels = 1 << num_bits
    return q.astype(np.float32) / (num_levels - 1) * (vmax - vmin) + vmin


# =====================================================================
# Per-channel encode / decode  — VECTORIZED (no per-symbol loops)
# =====================================================================
#
# Strategy:
#   1. Quantize to integers [0, L)
#   2. Fit distribution → PMF
#   3. Build a rank-order map (PMF-sorted): most probable symbols get
#      lowest rank values.  This clusters symbols near 0, making the
#      output highly compressible by any byte-level backend (zstd, zlib).
#   4. Map symbols → ranks (vectorized lookup)
#   5. Pack ranks into minimal bytes (≤8-bit → uint8, ≤16-bit → uint16)
#   6. Compress packed bytes with zstd/zlib.
#
# Decoding is the reverse: decompress → unpack → inverse-rank-map → dequant.
# =====================================================================

def _encode_channel(values: np.ndarray, num_bits: int,
                    dist_type: str,
                    gmm_max_components: int = 4
                    ) -> Tuple[bytes, dict]:
    """Quantize + fit distribution + PMF-rank-reorder + compress one channel.

    Fully vectorized — no Python loops over symbols.
    """
    N = values.shape[0]
    num_levels = 1 << num_bits

    # 1) Quantize
    q, vmin, vmax = _minmax_quantize(values, num_bits)

    # 2) Estimate distribution & build PMF
    if dist_type == "laplace":
        mu, b = _estimate_laplace(values)
        pmf = _build_pmf_laplace(num_levels, vmin, vmax, mu, b)
        dist_params = {"mu": mu, "b": b}
    elif dist_type == "gmm":
        comps = _estimate_gmm(values, gmm_max_components)
        pmf = _build_pmf_gmm(num_levels, vmin, vmax, comps)
        dist_params = {"components": comps}
    else:  # histogram
        pmf = _build_pmf_histogram(q, num_levels)
        dist_params = {}  # will store ranks directly

    # 3) Build rank-order map: sort symbols by descending probability
    rank_order = np.argsort(-pmf).astype(np.int32)     # symbol → rank position
    sym_to_rank = np.empty(num_levels, dtype=np.int32)
    sym_to_rank[rank_order] = np.arange(num_levels, dtype=np.int32)

    # 4) Map quantized symbols → ranks  (vectorized)
    ranked = sym_to_rank[q.ravel()].reshape(q.shape)

    # 5) Pack to minimal dtype
    if num_levels <= 256:
        packed_arr = ranked.astype(np.uint8)
    elif num_levels <= 65536:
        packed_arr = ranked.astype(np.uint16)
    else:
        packed_arr = ranked.astype(np.uint32)

    # 6) Compress with backend
    raw_bytes = packed_arr.tobytes()
    comp_bytes = _backend_compress(raw_bytes)

    # --- Compute Shannon entropy for analytics ---
    h = float(-np.sum(pmf * np.log2(np.clip(pmf, 1e-30, None))))

    meta = {
        "N": N,
        "num_bits": num_bits,
        "num_levels": num_levels,
        "vmin": vmin,
        "vmax": vmax,
        "dist_type": dist_type,
        "dist_params": dist_params,
        "rank_order": rank_order.tolist(),   # needed for decoding
        "packed_dtype": str(packed_arr.dtype),
        "compressed_size": len(comp_bytes),
        "raw_size": len(raw_bytes),
        "shannon_entropy_bps": round(h, 3),
        "original_shape": list(values.shape),
    }
    return comp_bytes, meta


def _decode_channel(comp_bytes: bytes, meta: dict) -> np.ndarray:
    """Decode one channel: decompress → un-rank → dequantize."""
    N = meta["N"]
    num_bits = meta["num_bits"]
    num_levels = meta["num_levels"]
    vmin = meta["vmin"]
    vmax = meta["vmax"]
    rank_order = np.array(meta["rank_order"], dtype=np.int32)
    packed_dtype = np.dtype(meta["packed_dtype"])

    # 1) Decompress
    raw_bytes = _backend_decompress(comp_bytes)
    ranked = np.frombuffer(raw_bytes, dtype=packed_dtype).copy()

    # 2) Inverse rank map: rank → symbol
    rank_to_sym = rank_order  # rank_order[rank] = original symbol
    symbols = rank_to_sym[ranked.astype(np.int32)]

    # 3) Dequantize
    return _minmax_dequantize(symbols, vmin, vmax, num_bits)


# =====================================================================
# Serialization helpers — pack multiple channel blobs into one bytes obj
# =====================================================================

def _pack_channels(channel_blobs: List[bytes]) -> bytes:
    """Pack a list of byte-blobs into a single bytes object with a length header."""
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(channel_blobs)))
    for blob in channel_blobs:
        buf.write(struct.pack("<I", len(blob)))
        buf.write(blob)
    return buf.getvalue()


def _unpack_channels(data: bytes) -> List[bytes]:
    """Inverse of ``_pack_channels``."""
    buf = io.BytesIO(data)
    n = struct.unpack("<I", buf.read(4))[0]
    blobs = []
    for _ in range(n):
        length = struct.unpack("<I", buf.read(4))[0]
        blobs.append(buf.read(length))
    return blobs


# =====================================================================
# Attribute-group classification (paper Section 3)
# =====================================================================

_ATTR_DIST_TYPE = {
    "xyz": "histogram",           # Geometry: no clear distribution → histogram
    "features_dc": "histogram",   # SH-DC: no clear distribution → histogram
    "features_rest": "laplace",   # SH-AC: Laplace
    "opacity": "gmm",             # GMM ≤ 4 components
    "scaling": "gmm",
    "rotation": "gmm",
}


# =====================================================================
# Strategy class
# =====================================================================

class EntropyGSStrategy(CompressionStrategy):
    """Factorized, parameterized entropy coding inspired by EntropyGS.

    This is a **replacement / upgrade** for the generic ``EntropyCodingStrategy``
    that applies *distribution-aware adaptive quantization* per attribute
    channel, combined with PMF-rank-reordered symbol packing and a
    high-performance byte-level compressor (zstd / zlib).

    Parameters
    ----------
    profile : str
        Quantization-depth preset: ``"large"`` | ``"medium"`` | ``"small"``.
    gmm_max_components : int
        Maximum number of Gaussian mixture components for rotation/scaling/
        opacity (default 4, as in the paper).
    custom_bits : dict, optional
        Override individual attribute bit-depths, e.g.
        ``{"features_rest": 3, "xyz": 15}``.
    """

    VALID_PROFILES = tuple(_QUANT_PROFILES.keys())

    def __init__(
        self,
        profile: str = "medium",
        gmm_max_components: int = 4,
        custom_bits: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        if profile not in self.VALID_PROFILES:
            raise ValueError(
                f"profile must be one of {self.VALID_PROFILES}, got '{profile}'"
            )
        self.profile = profile
        self.gmm_max_components = gmm_max_components
        self._bit_depths = dict(_QUANT_PROFILES[profile])
        if custom_bits:
            self._bit_depths.update(custom_bits)

        super().__init__(
            profile=profile,
            gmm_max_components=gmm_max_components,
            custom_bits=custom_bits,
            **kwargs,
        )
        self._encode_meta: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return f"entropygs_{self.profile}"

    # ────────────────────────────────────────────────────────────────
    # Gaussian compression
    # ────────────────────────────────────────────────────────────────

    def compress_gaussian(self, data: GaussianData) -> GaussianData:
        """Quantize + distribution-fit + PMF-rank + compress each channel."""
        self._encode_meta = {}
        ATTRS = ["xyz", "features_dc", "features_rest",
                 "opacity", "scaling", "rotation"]

        for attr in ATTRS:
            arr = getattr(data, attr)
            original_dtype = str(arr.dtype)
            original_shape = list(arr.shape)
            num_bits = self._bit_depths.get(attr, 8)
            dist_type = _ATTR_DIST_TYPE.get(attr, "histogram")

            # Process each "column" (channel) independently — factorized
            flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
            n_channels = flat.shape[1]

            channel_blobs: List[bytes] = []
            channel_metas: List[dict] = []

            for ch in range(n_channels):
                col = flat[:, ch]
                comp_bytes, ch_meta = _encode_channel(
                    col, num_bits, dist_type, self.gmm_max_components
                )
                channel_blobs.append(comp_bytes)
                channel_metas.append(ch_meta)

            packed = _pack_channels(channel_blobs)
            comp_arr = np.frombuffer(packed, dtype=np.uint8).copy()

            self._encode_meta[attr] = {
                "original_dtype": original_dtype,
                "original_shape": original_shape,
                "original_bytes": arr.nbytes,
                "compressed_bytes": len(packed),
                "num_bits": num_bits,
                "dist_type": dist_type,
                "n_channels": n_channels,
                "channel_metas": channel_metas,
            }

            setattr(data, attr,
                    comp_arr.reshape(1, -1) if comp_arr.ndim == 1 else comp_arr)

        # Auxiliary arrays — simple zlib fallback
        for attr in ("deformation_table", "deformation_accum"):
            arr = getattr(data, attr, None)
            if arr is not None:
                raw = arr.tobytes()
                compressed = zlib.compress(raw, 6)
                comp_arr = np.frombuffer(compressed, dtype=np.uint8).copy()
                self._encode_meta[attr] = {
                    "original_dtype": str(arr.dtype),
                    "original_shape": list(arr.shape),
                    "original_bytes": len(raw),
                    "compressed_bytes": len(compressed),
                    "codec": "zlib",
                }
                setattr(data, attr,
                        comp_arr.reshape(1, -1) if comp_arr.ndim == 1 else comp_arr)

        return data

    # ────────────────────────────────────────────────────────────────
    # Gaussian decompression
    # ────────────────────────────────────────────────────────────────

    def decompress_gaussian(
        self, data: GaussianData, metadata: Dict[str, Any]
    ) -> GaussianData:
        encode_meta = metadata.get("encode_meta", {})

        MAIN_ATTRS = ["xyz", "features_dc", "features_rest",
                      "opacity", "scaling", "rotation"]

        for attr in MAIN_ATTRS:
            if attr not in encode_meta:
                continue
            info = encode_meta[attr]
            comp_arr = getattr(data, attr, None)
            if comp_arr is None:
                continue

            packed = comp_arr.tobytes()
            channel_blobs = _unpack_channels(packed)
            n_channels = info["n_channels"]
            channel_metas = info["channel_metas"]
            original_shape = tuple(info["original_shape"])
            N = channel_metas[0]["N"]

            flat = np.empty((N, n_channels), dtype=np.float32)
            for ch in range(n_channels):
                flat[:, ch] = _decode_channel(channel_blobs[ch],
                                              channel_metas[ch])

            restored = flat.reshape(original_shape)
            orig_dt = np.dtype(info["original_dtype"])
            if orig_dt != np.float32:
                restored = restored.astype(orig_dt)
            setattr(data, attr, restored)

        # Auxiliary arrays — zlib
        for attr in ("deformation_table", "deformation_accum"):
            if attr not in encode_meta:
                continue
            info = encode_meta[attr]
            if info.get("codec") != "zlib":
                continue
            comp_arr = getattr(data, attr, None)
            if comp_arr is None:
                continue
            raw = zlib.decompress(comp_arr.tobytes())
            dtype = np.dtype(info["original_dtype"])
            shape = tuple(info["original_shape"])
            setattr(data, attr,
                    np.frombuffer(raw, dtype=dtype).reshape(shape).copy())

        return data

    # ────────────────────────────────────────────────────────────────
    # Deformation (pass-through)
    # ────────────────────────────────────────────────────────────────

    def compress_deformation(self, data: DeformationData) -> DeformationData:
        return data

    def decompress_deformation(
        self, data: DeformationData, metadata: Dict[str, Any]
    ) -> DeformationData:
        return data

    # ────────────────────────────────────────────────────────────────
    # Metadata
    # ────────────────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        summary = {}
        for attr, info in self._encode_meta.items():
            orig = info.get("original_bytes", 0)
            comp = info.get("compressed_bytes", 0)
            ratio = orig / comp if comp > 0 else 0
            summary[attr] = {
                "original_KB": round(orig / 1024, 1),
                "compressed_KB": round(comp / 1024, 1),
                "ratio": round(ratio, 2),
                "bits": info.get("num_bits", "n/a"),
                "dist": info.get("dist_type", "n/a"),
            }

        return {
            "strategy": self.name,
            "params": self.params,
            "encode_meta": self._encode_meta,
            "summary": summary,
        }
