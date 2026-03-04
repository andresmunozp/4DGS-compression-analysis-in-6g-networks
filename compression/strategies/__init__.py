"""
Compression strategies for 4DGS models.
"""

from compression.strategies.quantization import QuantizationStrategy
from compression.strategies.pruning import PruningStrategy
from compression.strategies.sh_reduction import SHReductionStrategy
from compression.strategies.hexplane_compression import HexPlaneCompressionStrategy
from compression.strategies.entropy_coding import EntropyCodingStrategy
from compression.strategies.entropy_coding_entropygs import EntropyGSStrategy
from compression.strategies.lightgaussian_pruning import LightGaussianPruningStrategy

STRATEGY_CLASSES = {
    "QuantizationStrategy": QuantizationStrategy,
    "PruningStrategy": PruningStrategy,
    "SHReductionStrategy": SHReductionStrategy,
    "HexPlaneCompressionStrategy": HexPlaneCompressionStrategy,
    "EntropyCodingStrategy": EntropyCodingStrategy,
    "EntropyGSStrategy": EntropyGSStrategy,
    "LightGaussianPruningStrategy": LightGaussianPruningStrategy,
}

__all__ = [
    "QuantizationStrategy",
    "PruningStrategy",
    "SHReductionStrategy",
    "HexPlaneCompressionStrategy",
    "EntropyCodingStrategy",
    "EntropyGSStrategy",
    "LightGaussianPruningStrategy",
    "STRATEGY_CLASSES",
]
