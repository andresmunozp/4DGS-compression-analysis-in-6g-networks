## 4.2 Compression Configurations

We implemented a comprehensive framework with multiple compression configurations, each designed to explore different trade-offs between compression ratio and visual quality. Our system leverages seven distinct compression strategies that can be combined in various configurations:

### Compression Strategies

Our framework supports the following core strategies:

- **Pruning Strategy**: Removes low-importance Gaussians based on opacity thresholds and parameter counts
- **SH Reduction Strategy**: Reduces spherical harmonics degree to decrease attribute dimensionality
- **Quantization Strategy**: Converts floating-point attributes to lower precision (float16, int8)
- **HexPlane Compression Strategy**: Compresses deformation fields using HexPlane factorization with SVD or quantization
- **Entropy Coding Strategy**: Applies byte-level compression using zlib/zstd algorithms
- **EntropyGS Strategy**: Applies distribution-aware adaptive quantization with arithmetic coding
- **LightGaussian Pruning Strategy**: Removes Gaussians using volume × importance scoring (parameter or render-based)

### Predefined Configurations

We evaluate our compression pipeline under seven main configurations, each representing different points in the quality-compression trade-off space:

#### 1. **Lossless Configuration**
The baseline for measuring maximum achievable lossless compression. This configuration applies only entropy coding without any lossy transformations:

| Parameter | Value |
|-----------|-------|
| **Strategy** | EntropyCodingStrategy |
| **Algorithm** | zlib |
| **Compression Level** | 9 |

This configuration serves as a lower bound for compression ratio, ensuring no visual degradation while measuring the overhead of structuring the model data.

#### 2. **Balanced Configuration**
A well-tuned configuration that balances compression efficiency with quality preservation. It combines soft pruning, moderate SH reduction, and selective quantization:

| Parameter | Value |
|-----------|-------|
| **Pruning** | |
| └─ Opacity Threshold | 0.005 |
| └─ Max Gaussians | 150,000 |
| **SH Reduction** | |
| └─ Target SH Degree | 1 |
| **Quantization** | |
| └─ All attributes | float16 |
| └─ Deformation quantization | False |
| **HexPlane** | quantize method |
| **Entropy Coding** | zlib level 6 |

This configuration prioritizes maintaining visual fidelity while achieving moderate compression through progressive reduction of less-critical information.

#### 3. **Aggressive Configuration**
Targets maximum compression ratio while accepting moderate quality loss. Employs heavy pruning, aggressive SH reduction, and aggressive quantization:

| Parameter | Value |
|-----------|-------|
| **Pruning** | |
| └─ Opacity Threshold | 0.01 |
| └─ Max Gaussians | 80,000 |
| **SH Reduction** | |
| └─ Target SH Degree | 0 |
| **Quantization** | |
| └─ All attributes | float16 |
| └─ Deformation quantization | True |
| **HexPlane** | SVD method, rank 4 |
| **Entropy Coding** | zlib level 9 |

This configuration maximizes compression ratio by aggressively reducing model size through stronger pruning and complete removal of spherical harmonics.

#### 4. **lightgaussian_balanced Configuration**
Applies the LightGaussian pruning strategy inspired by Fan et al. [2024], which uses volume × importance scoring with deformation-aware weighting for 4DGS:

| Parameter | Value |
|-----------|-------|
| **LightGaussian Pruning** | |
| └─ Prune Percentage | 30% |
| └─ Volume Power | 0.1 |
| └─ Importance Mode | parameter (fast) |
| └─ Deformation Weight | 0.5 |
| **SH Reduction** | |
| └─ Target SH Degree | 1 |
| **Quantization** | |
| └─ All attributes | float16 |
| └─ Deformation quantization | False |
| **HexPlane** | quantize method |
| **Entropy Coding** | zlib level 6 |

This configuration combines principled importance-based pruning with standard quantization, achieving good results for dynamic scenes while preserving temporal coherence.

#### 5. **sh Configuration**
Isolates spherical harmonics simplification to measure the impact of SH-order reduction without pruning or quantization effects:

| Parameter | Value |
|-----------|-------|
| **Strategy** | SHReductionStrategy |
| **Target SH Degree** | 2 |

This configuration is used as an ablation baseline to quantify how much compression can be attributed only to reducing SH representation complexity.

#### 6. **pruning Configuration**
Isolates structure-level reduction by applying only Gaussian pruning with conservative thresholds:

| Parameter | Value |
|-----------|-------|
| **Strategy** | PruningStrategy |
| **Opacity Threshold** | 0.005 |
| **Max Gaussians** | 150,000 |

This configuration is used as an ablation baseline to evaluate how many redundant Gaussians can be removed before introducing major visual degradation.

#### 7. **EntropyGS Configuration**
Implements distribution-aware entropy coding as proposed in Huang et al. [arXiv:2508.10227]. This approach replaces generic byte-level compression with adaptive quantization guided by attribute distributions:

| Parameter | Value |
|-----------|-------|
| **Pruning** | |
| └─ Opacity Threshold | 0.005 |
| └─ Max Gaussians | 150,000 |
| **SH Reduction** | |
| └─ Target SH Degree | 2 |
| **EntropyGS** | |
| └─ Profile | medium |
| └─ GMM Max Components | 4 |

This configuration treats each attribute group's distribution distinctly, enabling near-optimal compression without separate byte-level compression overhead.

### Additional Specialized Configurations

Beyond the main configurations, we provide specialized variants for specific research needs:

- **quantize_only**: Quantization without pruning (baseline for quantization analysis)
- **sh_reduction**: Alternative SH-only setup for polynomial approximation sensitivity studies
- **pruning**: Alternative pruning-only setup for structure simplification studies
- **hexplane_svd**: High-rank SVD factorization for deformation field analysis
- **hexplane_downsample**: Spatial downsampling combined with HexPlane compression
- **lightgaussian_aggressive**: Aggressive variant of LightGaussian with 50% pruning
- **streaming_optimized**: Configuration optimized for progressive rendering and streaming scenarios

### Configuration Selection Strategy

Each configuration was designed to address specific research questions:

1. **Lossless** provides a baseline for measuring model structure overhead
2. **Balanced** represents the production-ready sweet spot for most applications
3. **Aggressive** explores the limits of lossy compression for dynamic Gaussians
4. **lightgaussian_balanced** investigates importance-based pruning effectiveness
5. **sh** isolates SH-degree reduction effects
6. **pruning** isolates structure-level pruning effects
7. **EntropyGS** evaluates adaptive arithmetic coding for 3DGS

The modular pipeline design allows flexible combination of strategies, enabling researchers to design custom configurations for their specific needs.
