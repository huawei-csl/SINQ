## Overview

This document covers SINQ's internal architecture, including backend selection logic, kernel implementations, and configuration details not covered in the README. SINQ (Sinkhorn-Normalized Quantization; https://github.com/huawei-csl/SINQ) is quantization method that uses dual-scale quantization with Sinkhorn normalization. It supports GemLite and PyTorch kernels as a two inference backends in a codebase.

```bash
# Formula for dual-scale quantization
# Q: Quantized weight
# scale_row, scale_col: 2D scaling factors (per-group row & per-output column by Skin-horn)
Traditional:  W = Q * scale_row
SINQ:        W = Q * scale_row * scale_col
```

GemLite (https://github.com/dropbox/gemlite) is the optimized CUDA kernels for 4-bit quantization, which is recommended for production level. If GemLite is unable, fallback leads to PyTorch kernels, which supports all bit-widths and configurations.

SINQ introduces both 1D tiling and 2D tiling, but Figure 4 shows there is less overhead with minimal accuracy loss. Because SINQ is calibration-free, it requires no calibration dataset and avoids
element-wise scaling operations during inference. This makes it more memory-efficient
than calibration-based methods like AWQ, GPTQ, and SmoothQuant.

## Supported Kernels and Backends

### 01. Backend Selection Logic

SINQ automatically selects the optimal backend based on configuration, by using `can_use_gemlite`. PyTorch backend is adopted only when GemLite is unavailable.

**GemLite Backend Requirements**: This CUDA kernel requires `gemlite` package, and int4 inference, which is compatible with 1D tiling. `out_features >= 16` AND `in_features >= 16` are required as a layer dimension.

> Dimension size limit prevents low accuracy. See https://github.com/huawei-csl/SINQ/issues/10 for more info


**PyTorch Backend Requirements**: Universal fallback using PyTorch operations, which is compatible with torch.compile. Supported for both 1D and 2D tiling, and CPU inference. It can be optimized using `torch.compile` like:

```python
qmodel.forward = torch.compile(
    qmodel.forward,
    dynamic=True,  # Handle variable SeqLen
    fullgraph=False,  # Allow graph breaks for flexibility
    backend="inductor",  # Use PyTorch's optimizing compiler
    mode="reduce-overhead",  # Latency optimization
)
```

### 02. Quantization Kernels (Training-Time)

#### A. Sinkhorn Normalization (`sinkhorn.py:7-79`)

```python
def sinkhorn_normalize(W, num_iters=16):
    """
    Iteratively balances row/column statistics

    Args:
        W: Weight matrix [out_features, in_features]
        num_iters: Number of iterations (default: 16)

    Returns:
        W_normalized: Balanced matrix
        mu1: Row scaling factors
        mu2: Column scaling factors
    """
    # Log-domain computation for numerical stability
    for _ in range(num_iters):
        # Balance rows
        mu1 = row_std(W)
        W = W / mu1

        # Balance columns
        mu2 = col_std(W)
        W = W / mu2

    return W, mu1, mu2
```

#### B. Bit Packing (`BitPack in bitpack.py`)

Flexible bit type as a low-precision is supported.

| Bit-width | Storage Type | Packing Strategy |
|-----------|-------------|------------------|
| 1, 2, 4, 8 | uint8 | 1, 2, 4, 8 values per byte |
| 3, 5, 6 | int32 | 3, 5, 6 values per 32 bits |

### 03. Inference Kernels (Runtime)

#### A. GemLite Forward Pass (`sinqlinear.py:224-255`)

GemLite is the optimized CUDA kernel specialized to int4 MatMul. Packed tensor has `(int4 weight, fp16 zero point, fp16 scale)`.

> Note: Although dtype is different (int4 scale factor vs. fp16 zero point), it is accumulated in Tensor Core in GemLite kernel

```python
def forward_gemlite(self, x):
    # 1. Pre-multiply input by column scale
    x_scaled = self._gl_scale2 * x

    # 2. MatMul using GemLite CUDA kernel
    return gemlite.forward_functional(
        x_scaled,
        self._gl_bias,
        self._gl_tensor_args,  # Packed weights + row scales + zeros
        self._gl_meta_args,    # Shape info, group_size, etc.
        -1,  # Auto-select SM count
    )
```

#### B. PyTorch Forward Pass (`sinqlinear.py:257-261`)

```python
def forward_pytorch(self, x):
    # 1. Dequantize weights
    W_dequant = self.dequantize()

    # 2. Apply standard MatMul
    out = torch.matmul(x, W_dequant.t())

    # 3. Add bias if present
    if self.bias is not None:
        out += self.bias

    return out
```

#### C. Dequantization Process (`quantizer.py:242-341`)

```python
def dequantize(self):
    # 1. Unpack bits → int codes
    Q = unpack_bits(self.W_q, self.nbits)

    # 2. Load scales and zeros
    scale_row = self.scale
    scale_col = self.scale2
    zero = self.zero

    # 3. Apply dequantization: (W_q - z)*s
    if self.non_uniform:  # NF4/NF3
        W = codebook[Q] - zero
    else:  # Uniform quantization
        W = Q - zero

    # 4. Apply dual scales
    W = W * scale_row * scale_col

    return W
```

## Quantization Configuration API

### 01. BaseQuantizeConfig (`sinqlinear.py`)

```python
from sinq import BaseQuantizeConfig

quant_cfg = BaseQuantizeConfig(
    nbits=4,              # Bit-width: 2, 3, 4, 5, 6, 8
    group_size=64,        # Group size: 16, 32, 64, 128
    tiling_mode="1D",     # Tiling: "1D" or "2D"
    method="sinq",        # Method: "sinq", "asinq", custom
)
```

**Parameter Details:**

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| `nbits` | int | 2, 3, 4, 5, 6, 8 | 4 | Quantization bit-width |
| `group_size` | int | 16, 32, 64, 128 | 64 | Number of elements per group |
| `tiling_mode` | str | "1D", "2D" | "1D" | Grouping strategy |
| `method` | str | "sinq", "asinq", custom | "sinq" | Quantization algorithm |

**Common Flags:**

| Flag | Effect |
|------|--------|
| `"sinq" & "sinq_quantAux"` | Standard SINQ quantization |
| `"asinq" & "sinq_awq_l1_quantAux"` | Activation-aware SINQ (requires calibration) |
| `"hqq"` | Use HQQ refinement after Sinkhorn |
| `"nf4"` | 4-bit non-uniform quantization |
| `"nf3"` | 3-bit non-uniform quantization |
| `"quantAux"` | Basic RTN without skin-horn normalization |
| `"nogemlite"` | Force PyTorch backend (disable GemLite) |

**Example Configurations:**

```python
# Production 4-bit with GemLite
config_prod = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    tiling_mode="1D",
    method="sinq"  # Auto uses GemLite
)

# Research with custom bit-width
config_research = BaseQuantizeConfig(
    nbits=3,
    group_size=128,
    tiling_mode="2D",
    method="sinq_nf3"  # Non-uniform 3-bit
)

# CPU inference (force PyTorch)
config_cpu = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    tiling_mode="1D",
    method="sinq_nogemlite"  # Disable GemLite
)

# Calibrated quantization
config_calibrated = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    tiling_mode="1D",
    method="asinq"  # Requires calibration data
)
```

### 02. Quantization Workflow

**Basic Usage:**

```python
from transformers import AutoModelForCausalLM
from sinq import BaseQuantizeConfig, quantize_model

# 1. Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# 2. Create config
quant_cfg = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    method="sinq"
)

# 3. Quantize (no calibration needed)
quantized_model = quantize_model(model, quant_cfg)

# 4. Save
quantized_model.save_pretrained("./llama3-8b-sinq-4bit")

# 5. Load and use
loaded_model = AutoModelForCausalLM.from_pretrained(
    "./llama3-8b-sinq-4bit",
    device_map="cuda"
)
```

**With torch.compile (PyTorch backend):**

```python
# Optimize inference
loaded_model.forward = torch.compile(
    loaded_model.forward,
    dynamic=True,
    fullgraph=False,
    backend="inductor",
    mode="reduce-overhead",
)

# First run: compilation
outputs = loaded_model.generate(inputs, max_length=100)

# Subsequent runs: fast
outputs = loaded_model.generate(inputs, max_length=100)
```
