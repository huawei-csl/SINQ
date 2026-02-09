import os

import argparse
import gc
import types
import sys
import subprocess
import re
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from sinq.sinkhorn import sinkhorn_log
from sinq.awq import (
    get_calib_dataset,
    get_simple_calibration_data,
    collect_activations_blockwise,
    compute_awq_scale,
)


# ============================================================================
# Bash Command Integration for llama.cpp GGUF Conversion and Quantization
# ============================================================================

def convert_hf_to_gguf(model_path, output_gguf_path, outtype="f16"):
    """
    Convert HuggingFace model to GGUF format using llama.cpp conversion script.

    Args:
        model_path: Path to the HuggingFace model directory
        output_gguf_path: Path where the GGUF model will be saved
        outtype: Output type (f16, f32, bf16, q8_0, etc.)

    Returns:
        bool: True if successful, False otherwise
    """
    convert_script = "./llama.cpp/convert_hf_to_gguf.py"

    # Check if conversion script exists
    if not os.path.exists(convert_script):
        print(f"ERROR: Conversion script not found at {convert_script}")
        return False

    # Build command
    cmd = [
        "python3", convert_script,
        model_path,
        "--outfile", output_gguf_path,
        "--outtype", outtype
    ]

    print(f"Converting HF model to GGUF {outtype}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"ERROR during conversion:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

        print(f"✓ Successfully converted to GGUF: {output_gguf_path}")
        return True

    except subprocess.TimeoutExpired:
        print("ERROR: Conversion timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        return False


def quantize_gguf_with_llama_cpp(input_gguf_path, output_gguf_path, quant_type="Q4_0"):
    """
    Quantize a GGUF model using llama.cpp quantize tool.

    Args:
        input_gguf_path: Path to the input GGUF model (usually F16)
        output_gguf_path: Path where the quantized GGUF model will be saved
        quant_type: Quantization type (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_M, Q5_K_M, etc.)

    Returns:
        bool: True if successful, False otherwise
    """
    quantize_bin = "./llama.cpp/build/bin/llama-quantize"

    # Check if quantize binary exists
    if not os.path.exists(quantize_bin):
        print(f"ERROR: Quantize binary not found at {quantize_bin}")
        print("You may need to build llama.cpp first")
        return False

    # Build command
    cmd = [
        quantize_bin,
        input_gguf_path,
        output_gguf_path,
        quant_type
    ]

    print(f"Quantizing GGUF model to {quant_type}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"ERROR during quantization:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

        print(f"✓ Successfully quantized to {quant_type}: {output_gguf_path}")
        print(f"Output from llama-quantize:\n{result.stdout}")
        return True

    except subprocess.TimeoutExpired:
        print("ERROR: Quantization timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"ERROR during quantization: {e}")
        return False


def compute_perplexity_with_llama_cpp(gguf_model_path, test_text_file):
    """
    Compute perplexity using llama.cpp's llama-perplexity tool.

    Args:
        gguf_model_path: Path to the GGUF model file
        test_text_file: Path to text file for perplexity computation

    Returns:
        float or None: Perplexity value if successful, None otherwise
    """
    perplexity_bin = "./llama.cpp/build/bin/llama-perplexity"

    # Check if perplexity binary exists
    if not os.path.exists(perplexity_bin):
        print(f"ERROR: llama-perplexity binary not found at {perplexity_bin}")
        print("You may need to build llama.cpp first")
        return None

    # Build command
    cmd = [
        perplexity_bin,
        "--model", gguf_model_path,
        "-f", test_text_file
    ]

    print(f"Computing perplexity...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"ERROR during perplexity computation:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None

        # Parse perplexity from output
        output = result.stdout + result.stderr
        for line in output.split('\n'):
            if 'PPL' in line or 'perplexity' in line.lower():
                # Try to extract the perplexity value
                match = re.search(r'PPL\s*[=:]\s*([0-9.]+)', line, re.IGNORECASE)
                if match:
                    ppl = float(match.group(1))
                    print(f"✓ Perplexity: {ppl:.4f}")
                    return ppl

        print("Warning: Could not parse perplexity from output")
        print(f"Full output:\n{output}")
        return None

    except subprocess.TimeoutExpired:
        print("ERROR: Perplexity computation timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"ERROR during perplexity computation: {e}")
        return None


def convert_and_quantize_workflow(hf_model_path, output_dir, quant_type="Q4_0", keep_fp16=False):
    """
    Complete workflow: Convert HF model to GGUF FP16, then quantize it.

    Args:
        hf_model_path: Path to the HuggingFace model directory
        output_dir: Directory where GGUF models will be saved
        quant_type: Quantization type (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_M, Q5_K_M, etc.)
        keep_fp16: If True, keep the intermediate FP16 GGUF file

    Returns:
        str or None: Path to the final quantized GGUF model if successful, None otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract model name from path
    model_name = os.path.basename(hf_model_path.rstrip('/'))

    # Define paths
    fp16_gguf_path = os.path.join(output_dir, f"{model_name}_fp16.gguf")
    quantized_gguf_path = os.path.join(output_dir, f"{model_name}_{quant_type}.gguf")

    print("="*80)
    print("GGUF Conversion and Quantization Workflow")
    print("="*80)
    print(f"Input HF Model: {hf_model_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Quantization: {quant_type}")
    print("="*80)

    # Step 1: Convert to GGUF FP16
    print("\n[Step 1/2] Converting to GGUF FP16...")
    if not convert_hf_to_gguf(hf_model_path, fp16_gguf_path, outtype="f16"):
        print("ERROR: Failed to convert to GGUF FP16")
        return None

    # Step 2: Quantize
    print("\n[Step 2/2] Quantizing to {quant_type}...")
    if not quantize_gguf_with_llama_cpp(fp16_gguf_path, quantized_gguf_path, quant_type):
        print(f"ERROR: Failed to quantize to {quant_type}")
        return None

    # Optionally remove FP16 intermediate file
    if not keep_fp16:
        print(f"\nRemoving intermediate FP16 file: {fp16_gguf_path}")
        try:
            os.remove(fp16_gguf_path)
        except Exception as e:
            print(f"Warning: Could not remove FP16 file: {e}")

    print("\n" + "="*80)
    print("✓ Workflow completed successfully!")
    print(f"Final quantized model: {quantized_gguf_path}")
    print("="*80)

    return quantized_gguf_path


# -------------------------------
# Helpers: Sinkhorn + block logic
# -------------------------------

def find_block(H, W, block):
    for i in range(W):
        if (W % (block + i) == 0):
            return block + i
        elif (W % (block - i) == 0):
            return block - i


def get_sink_scale(matrix_list, cat_dim=0, block=64, n_iter=4):
    W = torch.cat(matrix_list, dim=cat_dim).cuda()
    H, Wid = W.shape
    dtype = W.dtype
    W = W.float()

    if block <= 0:
        W_hat, mu1, mu2 = sinkhorn_log(W, n_iter)
    else:
        if Wid % block != 0:
            block = find_block(H, Wid, block)
        assert Wid % block == 0, 'block must divide W'
        n_w = Wid // block
        W = W.view(H, Wid // block, block)
        W_batched = W.permute(1, 0, 2).contiguous().view(n_w, H, block)

        def process_block(mat):
            return sinkhorn_log(mat, n_iter)

        W_hat, mu1, mu2 = torch.vmap(process_block, randomness='different')(W_batched)

    mu1 = mu1 / mu1.median()
    return mu1.view(-1).cuda().to(dtype)


def get_sink_scale_with_awq(matrix_list, awq_scale=None, cat_dim=0, block=64, n_iter=4):
    """
    Compute sinkhorn scale with optional AWQ scale integration.

    This combines the sinkhorn normalization with AWQ activation-aware scaling.
    Following the approach in dual_shift.py:
    1. FIRST: Apply sinkhorn on the ORIGINAL weights
    2. SECOND: Adjust the resulting scale by AWQ factors

    The AWQ scale modifies the column scaling to account for activation importance.

    Args:
        matrix_list: List of weight matrices to concatenate
        awq_scale: AWQ scaling factors (computed from activations)
        cat_dim: Dimension along which to concatenate matrices
        block: Block size for tiled sinkhorn
        n_iter: Number of sinkhorn iterations

    Returns:
        Combined scale factors incorporating both sinkhorn and AWQ
    """
    W = torch.cat(matrix_list, dim=cat_dim).cuda()
    H, Wid = W.shape
    dtype = W.dtype
    W = W.float()

    if block <= 0:
        W_hat, mu1, mu2 = sinkhorn_log(W, n_iter)
    else:
        if Wid % block != 0:
            block = find_block(H, Wid, block)
        assert Wid % block == 0, 'block must divide W'
        n_w = Wid // block
        W = W.view(H, Wid // block, block)
        W_batched = W.permute(1, 0, 2).contiguous().view(n_w, H, block)

        def process_block(mat):
            return sinkhorn_log(mat, n_iter)

        W_hat, mu1, mu2 = torch.vmap(process_block, randomness='different')(W_batched)

    mu1 = mu1 / mu1.median()

    mu1 = mu1.view(-1)

    if awq_scale is not None:
        awq_scale = awq_scale.to(mu1.device).to(mu1.dtype)
        if awq_scale.shape[0] != mu1.shape[0]:
            # Resize awq_scale to match mu1 if needed
            if awq_scale.shape[0] > mu1.shape[0]:
                awq_scale = awq_scale[:mu1.shape[0]]
            else:
                awq_scale = F.pad(awq_scale, (0, mu1.shape[0] - awq_scale.shape[0]), value=1.0)
        mu1 = mu1 / awq_scale

    return mu1.cuda().to(dtype)

# -------------------------------
# Model traversal helper
# -------------------------------

def get_core_layers_module(model):
    """
    Walk down `.model` attributes until we find something with `.layers`.
    Works for plain HF CausalLM and AWQ wrappers.
    """
    core = model
    for _ in range(4):  # small safety bound
        if hasattr(core, "layers"):
            return core
        if hasattr(core, "model"):
            core = core.model
        else:
            break
    raise AttributeError("Could not find `.layers` module in model hierarchy")


def get_layer_prefix(model):
    """
    Get the prefix string for layer naming (e.g., 'model.layers').
    """
    core = model
    parts = []
    for _ in range(4):
        if hasattr(core, "layers"):
            parts.append("layers")
            return ".".join(parts[:-1]) if len(parts) > 1 else "model.layers"
        if hasattr(core, "model"):
            parts.append("model")
            core = core.model
        else:
            break
    return "model.layers"


# -------------------------------
# AWQ Scale Computation
# -------------------------------

def compute_awq_scales_for_model(model, tokenizer, activation_cache=None,
                                  group_size=64, num_alphas=20, method='sinq', nbits=4):
    """
    Compute AWQ scales for all linear layers in the model.

    Args:
        model: The model to compute scales for
        tokenizer: Tokenizer for calibration data
        activation_cache: Pre-computed activations (if None, will be collected)
        group_size: Group size for AWQ scale computation
        num_alphas: Number of alpha values to search over
        method: Quantization method ('sinq', 'awq+sinq', etc.)

    Returns:
        Dictionary mapping layer names to AWQ scale tensors
    """
    print("\n" + "="*60)
    print("Computing AWQ Scales")
    print("="*60)

    # Collect activations if not provided
    if activation_cache is None:
        print("Collecting calibration data...")
        try:
            calib_data = get_calib_dataset(tokenizer, n_samples=32, max_seq_len=512)
        except Exception as e:
            print(f"Failed to load pile dataset, using simple calibration: {e}")
            calib_data = get_simple_calibration_data(tokenizer, block_size=256)

        print(f"Collected {len(calib_data)} calibration samples")

        print("Collecting activations (blockwise for memory efficiency)...")
        activation_cache = collect_activations_blockwise(
            model, calib_data, num_samples=min(32, len(calib_data)), device="cuda"
        )

    # Compute AWQ scales for each linear layer
    awq_scales = {}
    min_max = [0, 2**nbits - 1]

    for name, module in tqdm(model.named_modules(), desc="Computing AWQ scales"):
        if isinstance(module, nn.Linear):
            if name not in activation_cache:
                continue

            activations = activation_cache[name]
            weights = module.weight.data.clone()

            # Compute AWQ scale using grid search
            scale = compute_awq_scale(
                weights=weights,
                activations=activations,
                min_max=min_max,
                tile=group_size,
                num_alphas=num_alphas,
                use_weightscale=True,
                method=method
            )

            awq_scales[name] = scale.cpu()

    print(f"Computed AWQ scales for {len(awq_scales)} layers")
    return awq_scales


# -------------------------------
# Pre-SINQ + AWQ for Qwen3
# -------------------------------

def absorb_sink_scale_qwen_awq(model, awq_scales=None, normalize_outproj=False,
                                n_gqa_groups=2, group_size=64, n_iter=4):
    """
    Apply pre-SINQ with AWQ scale integration for Qwen3 models.
    """
    print('Im applying the absorbing', flush=True)
    core = get_core_layers_module(model)
    layer_prefix = get_layer_prefix(model)

    for layer_idx, layer in tqdm(enumerate(core.layers), desc="Pre-SINQ+AWQ computation (Qwen3)"):
        dev = layer.input_layernorm.weight.device
        dtype = layer.input_layernorm.weight.dtype

        # Helper to get AWQ scale for a specific layer
        def get_awq_scale(sublayer_name):
            if awq_scales is None:
                return None
            full_name = f"{layer_prefix}.layers.{layer_idx}.{sublayer_name}"
            return awq_scales.get(full_name, None)

        # Optional: normalize attention out-proj (GQA-aware)
        if normalize_outproj:
            n_group = n_gqa_groups
            oOut, oIn = layer.self_attn.o_proj.weight.shape

            awq_scale_o = get_awq_scale("self_attn.o_proj")
            t_o = get_sink_scale_with_awq(
                [layer.self_attn.o_proj.weight.data.reshape(n_group * oOut, -1)],
                awq_scale=awq_scale_o,
                block=group_size, n_iter=n_iter
            )
            layer.self_attn.v_proj.weight.data = torch.matmul(
                torch.diag(t_o),
                layer.self_attn.v_proj.weight.data.cuda()
            ).to(dev)
            t_o = torch.cat([t_o] * n_group)
            layer.self_attn.o_proj.weight.data = torch.matmul(
                layer.self_attn.o_proj.weight.data.cuda(),
                torch.diag(1 / t_o)
            ).to(dev)

        # MLP down / up with AWQ
        awq_scale_down = get_awq_scale("mlp.down_proj")
        print(f'The AWQ scales for down proj are: {awq_scale_down}', flush=True)
        t_d = get_sink_scale_with_awq(
            [layer.mlp.down_proj.weight.data],
            awq_scale=awq_scale_down,
            block=group_size, n_iter=n_iter
        )
        layer.mlp.down_proj.weight.data = torch.matmul(
            layer.mlp.down_proj.weight.data.cuda(), torch.diag(1 / t_d)
        ).to(dev)
        layer.mlp.up_proj.weight.data = torch.matmul(
            torch.diag(t_d),
            layer.mlp.up_proj.weight.data.cuda()
        ).to(dev)

        # QKV with AWQ
        awq_scale_q = get_awq_scale("self_attn.q_proj")
        awq_scale_k = get_awq_scale("self_attn.k_proj")
        awq_scale_v = get_awq_scale("self_attn.v_proj")

        # Average the AWQ scales for concatenated QKV
        if awq_scale_q is not None and awq_scale_k is not None and awq_scale_v is not None:
            awq_scale_qkv = (awq_scale_q + awq_scale_k + awq_scale_v) / 3
        else:
            awq_scale_qkv = None

        t_qkv = get_sink_scale_with_awq(
            [layer.self_attn.q_proj.weight.data,
             layer.self_attn.k_proj.weight.data,
             layer.self_attn.v_proj.weight.data],
            awq_scale=awq_scale_qkv,
            block=group_size, n_iter=n_iter
        )
        layer.input_layernorm.weight.data = (
            layer.input_layernorm.weight.data.cuda() * t_qkv.view(-1)
        ).to(dev)
        layer.self_attn.q_proj.weight.data = torch.matmul(
            layer.self_attn.q_proj.weight.data.cuda(), torch.diag(1 / t_qkv)
        ).to(dev)
        layer.self_attn.k_proj.weight.data = torch.matmul(
            layer.self_attn.k_proj.weight.data.cuda(), torch.diag(1 / t_qkv)
        ).to(dev)
        layer.self_attn.v_proj.weight.data = torch.matmul(
            layer.self_attn.v_proj.weight.data.cuda(), torch.diag(1 / t_qkv)
        ).to(dev)

        # gate + up with AWQ
        awq_scale_gate = get_awq_scale("mlp.gate_proj")
        awq_scale_up = get_awq_scale("mlp.up_proj")

        if awq_scale_gate is not None and awq_scale_up is not None:
            awq_scale_gu = (awq_scale_gate + awq_scale_up) / 2
        else:
            awq_scale_gu = None

        t_gu = get_sink_scale_with_awq(
            [layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data],
            awq_scale=awq_scale_gu,
            block=group_size, n_iter=n_iter
        )
        layer.mlp.gate_proj.weight.data = torch.matmul(
            layer.mlp.gate_proj.weight.data.cuda(), torch.diag(1 / t_gu)
        ).to(dev)
        layer.mlp.up_proj.weight.data = torch.matmul(
            layer.mlp.up_proj.weight.data.cuda(), torch.diag(1 / t_gu)
        ).to(dev)
        layer.post_attention_layernorm.weight.data = (
            layer.post_attention_layernorm.weight.data.cuda() * t_gu.view(-1)
        ).to(dev)


def pre_sinq_qwen3_awq(model, awq_scales=None, normalize_outproj=False,
                        n_gqa_groups=2, n_repeat=3, group_size=64, n_iter=4):
    """
    Apply pre-SINQ with AWQ for Qwen3 models.
    """
    for _ in range(n_repeat):
        absorb_sink_scale_qwen_awq(
            model,
            awq_scales=awq_scales,
            normalize_outproj=normalize_outproj,
            n_gqa_groups=n_gqa_groups,
            group_size=group_size,
            n_iter=n_iter,
        )


# -------------------------------
# Pre-SINQ + AWQ for DeepSeek V2 Lite
# -------------------------------

def absorb_sink_scale_DSlite_awq(model, awq_scales=None, normalize_outproj=False,
                                  group_size=64, n_iter=4, normalize_downproj=True,
                                  normalize_gu=True):
    """
    Apply pre-SINQ with AWQ scale integration for DeepSeek V2 Lite models.
    """
    first = True
    core = get_core_layers_module(model)
    layer_prefix = get_layer_prefix(model)

    for layer_idx, layer in tqdm(enumerate(core.layers), desc="Pre-SINQ+AWQ computation (DS Lite)"):
        dev = layer.input_layernorm.weight.device
        dtype = layer.input_layernorm.weight.dtype

        # Helper to get AWQ scale for a specific layer
        def get_awq_scale(sublayer_name):
            if awq_scales is None:
                return None
            full_name = f"{layer_prefix}.layers.{layer_idx}.{sublayer_name}"
            return awq_scales.get(full_name, None)

        # Attention: Q + KV-A with AWQ
        awq_scale_q = get_awq_scale("self_attn.q_proj")
        awq_scale_kva = get_awq_scale("self_attn.kv_a_proj_with_mqa")

        if awq_scale_q is not None and awq_scale_kva is not None:
            awq_scale_qkv = (awq_scale_q + awq_scale_kva) / 2
        else:
            awq_scale_qkv = None

        t_qkv = get_sink_scale_with_awq(
            [layer.self_attn.q_proj.weight.data,
             layer.self_attn.kv_a_proj_with_mqa.weight.data],
            awq_scale=awq_scale_qkv,
            block=group_size, n_iter=n_iter
        )
        layer.input_layernorm.weight.data = (
            layer.input_layernorm.weight.data.cuda() * t_qkv.view(-1)
        ).to(dev)
        layer.self_attn.q_proj.weight.data = torch.matmul(
            layer.self_attn.q_proj.weight.data.cuda(), torch.diag(1 / t_qkv)
        ).to(dev)
        layer.self_attn.kv_a_proj_with_mqa.weight.data = torch.matmul(
            layer.self_attn.kv_a_proj_with_mqa.weight.data.cuda(), torch.diag(1 / t_qkv)
        ).to(dev)

        # KV-B with AWQ
        awq_scale_kvb = get_awq_scale("self_attn.kv_b_proj")
        t_kvb = get_sink_scale_with_awq(
            [layer.self_attn.kv_b_proj.weight.data],
            awq_scale=awq_scale_kvb,
            block=group_size, n_iter=n_iter
        )
        layer.self_attn.kv_a_layernorm.weight.data = (
            layer.self_attn.kv_a_layernorm.weight.data.cuda() * t_kvb.view(-1)
        ).to(dev)
        layer.self_attn.kv_b_proj.weight.data = torch.matmul(
            layer.self_attn.kv_b_proj.weight.data.cuda(), torch.diag(1 / t_kvb)
        ).to(dev)

        # Optional out-proj normalization with AWQ
        if normalize_outproj:
            awq_scale_o = get_awq_scale("self_attn.o_proj")
            t_o = get_sink_scale_with_awq(
                [layer.self_attn.o_proj.weight.data],
                awq_scale=awq_scale_o,
                block=group_size, n_iter=n_iter
            )
            layer.self_attn.o_proj.weight.data = torch.matmul(
                layer.self_attn.o_proj.weight.data, torch.diag(1 / t_o)
            ).to(dev)
            layer.self_attn.kv_b_proj.weight.data[:t_o.view(-1).shape[0]] = torch.matmul(
                torch.diag(t_o),
                layer.self_attn.kv_b_proj.weight.data.cuda()[:t_o.view(-1).shape[0]]
            ).to(dev)

        # MLP + MoE
        if first:
            first = False
            # dense MLP in first layer
            if normalize_downproj:
                awq_scale_down = get_awq_scale("mlp.down_proj")
                t_d = get_sink_scale_with_awq(
                    [layer.mlp.down_proj.weight.data],
                    awq_scale=awq_scale_down,
                    block=group_size, n_iter=n_iter
                )
                layer.mlp.down_proj.weight.data = torch.matmul(
                    layer.mlp.down_proj.weight.data.cuda(),
                    torch.diag(1 / t_d)
                ).to(dev)
                layer.mlp.up_proj.weight.data = torch.matmul(
                    torch.diag(t_d),
                    layer.mlp.up_proj.weight.data.cuda()
                ).to(dev)

            if normalize_gu:
                awq_scale_gate = get_awq_scale("mlp.gate_proj")
                awq_scale_up = get_awq_scale("mlp.up_proj")

                if awq_scale_gate is not None and awq_scale_up is not None:
                    awq_scale_gu = (awq_scale_gate + awq_scale_up) / 2
                else:
                    awq_scale_gu = None

                t_gu = get_sink_scale_with_awq(
                    [layer.mlp.up_proj.weight.data, layer.mlp.gate_proj.weight.data],
                    awq_scale=awq_scale_gu,
                    block=group_size, n_iter=n_iter
                )
                layer.post_attention_layernorm.weight.data = (
                    layer.post_attention_layernorm.weight.data.cuda() * t_gu.view(-1)
                ).to(dev)
                layer.mlp.gate_proj.weight.data = torch.matmul(
                    layer.mlp.gate_proj.weight.data.cuda(), torch.diag(1 / t_gu)
                ).to(dev)
                layer.mlp.up_proj.weight.data = torch.matmul(
                    layer.mlp.up_proj.weight.data.cuda(), torch.diag(1 / t_gu)
                ).to(dev)
        else:
            # MoE experts on subsequent layers
            if normalize_downproj:
                for i in range(64):
                    expert_name = f"mlp.experts.{i}.down_proj"
                    awq_scale_exp = get_awq_scale(expert_name)
                    t_d = get_sink_scale_with_awq(
                        [layer.mlp.experts[i].down_proj.weight.data],
                        awq_scale=awq_scale_exp,
                        block=group_size, n_iter=n_iter
                    )
                    layer.mlp.experts[i].down_proj.weight.data = torch.matmul(
                        layer.mlp.experts[i].down_proj.weight.data.cuda(),
                        torch.diag(1 / t_d)
                    ).to(dev)
                    layer.mlp.experts[i].up_proj.weight.data = torch.matmul(
                        torch.diag(t_d),
                        layer.mlp.experts[i].up_proj.weight.data.cuda()
                    ).to(dev)

            if normalize_gu:
                weight_list = [
                    layer.mlp.experts[i].up_proj.weight.data for i in range(64)
                ]
                weight_list = weight_list + [
                    layer.mlp.experts[i].gate_proj.weight.data for i in range(64)
                ]

                # Average AWQ scales for all experts
                awq_scale_gu = None
                awq_scales_list = []
                for i in range(64):
                    up_scale = get_awq_scale(f"mlp.experts.{i}.up_proj")
                    gate_scale = get_awq_scale(f"mlp.experts.{i}.gate_proj")
                    if up_scale is not None:
                        awq_scales_list.append(up_scale)
                    if gate_scale is not None:
                        awq_scales_list.append(gate_scale)

                if awq_scales_list:
                    awq_scale_gu = torch.stack(awq_scales_list).mean(dim=0)

                t_gu = get_sink_scale_with_awq(
                    weight_list,
                    awq_scale=awq_scale_gu,
                    block=group_size, n_iter=n_iter
                )

                layer.post_attention_layernorm.weight.data = (
                    layer.post_attention_layernorm.weight.data.cuda() * t_gu.view(-1)
                ).to(dev)

                for i in range(64):
                    layer.mlp.experts[i].gate_proj.weight.data = torch.matmul(
                        layer.mlp.experts[i].gate_proj.weight.data.cuda(),
                        torch.diag(1 / t_gu)
                    ).to(dev)
                    layer.mlp.experts[i].up_proj.weight.data = torch.matmul(
                        layer.mlp.experts[i].up_proj.weight.data.cuda(),
                        torch.diag(1 / t_gu)
                    ).to(dev)


def pre_sinq_dslite_awq(model, awq_scales=None, group_size=64, n_iter=4, n_repeat=1):
    """
    Apply pre-SINQ with AWQ for DeepSeek V2 Lite models.
    """
    for _ in range(n_repeat):
        absorb_sink_scale_DSlite_awq(
            model,
            awq_scales=awq_scales,
            group_size=group_size,
            n_iter=n_iter,
        )


# -------------------------------
# DeepSeek AWQ rotary patch
# -------------------------------

def patch_deepseek_rotary(model):
    """
    Attach a model-level `.rotary_emb(x)` to the inner DeepseekV2Model that
    AutoAWQ expects as `model.model.rotary_emb(...)`.
    We infer seq_len from x.shape[-2].
    """
    core = model
    chain_types = [type(core).__name__]
    for _ in range(3):
        if hasattr(core, "model"):
            core = core.model
            chain_types.append(type(core).__name__)
        else:
            break

    print("AWQ / HF model chain:", " -> ".join(chain_types))
    print("Patch target type:", type(core).__name__)

    def rotary_emb_proxy(self, x, *args, **kwargs):
        attn = self.layers[0].self_attn
        if x.dim() >= 2:
            seq_len = x.shape[-2]
        else:
            seq_len = x.shape[0]
        return attn.rotary_emb(x, seq_len=int(seq_len))

    if not hasattr(core, "rotary_emb"):
        core.rotary_emb = types.MethodType(rotary_emb_proxy, core)
        print("Attached rotary_emb proxy to", type(core).__name__)
    else:
        print("core already had rotary_emb:", core.rotary_emb)

    try:
        inner = model.model.model
        print("Has inner rotary_emb?", hasattr(inner, "rotary_emb"))
    except Exception as e:
        print("Inner model sanity check failed:", e)


# -------------------------------
# KL tracker
# -------------------------------

class KLTracker:
    def __init__(self, calib_string='the'):
        # tokenizer must exist in global scope (set in main)
        global tokenizer
        self.p_logits = None
        self.inputs = tokenizer(calib_string, return_tensors="pt").to("cuda")

    def get_baseline_logits(self, model):
        model.eval()
        with torch.no_grad():
            self.p_logits = model(**self.inputs).logits.detach()

    def get_KL_to_baseline(self, model):
        print('computing KL div')
        if self.p_logits is None:
            raise ValueError("Baseline logits not set. Call get_baseline_logits first.")

        q_logits = model(**self.inputs).logits

        log_prob_p = F.log_softmax(self.p_logits.to(q_logits.device).double(), dim=-1)
        log_prob_q = F.log_softmax(q_logits.double(), dim=-1)

        return F.kl_div(
            input=log_prob_q,
            target=log_prob_p,
            reduction='batchmean',
            log_target=True
        ).mean()


# -------------------------------
# Calibration prompts
# -------------------------------

prompt1 = """- Fiction: "In a hidden valley where time moved slower, an old painter discovered a brush that could bring his creations to life. His first stroke awoke something unexpected..."
- News: "A rare celestial event—a triple conjunction of Jupiter, Saturn, and Mars—will be visible tonight for the first time in over 200 years. Astronomers urge skywatchers not to miss..."
- Code: `const countVowels = (str) => [...str].filter(c => "aeiou".includes(c.toLowerCase())).length;
console.log(countVowels("Hello, world!"));`
- Math: A car travels 240 km in 3 hours at constant speed. If it then accelerates by 20 km/h for the next 2 hours, what's the total distance traveled?
- Facts: "The Great Wall of China is approximately 21,196 km long. However, contrary to myth, it cannot be seen from space with the naked eye..."
- Fiction: "The last tree in the desert city whispered secrets to those who listened. When a young girl finally understood its language, she discovered it held the blueprint to regrow the entire forest..."
- News: "Scientists develop biodegradable battery that decomposes in soil after 30 days, offering potential solution to electronic waste pollution..."
- Code: `def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b`
- Math: Find the area of a triangle with vertices at (2,3), (5,7), and (9,4) using the determinant formula
- Facts: "Octopuses have three hearts and blue blood. Two hearts pump blood to the gills while the third circulates it to the rest of the body..."
"""

prompt2 = """- Fiction: "When the stars aligned, a librarian in Prague found every book in the library rearranged into an unknown language—except one, which bore her name on the cover..."
- News: "New legislation bans single-use plastics in the European Union, with critics arguing the policy doesn't address industrial waste, while proponents hail it as a critical first step..."
- Code: `import numpy as np
arr = np.array([1, 2, 3])
print(arr * 2)`
- Math: (14.6 * 3.2) - (5.9 ** 2) + (18 / 1.8) =
- Facts: "The male seahorse carries and gives birth to its young. Females deposit eggs into the male's pouch, where they are fertilized and nurtured until birth..."
- Fiction: "Every full moon, the antique shop's items would rearrange themselves. The owner kept meticulous records until he noticed a pattern that predicted future events with uncanny accuracy..."
- News: "Global coral bleaching event declared as ocean temperatures reach record highs, threatening marine ecosystems worldwide..."
- Code: `from collections import defaultdict
d = defaultdict(int)
for word in text.split():
    d[word] += 1`
- Math: Solve the quadratic equation: 2x² - 7x + 3 = 0
- Facts: "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible..."
"""

prompt3 = """- Fiction: "A lighthouse keeper on a remote island noticed the beacon dimming each night, replaced by a faint chorus of voices singing in unison—until one evening, they called his name..."
- News: "AI-designed proteins could revolutionize medicine, with researchers announcing the creation of molecules that target previously 'undruggable' diseases..."
- Code: `class Cat:
  def __init__(self, name):
    self.name = name
  def speak(self):
    return f"{self.name} says 'Meow!'"`
- Math: If 3x + 5 = 20, what is the value of x² - 2x?
- Facts: "Venus rotates backward compared to most planets in the solar system, meaning its sun rises in the west and sets in the east..."
- Fiction: "The clockmaker's final creation could manipulate time itself. But when he tried to undo his greatest regret, he discovered why some moments were meant to remain unchanged..."
- News: "Breakthrough in quantum computing: Researchers achieve quantum supremacy with 128-qubit processor, solving problems previously thought impossible..."
- Code: `const debounce = (func, delay) => {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), delay);
  };
};`
- Math: Calculate the volume of a sicut sphere with radius 5 cm (V = 4/3πr³)
- Facts: "A single strand of spider silk is stronger than steel of the same diameter and can stretch up to five times its length without breaking..."
"""

prompt4 = """- Fiction: "In a village where every resident shared the same dream nightly, a child was born who dreamed of nothing—until the others' dreams began vanishing one by one..."
- News: "SpaceX successfully landed a reusable rocket on its tenth flight, setting a new milestone for cost efficiency in space exploration..."
- Code: `list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, [1, 2, 3, 4])))`
- Math: (7! / (4! * 3!)) + (√144 * 2³) =
- Facts: "A group of flamingos is called a 'flamboyance.' These birds often balance on one leg to conserve body heat..."
- Fiction: "The museum's newest exhibit—a perfectly preserved Victorian doll—began appearing in visitors' dreams with whispered warnings that always came true the next day..."
- News: "Researchers discover new species of deep-sea fish that glows with bioluminescent patterns never before documented in marine biology..."
- Code: `using System.Linq;
var evenNumbers = numbers.Where(n => n % 2 == 0).ToList();`
- Math: Find the derivative of f(x) = 3x⁴ - 2x³ + 7x - 5
- Facts: "The shortest war in history lasted only 38 minutes. It occurred in 1896 between Britain and Zanzibar when the Sultan's forces surrendered after a brief naval bombardment..."
"""

prompt5 = """- Fiction: "A detective specializing in 'impossible crimes' received a letter postmarked from 1942. The handwriting matched her own—but she hadn't been born yet..."
- News: "Archaeologists uncovered a 1,500-year-old mosaic beneath a vineyard in Italy, depicting scenes from Greek mythology in near-perfect condition..."
- Code: `String reverse(String s) {
  return new StringBuilder(s).reverse().toString();
}`
- Math: (log₁₀1000 * 5²) - (e³ / ln(20)) ≈ (round to two decimal places)
- Facts: "Sharks have been around longer than trees. The earliest shark fossils date back 400 million years, while trees appeared roughly 350 million years ago..."
- Fiction: "The bookstore that only appeared during rainstorms contained volumes written by authors from parallel universes. One rainy Tuesday, a customer found a book with their life story—but with a different ending..."
- News: "World's first successful transplant of 3D-printed functional organ performed, marking major advancement in regenerative medicine..."
- Code: `function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}`
- Math: Calculate the compound interest on $10,000 at 5% annual rate compounded quarterly for 3 years
- Facts: "The human nose can detect over 1 trillion different scents, far more than the previously believed 10,000 scents..."
"""

test_string = prompt1 + prompt2 + prompt3 + prompt4 + prompt5


# -------------------------------
# Main
# -------------------------------

if __name__ == '__main__':
    import sys
    sys.path.append("./SINQ/tests")
    from eval_my.evaluate_ import evaluate_model

    parser = argparse.ArgumentParser()
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model identifier (e.g. Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, deepseek-ai/DeepSeek-V2-Lite, etc.)"
    )

    parser.add_argument(
        "--temp_dir",
        type=str,
        default="./tmp_best_presinq_awq_models",
        help="Directory for temporary / final model storage"
    )

    # Boolean flags
    parser.add_argument(
        "--save_to_disk",
        action="store_true",
        help="Whether to save the quantized model to disk"
    )

    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Run only the baseline (no pre-SINQ / no quantization) evaluation"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation (ppl on WikiText-2) during hyperparam search"
    )

    parser.add_argument(
        "--disable_awq",
        action="store_true",
        help="Disable AWQ scaling (use pure SINQ without activation awareness)"
    )

    # GGUF quantization configuration
    parser.add_argument(
        "--quant_type",
        type=str,
        default="Q4_K_M",
        help="Quantization type for llama.cpp (Q4_0, Q4_K_M, Q5_K_M, Q6_K, etc.)"
    )

    parser.add_argument(
        "--keep_fp16_gguf",
        action="store_true",
        help="Keep intermediate FP16 GGUF file after quantization"
    )

    parser.add_argument(
        "--skip_gguf_conversion",
        action="store_true",
        help="Skip GGUF conversion/quantization (only save full-precision pre-SINQ model)"
    )

    # Quantization / grouping configuration
    parser.add_argument(
        "--group_size",
        type=int,
        default=64,
        help="The size of the groups for pre-SINQ / AWQ q_group_size"
    )

    parser.add_argument(
        "--nbits",
        type=int,
        default=4,
        help="Number of bits for AWQ weight quantization"
    )

    # AWQ-specific configuration
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="Number of calibration samples for AWQ scale computation"
    )

    parser.add_argument(
        "--num_alphas",
        type=int,
        default=20,
        help="Number of alpha values to search over for AWQ scale optimization"
    )

    parser.add_argument(
        "--awq_method",
        type=str,
        default="awq+sinq",
        help="AWQ method variant (awq+sinq, awq+sinq+l1, etc.)"
    )

    args = parser.parse_args()

    model_name = args.model_name
    save_to_disk = args.save_to_disk
    baseline_only = args.baseline_only
    validate = args.validate
    group_size = args.group_size
    nbits = args.nbits
    temp_dir = args.temp_dir
    quant_type = args.quant_type
    keep_fp16_gguf = args.keep_fp16_gguf
    skip_gguf_conversion = args.skip_gguf_conversion
    disable_awq = args.disable_awq
    num_calib_samples = args.num_calib_samples
    num_alphas = args.num_alphas
    awq_method = args.awq_method

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if disable_awq:
        print(f"Workflow: Pre-SINQ calibration (no AWQ) → Save FP model → Convert to GGUF → Quantize with llama.cpp ({quant_type})")
    else:
        print(f"Workflow: Pre-SINQ + AWQ (ASINQ) calibration → Save FP model → Convert to GGUF → Quantize with llama.cpp ({quant_type})")

    if skip_gguf_conversion:
        print("Note: GGUF conversion/quantization will be SKIPPED (--skip_gguf_conversion)")
    else:
        print(f"GGUF quantization will be applied after finding best pre-SINQ+AWQ config")

    # ---------------------------
    # Load baseline unquantized model + tokenizer
    # ---------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )

    if "deepseek" in model_name.lower():
        patch_deepseek_rotary(model)

    model.eval()

    # ---------------------------
    # Compute AWQ scales (if not disabled)
    # ---------------------------
    awq_scales = None
    if not disable_awq and not baseline_only:
        print("\n" + "="*80)
        print("Computing AWQ Activation Scales")
        print("="*80)

        # Get calibration data
        print("Loading calibration data...")
        try:
            calib_data = get_calib_dataset(tokenizer, n_samples=num_calib_samples, max_seq_len=512)
            print(f"Loaded {len(calib_data)} samples from pile-val-backup")
        except Exception as e:
            print(f"Failed to load pile dataset: {e}")
            print("Using simple calibration data instead...")
            calib_data = get_simple_calibration_data(tokenizer, block_size=256)
            print(f"Created {len(calib_data)} simple calibration samples")

        # Collect activations
        print("\nCollecting activations (blockwise for memory efficiency)...")
        activation_cache = collect_activations_blockwise(
            model, calib_data,
            num_samples=min(num_calib_samples, len(calib_data)),
            device="cuda"
        )

        # Compute AWQ scales
        awq_scales = compute_awq_scales_for_model(
            model, tokenizer,
            activation_cache=activation_cache,
            group_size=group_size,
            num_alphas=num_alphas,
            method=awq_method, 
            nbits=nbits
        )

        # Clear activation cache to free memory
        del activation_cache
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\nAWQ scales computed for {len(awq_scales)} layers")

    # ---------------------------
    # Hyperparam search for pre-SINQ + AWQ (if requested)
    # ---------------------------
    best_config = None
    best_ppl = None
    ppls = []

    if not baseline_only:
        # Create temporary text file with calibration strings for perplexity computation
        calib_text_file = os.path.join(temp_dir, "calibration_text.txt")
        config_text_file = os.path.join(temp_dir, "config_text.txt")
        os.makedirs(temp_dir, exist_ok=True)
        with open(calib_text_file, 'w', encoding='utf-8') as f:
            f.write(test_string)
        print(f"Created calibration text file: {calib_text_file}")

        print(model)

        configs = []
        for gs in [32, 64, 128]: # Modify if needed
            for n_iter in [2, 4, 8, 16, 32]: # Modify if needed
                for n_repeat in [1, 2]: # Modify if needed
                    configs.append({
                        'group_size': gs,
                        'n_iter': n_iter,
                        'n_repeat': n_repeat
                    })

        best_ppl = 1e12
        best_config = None

        for idx, config in enumerate(configs):
            print("\n" + "="*80)
            print(f"Configuration {idx+1}/{len(configs)}: {config}")
            if not disable_awq:
                print(f"AWQ Method: {awq_method}")
            print("="*80)

            # fresh unquantized model for this config
            model_q = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            if "deepseek" in model_name.lower():
                patch_deepseek_rotary(model_q)

            # pre-SINQ + AWQ with current config
            if "deepseek" in model_name.lower():
                pre_sinq_dslite_awq(model_q, awq_scales=awq_scales, **config)
            elif "qwen" in model_name.lower():
                print('Im absorbing the scales for Qwen model', flush=True)
                pre_sinq_qwen3_awq(model_q, awq_scales=awq_scales, **config)

            model_q.eval()

            # Save this pre-SINQ+AWQ model temporarily
            config_name = f"config_{idx}_gs{config['group_size']}_iter{config['n_iter']}_rep{config['n_repeat']}"
            if not disable_awq:
                config_name += "_awq"
            temp_model_dir = os.path.join(temp_dir, "temp_models", config_name)
            print(f"\nSaving pre-SINQ+AWQ model to {temp_model_dir}")
            model_q.save_pretrained(temp_model_dir)
            tokenizer.save_pretrained(temp_model_dir)

            # Convert to GGUF and quantize
            temp_gguf_dir = os.path.join(temp_dir, "temp_gguf", config_name)
            print(f"\nConverting to GGUF and quantizing...")
            quantized_gguf_path = convert_and_quantize_workflow(
                hf_model_path=temp_model_dir,
                output_dir=temp_gguf_dir,
                quant_type=quant_type,
                keep_fp16=False  # Don't keep FP16 for temp models
            )

            if quantized_gguf_path is None:
                print(f"ERROR: Failed to convert/quantize model for config {config}")
                print("Skipping this configuration...")
                ppls.append(float('inf'))

                # Clean up
                del model_q
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # Compute perplexity on the quantized GGUF model
            print(f"\nComputing perplexity on quantized GGUF model...")
            ppl = compute_perplexity_with_llama_cpp(quantized_gguf_path, "pile_val_5k.txt")

            if ppl is None:
                print(f"ERROR: Failed to compute perplexity for config {config}")
                print("Skipping this configuration...")
                ppls.append(float('inf'))
            else:
                ppls.append(ppl)
                print('~~~~' * 5)
                print(f'Perplexity = {ppl:.4f}')
                print(f"Config: {config}")
                with open(config_text_file, 'a', encoding='utf-8') as f:
                    f.write(str(config))
                    f.write(str(ppl))
                if not disable_awq:
                    print(f"AWQ Method: {awq_method}")

                if ppl < best_ppl:
                    best_ppl = ppl
                    best_config = config
                    print(f"✓ New best perplexity: {ppl:.4f}")

            # Optional: Run WikiText-2 validation if requested
            if validate:
                print(f"\nRunning WikiText-2 validation on PyTorch model...")
                results = evaluate_model(
                    model=model_q,
                    tokenizer=tokenizer,
                    tasks="",
                    eval_ppl='wikitext2',
                    batch_size=8
                )
                task_results = results['wikitext2']
                print(f"WikiText-2 PPL (PyTorch): {task_results}")

            # Clean up temporary files for this config
            print(f"\nCleaning up temporary files for config {idx}...")
            try:
                shutil.rmtree(temp_model_dir)
                shutil.rmtree(temp_gguf_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp files: {e}")

            # free memory
            del model_q
            torch.cuda.empty_cache()
            gc.collect()

        print('\n' + '='*80)
        print('HYPERPARAMETER SEARCH COMPLETE')
        print('='*80)
        print(f'Best Perplexity: {best_ppl:.4f}')
        print(f'Best Configuration: {best_config}')
        if not disable_awq:
            print(f'AWQ Method: {awq_method}')
        print(f'All Perplexities: {[f"{p:.4f}" if p != float("inf") else "FAILED" for p in ppls]}')
        print('='*80)

        # ---------------------
        # Build final pre-SINQ + AWQ + GGUF model using best_config
        # ---------------------
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        if "deepseek" in model_name.lower():
            patch_deepseek_rotary(model)

        if "deepseek" in model_name.lower():
            pre_sinq_dslite_awq(model, awq_scales=awq_scales, **best_config)
        elif "qwen" in model_name.lower():
            pre_sinq_qwen3_awq(model, awq_scales=awq_scales, **best_config)

        # Save the full-precision pre-SINQ+AWQ model
        if save_to_disk:
            fp_model_dir = os.path.join(temp_dir, "best_presinq_awq_fp")
            print(f'\nSaving full-precision pre-SINQ+AWQ model to {fp_model_dir}')
            model.save_pretrained(fp_model_dir)
            tokenizer.save_pretrained(fp_model_dir)
            print('Full-precision model saved')
            print(f'The best config for hyperparameters is: {best_config}')