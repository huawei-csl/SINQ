"""
written by SINQ authors
"""
import os
import argparse
from timeit import default_timer as timer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from torch.utils.data import DataLoader

import os
from functools import partial
from tqdm import tqdm
import numpy as np


import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# from patch_model_demo import quantize_model
use_sinq = True 
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def get_model(model_name, nbits=None, group_size=128, tiling_mode='1D', method='dual', axis=1, device='cpu'):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if nbits is None:
        return model, tokenizer

    quant_config = BaseQuantizeConfig(
        nbits=nbits, group_size=group_size, axis=axis, tiling_mode=tiling_mode,
        method=method
    )

    start_time = timer()
    AutoSINQHFModel.quantize_model(
        model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device=device
    )
    end_time = timer()
    duration = (end_time - start_time)
    print(f"Time to quantize model: {duration:.2f} seconds")

    print(
        "CUDA memory allocated and reserved (GB):",
        torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9  # in GB
    )
    memory_alloc = torch.cuda.memory_allocated() / 1e9
    print(model)
    return model, tokenizer, memory_alloc

if __name__ == '__main__':
    import os
    from eval_my.evaluate_ import evaluate_model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen3-1.7B"
    ) # can be e.g. Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B
    parser.add_argument("--device", type=str, default="cuda:0") # fixed
    parser.add_argument("--nbits", type=int, default=4) # can be [3,4,5,8]
    parser.add_argument("--group_size", type=int, default=64) # can be [32, 64, 128, 256]
    parser.add_argument("--axis", type=int, default=1) # fixed
    parser.add_argument("--batch_size", type=int, default=1) # fixed
    parser.add_argument("--dataset_name", type=str, default='wikitext2') # fixed to this
    parser.add_argument("--tiling_mode", type=str, default='1D') # can be 1D, 2D
    parser.add_argument("--method", type=str, default='sinq') # can contain flags 'sinq', 'noz', 'hqq', 'quantAux', 'awq', 'l1'

    args = parser.parse_args()
    model, tokenizer, memory = get_model(args.model_name, nbits=args.nbits, group_size=args.group_size, 
                                         axis=args.axis, device=args.device, tiling_mode=args.tiling_mode,
                                         method= args.method+"_nogemlite")

    model = torch.compile(model)
    print(f'alloc memory (GB): {memory:.2f}')

    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks="",
        eval_ppl=args.dataset_name,
        batch_size=8
    )
    task_results = results[args.dataset_name] #perplexity / ppl

    print(args.model_name, args.nbits, args.method, task_results, memory)


