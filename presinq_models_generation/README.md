# Pre-SINQ Quantization Runner

In this folder you can find scripts to:
1. Load a HuggingFace causal language model (e.g., Qwen3 or DeepSeek Lite),
2. Apply **Pre-SINQ** (Sinkhorn-based weight rescaling / absorption),
3. Quantize the model using **HQQ**, **AWQ** or any **GGUF** quantization method,
4. Optionally run a small hyperparameter search to find the best **Pre-SINQ** model for the selected quantization method. To evaluate the model it is possible to use a small test set (calibration_text.txt) or a larger one (pile_5K.txt).
5. Optionally validate configurations using **WikiText-2 perplexity**.

---

## Features

- ✅ Supports **Qwen3** family and **DeepSeek Lite** variants 
- ✅ Hyperparameter search over:
  - `n_iter` (Sinkhorn iterations)
  - `n_repeat` (number of Pre-SINQ passes)
  - `group_size` (block/group size used in scaling and quantization)
- ✅ Optional evaluation on WikiText-2 perplexity
- ✅ Optional saving of **Pre-SINQ** best model to disk

---
## Pre-requisite

1) Install SINQ library from the official repository
2) Install the repository that contains the code necessary to run the desired quantization strategy
3) Check that all the path in the script are correctly set
4) Create the pile_5k.txt file by using the ```convert_hf_dataset_to_txt.py```. You can do it by running
```bash
python convert_hf_dataset_to_txt.py \
--dataset mit-han-lab/pile-val-backup \
--split validation \
--max_rows 5000 \
--out pile_5k.txt
```

## Script Arguments

All the scripts expose the following command-line arguments:

| Argument | Type | Default | Description |
|--------|------|---------|-------------|
| `--model_name` | `str` | `Qwen/Qwen3-1.7B` | HuggingFace model identifier (e.g., `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`) |
| `--temp_dir` | `str` | `./presinq_models` | Directory where the processed model and tokenizer are saved (only used with `--save_to_disk`) |
| `--save_to_disk` | flag | `False` | Save the final Pre-SINQ model to disk |
| `--baseline_only` | flag | `False` | Run the baseline FP model only (no Pre-SINQ, no quantization) |
| `--validate` | flag | `False` | Evaluate perplexity on WikiText-2 (and track it during hyperparameter search) |
| `--group_size` | `int` | `64` | Group / block size used for Sinkhorn scaling |
| `--nbits` | `int` | `4` | Number of bits used for weight quantization |

### GGUF specific (in addition to the previous ones)

| Argument | Type | Default | Description |
|--------|------|---------|-------------|
| `--disable_awq` | flag | `False` | Disable the ASINQ strategy to obtain the Presinq model and apply the SINQ standard algorithm |
| `--quant_type` | `str` | `Q4_K_M` | GGUF Quantization method |
| `--awq_method` | `str` | `awq+sinq` | Specify which is the variant of ASINQ to apply |


### Notes

- If `--baseline_only` is set, Pre-SINQ and quantization are skipped.
- If `--validate` is enabled, the script evaluates perplexity on WikiText-2 after each configuration and for the final model.
- The Pre-SINQ hyperparameters `n_iter` (Sinkhorn iterations) and `n_repeat` (number of Pre-SINQ passes) are currently defined inside the script as part of the hyperparameter search grid.
- The script automatically selects the appropriate Pre-SINQ routine based on the model name (`Qwen` vs `deepseek`).

## Quick Start

>  **HQQ** 

Run baseline model evaluation only

```bash
python presinq_hqq.py \
  --model_name Qwen/Qwen3-1.7B \
  --baseline_only \
  --validate
```

Run Pre-SINQ with HQQ as quantization method to find the best Pre-SINQ model (the final model is in full precision)

```bash
python presinq_hqq.py \
  --model_name Qwen/Qwen3-1.7B \
  --group_size 64 \
  --nbits 4
```

Run Pre-SINQ with HQQ as quantization method to find the best Pre-SINQ model and save it

```bash
python presinq_hqq.py \
  --model_name Qwen/Qwen3-1.7B \
  --group_size 64 \
  --nbits 4 \
  --save_to_disk \
  --temp_dir /path/to/output_dir
```

>  **AWQ** 

Run baseline model evaluation only

```bash
python presinq_awq.py \
  --model_name Qwen/Qwen3-1.7B \
  --baseline_only \
  --validate
```

Run Pre-SINQ with AWQ as quantization method to find the best Pre-SINQ model (the final model is in full precision)

```bash
python presinq_awq.py \
  --model_name Qwen/Qwen3-1.7B \
  --group_size 64 \
  --nbits 4
```

Run Pre-SINQ with AWQ as quantization method to find the best Pre-SINQ model and save it

```bash
python presinq_awq.py \
  --model_name Qwen/Qwen3-1.7B \
  --group_size 64 \
  --nbits 4 \
  --save_to_disk \
  --temp_dir /path/to/output_dir
```

>  **GGUF** 

Run Pre-SINQ with GGUF as quantization method to find the best Pre-SINQ model (the final model is in full precision) and save it.
Please notice that the default setting is using the ASINQ algorithm, if you want to use the SINQ you can disable AWQ scales computation by adding ```--disable_awq```

```bash
python presinq_gguf.py \
  --model_name Qwen/Qwen3-1.7B \
  --group_size 64 \
  --quant_type Q4_K_S \
  --awq_method awq+sinq \
  --nbits 4 \
  --save_to_disk \
  --temp_dir /path/to/output_dir
```

## Usage of PreSINQ models

The PreSINQ models are models in full precision (fp16), they can be quantized and the quantization method should be the same of the one used in the hyperaparameters search for the PreSINQ model. 
This strategy allows to use the PreSINQ model in all popular inference framwork like Llama.cpp, SGLang, vllm and Hugging Face without the specific need of having SINQ quantization strategy already integrated in the framework.

## Release of GGUF PreSINQ models

Several GGUF PreSINQ models have already been released and are available on our [Hugging Face Hub](https://huggingface.co/huawei-csl). These models represent the best PreSINQ configurations for the quantization methods specified in their respective model cards.
However, as described in each model card, strong PreSINQ models can also be obtained using a faster, less compute-intensive PreSINQ script, which evaluates a smaller set of candidate configurations.
