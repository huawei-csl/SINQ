import os
import numpy as np
import torch
import json
import datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk
import random
from itertools import chain
from transformers import DataCollatorForLanguageModeling, default_data_collator
import transformers
import tqdm

num_proc=64
use_cache = False

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_qat_dataset(name, tokenizer, model="llama3", seed="random"):
    cache_file=f'cache/{name}_{model}_{seed}.pt'
    if use_cache:
        try:
            return torch.load(cache_file), default_data_collator
        except Exception:
            pass
    if name == "wikitext2":
        data, data_collator = get_wikitext2_train(tokenizer=tokenizer)
    elif name == "c4":
        data, data_collator = get_c4_train(tokenizer=tokenizer)
    elif name == "c4_wiki":
        data, data_collator = get_c4_wiki_train(tokenizer=tokenizer)
    elif name == "tulu":
        data, data_collator = get_tulu_train(tokenizer=tokenizer)
    elif name == "alpaca_clean":
        data, data_collator = get_alpaca_clean_train(tokenizer=tokenizer)
    elif name == "wikitext2_test":
        data, data_collator = get_wikitext2_test4train(tokenizer=tokenizer)

    if use_cache:
        directory='/'.join(cache_file.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(data, cache_file)

    return data, data_collator

def get_eval_loaders(name, tokenizer):
    if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
                print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
            except AttributeError:
                pass
                print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(tokenizer)
        return get_ptb(tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(tokenizer)
        return get_c4(tokenizer)


def get_wikitext2_train(tokenizer, seed=0, seqlen=2048):
    dataset = load_from_disk('../eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/train')

    wikitext_dataset = datasets.Dataset.from_dict(
            {
                "text": [
                    "\n\n".join(dataset["text"])
                ],
            },
        )

    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )
    column_names = list(wikitext_dataset.features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset, default_data_collator

def get_wikitext2_test4train(tokenizer, seed=0, seqlen=2048):
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    wikitext_dataset = datasets.Dataset.from_dict(
            {
                "text": [
                    "\n\n".join(dataset["text"])
                ],
            },
        )

    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )
    column_names = list(wikitext_dataset.features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset, default_data_collator

def get_c4_train(tokenizer, seed=0, seqlen=2048):
    raw_datasets = load_from_disk('../eval_my/ppl_datasets/allenai/c4/allenai--c4')
    _wikitext_dataset = load_from_disk('../eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/test')
    wikitext_dataset = datasets.Dataset.from_dict(
        {
            "text": [
                "\n\n".join(_wikitext_dataset["text"])
            ],
        },
    )

    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )

    raw_datasets["wikitext"] = wikitext_dataset

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset["train"], default_data_collator




def filter_c4_train(example, tokenizer, source_min_len, source_max_len):
    messages = example['text']
    input_text = ""
    for message in messages:
        input_text += message
    input_token = tokenizer(input_text)
    return len(input_token['input_ids']) > source_min_len and len(input_token['input_ids']) < source_max_len

def get_c4_wiki_train(tokenizer, seed=0, seqlen=2048):
    raw_datasets = load_from_disk('../eval_my/ppl_datasets/allenai/c4/allenai--c4')
    wikitext_dataset_eval = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        )
    wikitext_dataset_train = load_from_disk('../eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/train')
    wikitext_dataset_train = (
        wikitext_dataset_train  # type: ignore
        .add_column(
            name="timestamp",
            column=[None for _ in range(len(wikitext_dataset_train["text"]))])
        .add_column(
            name="url",
            column=wikitext_dataset_train["text"])
    )
    wikitext_dataset_eval = (
        wikitext_dataset_eval  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset_eval["text"])
        .add_column(
            name="url",
            column=wikitext_dataset_eval["text"])
    )
    
    raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], wikitext_dataset_train])
    raw_datasets["wikitext"] = wikitext_dataset_eval
    raw_datasets["train"][0]  # fast
    raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
    raw_datasets["train"][0]  # up to 10x slower
    raw_datasets["train"] = raw_datasets["train"].flatten_indices()  # rewrite the shuffled dataset on disk as contiguous chunks of data
    raw_datasets["train"][0]  # fast again

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset["train"], default_data_collator


def get_wikitext2(tokenizer, seqlen=2048):
    testdata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return testenc

def get_ptb(tokenizer, seqlen=2048):
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    return testenc

save_seed = False

def get_c4(tokenizer, seqlen=2048):
    valdata = load_from_disk('./eval_my/ppl_datasets/allenai/c4/allenai--c4/validation')
    import random
    random.seed(0)
    valenc = []
    if save_seed:
        index_list = []
        tmp_list = []
    else:
        index_list = torch.load('./eval_my/c4_seed_index.pth')
        tmp_list = torch.load('./eval_my/c4_seed_tmp.pth')
    for _ in range(256):
        if save_seed:
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] > seqlen:
                    tmp_list.append(i)
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            index_list.append(i)
        else:
            i = index_list[_]
            tmp = tmp_list[_]
        j = i + seqlen
        if save_seed:
            valenc.append(tmp.input_ids[:, i:j])
        else:
            valenc.append(tokenizer(valdata[tmp]['text'], return_tensors='pt').input_ids[:, i:j])
    if save_seed:
        torch.save(index_list, './eval_my/c4_seed_index.pth')
        torch.save(tmp_list, './eval_my/c4_seed_tmp.pth')
        breakpoint()
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return valenc 

def get_ptb_new(tokenizer, seqlen=2048):
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    return testenc

def get_c4_new(tokenizer, seqlen=2048):
    valdata = load_from_disk('./eval_my/ppl_datasets/allenai/c4/allenai--c4/validation')

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return valenc


if __name__ == "__main__":
    get_c4_wiki_train(None)



