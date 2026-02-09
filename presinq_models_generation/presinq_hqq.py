import os
from sinq.sinkhorn import sinkhorn_log
import torch
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig

def find_block(H,W,block):
    for i in range(W):
        if (W % (block+i) == 0):
            return block + i
        elif (W % (block-i) == 0):
            return block - i

def get_sink_scale(matrix_list, cat_dim=0, block=64, n_iter=4):
    W = torch.cat(matrix_list, dim=cat_dim).cuda()
    H,Wid = W.shape
    dtype = W.dtype
    W = W.float()
    
    if block <= 0:
        W_hat, mu1, mu2 = sinkhorn_log(W, n_iter)
    else:
        if Wid%block != 0:
            block = find_block(H,W,block)
        assert Wid%block==0, 'block must divide W'
        n_w = Wid//block
        W = W.view(H, Wid//block, block)
        W_batched = W.permute(1,0,2).contiguous().view(n_w, H, block)
        def process_block(mat):
            return sinkhorn_log(mat, n_iter)
        W_hat, mu1, mu2 = torch.vmap(process_block, randomness='different')(W_batched)

    mu1 = mu1 / mu1.median()
    return mu1.view(-1).cuda().to(dtype)

def absorb_sink_scale_qwen(model, normalize_outproj=False, n_gqa_groups=2, group_size=64, n_iter=4):
    for _, layer in tqdm(enumerate(model.model.layers), desc="Pre-SINQ computation"): 
        dev = layer.input_layernorm.weight.device
        dtype = layer.input_layernorm.weight.dtype

        # get out scale and absorb
        # We need to take care of the GQA grouping -- v is repeated n_group times
        if normalize_outproj:
            n_group = n_gqa_groups
            oOut, oIn = layer.self_attn.o_proj.weight.shape
            t_o = get_sink_scale([layer.self_attn.o_proj.weight.data.reshape(n_group*oOut, -1)], block=group_size, n_iter=n_iter)
            layer.self_attn.v_proj.weight.data = torch.matmul(torch.diag(t_o), layer.self_attn.v_proj.weight.data.cuda()).to(dev)
            t_o = torch.cat([t_o]*n_group)
            layer.self_attn.o_proj.weight.data = torch.matmul(layer.self_attn.o_proj.weight.data.cuda(), torch.diag(1/t_o)).to(dev)

        # get down scale and absorb
        t_d = get_sink_scale([layer.mlp.down_proj.weight.data], block=group_size, n_iter=n_iter)
        layer.mlp.down_proj.weight.data = torch.matmul(layer.mlp.down_proj.weight.data.cuda(), torch.diag(1/t_d)).to(dev)
        layer.mlp.up_proj.weight.data = torch.matmul(torch.diag(t_d), layer.mlp.up_proj.weight.data.cuda()).to(dev)

        # get qkv scale and absorb
        t_qkv = get_sink_scale([layer.self_attn.q_proj.weight.data, layer.self_attn.k_proj.weight.data,layer.self_attn.v_proj.weight.data], block=group_size, n_iter=n_iter)
        layer.input_layernorm.weight.data = (layer.input_layernorm.weight.data.cuda() * t_qkv.view(-1)).to(dev)
        layer.self_attn.q_proj.weight.data = torch.matmul(layer.self_attn.q_proj.weight.data.cuda(), torch.diag(1/t_qkv)).to(dev)
        layer.self_attn.k_proj.weight.data = torch.matmul(layer.self_attn.k_proj.weight.data.cuda(), torch.diag(1/t_qkv)).to(dev)
        layer.self_attn.v_proj.weight.data = torch.matmul(layer.self_attn.v_proj.weight.data.cuda(), torch.diag(1/t_qkv)).to(dev)

        # get gate/up scale and absorb
        t_gu = get_sink_scale([layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data], block=group_size, n_iter=n_iter)
        layer.mlp.gate_proj.weight.data = torch.matmul(layer.mlp.gate_proj.weight.data.cuda(), torch.diag(1/t_gu)).to(dev)
        layer.mlp.up_proj.weight.data = torch.matmul(layer.mlp.up_proj.weight.data.cuda(), torch.diag(1/t_gu)).to(dev)
        layer.post_attention_layernorm.weight.data = (layer.post_attention_layernorm.weight.data.cuda() * t_gu.view(-1)).to(dev)

def pre_sinq_qwen3(model, normalize_outproj=False, n_gqa_groups=2, n_repeat=3, group_size=64, n_iter=4):
    for i in range(n_repeat):
        absorb_sink_scale_qwen(model, normalize_outproj, n_gqa_groups, group_size, n_iter=n_iter)


def absorb_sink_scale_DSlite(model, normalize_outproj=False, group_size=64, n_iter=4, normalize_downproj=True, normalize_gu=True):
    first = True
    for _, layer in tqdm(enumerate(model.model.layers), desc="Pre-SINQ computation"): 
        dev = layer.input_layernorm.weight.device
        dtype = layer.input_layernorm.weight.dtype

        ####
        # Attention layer
        ####
        t_qkv = get_sink_scale([layer.self_attn.q_proj.weight.data, layer.self_attn.kv_a_proj_with_mqa.weight.data], block=group_size, n_iter=n_iter)
        layer.input_layernorm.weight.data = (layer.input_layernorm.weight.data.cuda() * t_qkv.view(-1)).to(dev)
        layer.self_attn.q_proj.weight.data = torch.matmul(layer.self_attn.q_proj.weight.data.cuda(), torch.diag(1/t_qkv)).to(dev)
        layer.self_attn.kv_a_proj_with_mqa.weight.data = torch.matmul(layer.self_attn.kv_a_proj_with_mqa.weight.data.cuda(), torch.diag(1/t_qkv)).to(dev)

        t_kvb = get_sink_scale([layer.self_attn.kv_b_proj.weight.data], block=group_size, n_iter=n_iter)
        layer.self_attn.kv_a_layernorm.weight.data = (layer.self_attn.kv_a_layernorm.weight.data.cuda() * t_kvb.view(-1)).to(dev)
        layer.self_attn.kv_b_proj.weight.data = torch.matmul(layer.self_attn.kv_b_proj.weight.data.cuda(), torch.diag(1/t_kvb)).to(dev)

        if normalize_outproj:
            t_o = get_sink_scale([layer.self_attn.o_proj.weight.data], block=group_size, n_iter=n_iter)
            layer.self_attn.o_proj.weight.data = torch.matmul(layer.self_attn.o_proj.weight.data, torch.diag(1/t_o)).to(dev)
            layer.self_attn.kv_b_proj.weight.data[:t_o.view(-1).shape[0]] = torch.matmul(torch.diag(t_o), layer.self_attn.kv_b_proj.weight.data.cuda()[:t_o.view(-1).shape[0]]).to(dev)

        ####
        # MLP + MoE
        ####
        if first:
            first = False

            # get and absorb down scale
            if normalize_downproj:
                t_d = get_sink_scale([layer.mlp.down_proj.weight.data], block=group_size, n_iter=n_iter)
                layer.mlp.down_proj.weight.data = torch.matmul(layer.mlp.down_proj.weight.data.cuda(), torch.diag(1/t_d)).to(dev)
                layer.mlp.up_proj.weight.data = torch.matmul(torch.diag(t_d), layer.mlp.up_proj.weight.data.cuda()).to(dev)

            # get and absorb gate + up scale
            # gather tensors
            if normalize_gu:
                t_gu = get_sink_scale([layer.mlp.up_proj.weight.data, layer.mlp.gate_proj.weight.data], block=group_size, n_iter=n_iter)
                # norm
                layer.post_attention_layernorm.weight.data = (layer.post_attention_layernorm.weight.data.cuda() * t_gu.view(-1)).to(dev)
                # gate and up
                layer.mlp.gate_proj.weight.data = torch.matmul(layer.mlp.gate_proj.weight.data.cuda(), torch.diag(1/t_gu)).to(dev)
                layer.mlp.up_proj.weight.data = torch.matmul(layer.mlp.up_proj.weight.data.cuda(), torch.diag(1/t_gu)).to(dev)
        else:
            if normalize_downproj:
                # get and absorb down scale
                for i in range(64):
                    t_d = get_sink_scale([layer.mlp.experts[i].down_proj.weight.data], block=group_size, n_iter=n_iter)
                    layer.mlp.experts[i].down_proj.weight.data = torch.matmul(layer.mlp.experts[i].down_proj.weight.data.cuda(), torch.diag(1/t_d)).to(dev)
                    layer.mlp.experts[i].up_proj.weight.data = torch.matmul(torch.diag(t_d), layer.mlp.experts[i].up_proj.weight.data.cuda()).to(dev)

            # get and absorb gate + up scale
            # gather tensors
            if normalize_gu:
                weight_list = [layer.mlp.experts[i].up_proj.weight.data for i in range(64)] 
                weight_list = weight_list + [layer.mlp.experts[i].gate_proj.weight.data for i in range(64)] 
                t_gu = get_sink_scale(weight_list, block=group_size, n_iter=n_iter)
                # norm
                layer.post_attention_layernorm.weight.data = (layer.post_attention_layernorm.weight.data.cuda() * t_gu.view(-1)).to(dev)
                # gate and up
                for i in range(64):
                    layer.mlp.experts[i].gate_proj.weight.data = torch.matmul(layer.mlp.experts[i].gate_proj.weight.data.cuda(), torch.diag(1/t_gu)).to(dev)
                    layer.mlp.experts[i].up_proj.weight.data = torch.matmul(layer.mlp.experts[i].up_proj.weight.data.cuda(), torch.diag(1/t_gu)).to(dev)

def pre_sinq_dslite(model, group_size=64, n_iter=4, n_repeat=1):
    for i in range(n_repeat):
        absorb_sink_scale_DSlite(model, group_size=group_size, n_iter=n_iter)

class KLTracker:
    def __init__(self, calib_string='the'):
        self.p_logits = None
        self.inputs = tokenizer(calib_string, return_tensors="pt").to("cuda")

    def get_baseline_logits(self, model):
        """Computes and stores reference logits (P)."""
        model.eval()
        with torch.no_grad():
            # Detach is crucial to prevent memory leaks/graph retention
            self.p_logits = model(**self.inputs).logits.detach()

    def get_KL_to_baseline(self, model):
        """Computes KL(P || Q) where Q is the provided model."""
        print('computing KL div')
        if self.p_logits is None:
            raise ValueError("Baseline logits not set. Call get_baseline_logits first.")

        # Get Q logits (Active model)
        q_logits = model(**self.inputs).logits
        
        # Align P to Q's device and calculate Log Softmax for both
        # P = Target (Baseline), Q = Input (Active)
        log_prob_p = F.log_softmax(self.p_logits.to(q_logits.device).double(), dim=-1)
        log_prob_q = F.log_softmax(q_logits.double(), dim=-1)

        # Compute KL divergence
        return F.kl_div(
            input=log_prob_q, 
            target=log_prob_p, 
            reduction='batchmean', 
            log_target=True
        ).mean()
    
prompt1 = """- Fiction: "In a hidden valley where time moved slower, an old painter discovered a brush that could bring his creations to life. His first stroke awoke something unexpected..."
- News: "A rare celestial event—a triple conjunction of Jupiter, Saturn, and Mars—will be visible tonight for the first time in over 200 years. Astronomers urge skywatchers not to miss..."
- Code: `const countVowels = (str) => [...str].filter(c => "aeiou".includes(c.toLowerCase())).length;\nconsole.log(countVowels("Hello, world!"));`
- Math: A car travels 240 km in 3 hours at constant speed. If it then accelerates by 20 km/h for the next 2 hours, what's the total distance traveled?
- Facts: "The Great Wall of China is approximately 21,196 km long. However, contrary to myth, it cannot be seen from space with the naked eye..."
- Fiction: "The last tree in the desert city whispered secrets to those who listened. When a young girl finally understood its language, she discovered it held the blueprint to regrow the entire forest..."
- News: "Scientists develop biodegradable battery that decomposes in soil after 30 days, offering potential solution to electronic waste pollution..."
- Code: `def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a)\n        a, b = b, a + b`
- Math: Find the area of a triangle with vertices at (2,3), (5,7), and (9,4) using the determinant formula
- Facts: "Octopuses have three hearts and blue blood. Two hearts pump blood to the gills while the third circulates it to the rest of the body..."
"""

prompt2 = """- Fiction: "When the stars aligned, a librarian in Prague found every book in the library rearranged into an unknown language—except one, which bore her name on the cover..."
- News: "New legislation bans single-use plastics in the European Union, with critics arguing the policy doesn't address industrial waste, while proponents hail it as a critical first step..."
- Code: `import numpy as np\narr = np.array([1, 2, 3])\nprint(arr * 2)`
- Math: (14.6 * 3.2) - (5.9 ** 2) + (18 / 1.8) =
- Facts: "The male seahorse carries and gives birth to its young. Females deposit eggs into the male's pouch, where they are fertilized and nurtured until birth..."
- Fiction: "Every full moon, the antique shop's items would rearrange themselves. The owner kept meticulous records until he noticed a pattern that predicted future events with uncanny accuracy..."
- News: "Global coral bleaching event declared as ocean temperatures reach record highs, threatening marine ecosystems worldwide..."
- Code: `from collections import defaultdict\nd = defaultdict(int)\nfor word in text.split():\n    d[word] += 1`
- Math: Solve the quadratic equation: 2x² - 7x + 3 = 0
- Facts: "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible..."
"""

prompt3 = """- Fiction: "A lighthouse keeper on a remote island noticed the beacon dimming each night, replaced by a faint chorus of voices singing in unison—until one evening, they called his name..."
- News: "AI-designed proteins could revolutionize medicine, with researchers announcing the creation of molecules that target previously 'undruggable' diseases..."
- Code: `class Cat:\n  def __init__(self, name):\n    self.name = name\n  def speak(self):\n    return f"{self.name} says 'Meow!'"`
- Math: If 3x + 5 = 20, what is the value of x² - 2x?
- Facts: "Venus rotates backward compared to most planets in the solar system, meaning its sun rises in the west and sets in the east..."
- Fiction: "The clockmaker's final creation could manipulate time itself. But when he tried to undo his greatest regret, he discovered why some moments were meant to remain unchanged..."
- News: "Breakthrough in quantum computing: Researchers achieve quantum supremacy with 128-qubit processor, solving problems previously thought impossible..."
- Code: `const debounce = (func, delay) => {\n  let timeout;\n  return (...args) => {\n    clearTimeout(timeout);\n    timeout = setTimeout(() => func.apply(this, args), delay);\n  };\n};`
- Math: Calculate the volume of a sphere with radius 5 cm (V = 4/3πr³)
- Facts: "A single strand of spider silk is stronger than steel of the same diameter and can stretch up to five times its length without breaking..."
"""

prompt4 = """- Fiction: "In a village where every resident shared the same dream nightly, a child was born who dreamed of nothing—until the others' dreams began vanishing one by one..."
- News: "SpaceX successfully landed a reusable rocket on its tenth flight, setting a new milestone for cost efficiency in space exploration..."
- Code: `list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, [1, 2, 3, 4])))`
- Math: (7! / (4! * 3!)) + (√144 * 2³) =
- Facts: "A group of flamingos is called a 'flamboyance.' These birds often balance on one leg to conserve body heat..."
- Fiction: "The museum's newest exhibit—a perfectly preserved Victorian doll—began appearing in visitors' dreams with whispered warnings that always came true the next day..."
- News: "Researchers discover new species of deep-sea fish that glows with bioluminescent patterns never before documented in marine biology..."
- Code: `using System.Linq;\nvar evenNumbers = numbers.Where(n => n % 2 == 0).ToList();`
- Math: Find the derivative of f(x) = 3x⁴ - 2x³ + 7x - 5
- Facts: "The shortest war in history lasted only 38 minutes. It occurred in 1896 between Britain and Zanzibar when the Sultan's forces surrendered after a brief naval bombardment..."
"""

prompt5 = """- Fiction: "A detective specializing in 'impossible crimes' received a letter postmarked from 1942. The handwriting matched her own—but she hadn't been born yet..."
- News: "Archaeologists uncovered a 1,500-year-old mosaic beneath a vineyard in Italy, depicting scenes from Greek mythology in near-perfect condition..."
- Code: `String reverse(String s) {\n  return new StringBuilder(s).reverse().toString();\n}`
- Math: (log₁₀1000 * 5²) - (e³ / ln(20)) ≈ (round to two decimal places)
- Facts: "Sharks have been around longer than trees. The earliest shark fossils date back 400 million years, while trees appeared roughly 350 million years ago..."
- Fiction: "The bookstore that only appeared during rainstorms contained volumes written by authors from parallel universes. One rainy Tuesday, a customer found a book with their life story—but with a different ending..."
- News: "World's first successful transplant of 3D-printed functional organ performed, marking major advancement in regenerative medicine..."
- Code: `function deepClone(obj) {\n  return JSON.parse(JSON.stringify(obj));\n}`
- Math: Calculate the compound interest on $10,000 at 5% annual rate compounded quarterly for 3 years
- Facts: "The human nose can detect over 1 trillion different scents, far more than the previously believed 10,000 scents..."
"""

test_string = prompt1+prompt2+prompt3+prompt4+prompt5

if __name__ == '__main__':
    import sys
    # Evaluation script availbale in the SINQ official repository
    from eval_my.evaluate_ import evaluate_model   
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hqq.models.hf.base import AutoHQQHFModel
    from hqq.core.quantize import BaseQuantizeConfig
    import argparse

    parser = argparse.ArgumentParser()
    # Model configuration
    parser.add_argument(
        "--model_name", 
        type=str,
        default="Qwen/Qwen3-1.7B", 
        help="Model identifier (e.g. Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, etc.)"
    )

    parser.add_argument(
        "--temp_dir", 
        type=str,
        default="./presinq_models",
        help="Directory for temporary model storage"
    )

    # Boolean flags (Default to False, set to True if flag is present)
    parser.add_argument(
        "--save_to_disk", 
        action="store_true",
        help="Whether to save the results/model to disk"
    )

    parser.add_argument(
        "--baseline_only", 
        action="store_true",
        help="Run only the baseline model without modifications"
    )

    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Run validation steps"
    )

    # Quantization/Grouping configuration
    parser.add_argument(
        "--group_size", 
        type=int, 
        default=64,
        help="The size of the groups for quantization"
    )

    parser.add_argument(
        "--nbits", 
        type=int, 
        default=4,
        help="Number of bits for quantization"
    )

    args = parser.parse_args()

    # Accessing the arguments
    model_name = args.model_name
    save_to_disk = args.save_to_disk
    baseline_only = args.baseline_only
    validate = args.validate
    group_size = args.group_size
    nbits = args.nbits
    temp_dir = args.temp_dir


    # load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hqq_quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size)

    # Hyperparameter search
    if not baseline_only:
        tracker = KLTracker(test_string)
        tracker.get_baseline_logits(model)

        print(model)

        configs = []
        for gs in [group_size]: # You can modify it if needed
            for n_iter in [2,8,16]: # You can modify it if needed
                for n_repeat in [1,2]: # You can modify it if needed
                    configs.append({'group_size': gs, 'n_iter':n_iter, 'n_repeat':n_repeat})
                
        best_kl = 1_000_000_000
        best_config = None
        ppls = []
        kls = []
        for config in configs:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')

            # pre-sinq the model with the current configuration
            if 'deepseek' in model_name:
                pre_sinq_dslite(model, **config)
            elif 'Qwen' in model_name:
                pre_sinq_qwen3(model, **config)
                pass

            # quantize it
            AutoHQQHFModel.quantize_model(model, quant_config=hqq_quant_config, compute_dtype=torch.bfloat16, device='cuda')
            print('model loaded')

            # option to validate hparam search against wikitext2
            if validate:
                results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    tasks="",
                    eval_ppl='wikitext2',
                    batch_size=8)
                task_results = results['wikitext2'] #perplexity / ppl
                ppls.append(task_results)
                
            # track kl-div against original model
            KL = tracker.get_KL_to_baseline(model).item()
            
            kls.append(KL)
            print('got KL div')

            print('~~~~'*5)
            print(f'{KL=}')
            print(config)

            if KL < best_kl:
                best_kl = KL
                best_config = config

        # presinq again with the best configuration
        print('*******'*8)
        print('best KL:')
        print(best_kl)
        print('best configuration:')
        print(best_config)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')

        if 'deepseek' in model_name:
            pre_sinq_dslite(model, **best_config)
        elif 'Qwen' in model_name:
            pre_sinq_qwen3(model, **best_config)

        if save_to_disk:
            print('saving model to disk')
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            print('model saved')

    torch.cuda.empty_cache()
    gc.collect()
    print(
        "CUDA memory allocated and reserved (GB):",
        torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9  # in GB
    )

    # final eval
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        tasks="",
        eval_ppl='wikitext2',
        batch_size=8)
    task_results = results['wikitext2'] #perplexity / ppl
    print('ppl:', task_results)

    # make a plot if validating the hparam search
    if validate:
        print(kls)
        print(ppls)
        import numpy as np
        import matplotlib.pyplot as plt
        plt.scatter(np.asarray(kls), np.asarray(ppls))
        plt.savefig('test.png')