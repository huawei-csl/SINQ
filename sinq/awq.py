
'''
written by SINQ authors
'''

import torch
from datasets import load_dataset  # Added for calibration dataset
from torch import nn
import numpy as np
from .sinkhorn import sinkhorn_log
from collections import deque

def quantize_rtn(matrix, awq_scales=None, min_max=[], niter=None):
    w = matrix
    if not (awq_scales is None):
        w = w * awq_scales
    dtype = w.dtype
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = min_max[1]
    min_int = min_max[0]
    scales = (max_val - min_val).clamp(min=1e-4) / max_int
    zeros = (-torch.round(min_val / scales))
    q = torch.clamp(torch.round(w / scales + zeros), min_int, max_int)
    
    if not (awq_scales is None):
        scales = scales / awq_scales
    return q, scales, zeros, dtype

def dequantize_rtn(q, scales, zeros, dtype):
    return ((q - zeros)*scales).to(dtype)

def rtn_fake_quant(matrix, scale, min_max):
    return dequantize_rtn(*quantize_rtn(matrix, scale, min_max))

def sinq_fake_quant(matrix, awq_scales, min_max):
    matrix, mu1, mu2 = sinkhorn_log(matrix, 16)
    matrix = matrix*awq_scales
    dq = dequantize_rtn(*quantize_rtn(matrix, None, min_max))
    mu1 = mu1 / awq_scales
    return dq*mu1*mu2


def _get_tokenizer(tokenizer_or_processor):
    """Extract tokenizer from processor if needed (for multimodal models)."""
    # If it's a processor (e.g., PixtralProcessor), extract the tokenizer
    if hasattr(tokenizer_or_processor, 'tokenizer'):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def get_calib_dataset(
    tokenizer=None,
    n_samples=16,
    max_seq_len=512,
    col="text",
):
    tokenizer = _get_tokenizer(tokenizer)
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    s = []
    count = 0
    for d in dataset:
        line = d[col]
        line = line.strip()
        encoded = tokenizer.encode(line)
        if len(encoded) > max_seq_len:
            continue
        sample = torch.tensor([encoded])
        s.append(sample)
        count += 1
        if count == n_samples:
            break

    cat_samples = torch.cat(s, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    return [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
    ]


def get_simple_calibration_data(tokenizer, block_size=128):
    tokenizer = _get_tokenizer(tokenizer)
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

    prompts = [prompt1,prompt2,prompt3,prompt4,prompt5]

    samples = []
    for prompt in prompts:
        line_encoded = tokenizer.encode(prompt)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
    if samples:
        cat_samples = torch.cat(samples, dim=1)
        n_split = cat_samples.shape[1] // block_size
        print(f" * Split into {n_split} blocks")
        return [
            cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
        ]

@torch.inference_mode()
def collect_activations(model, calibration_data, num_samples=16):
    """
    Collect input activations for all linear layers using calibration data.
    Returns a dictionary mapping layer names to activation tensors.
    """
    activation_cache = {}
    hooks = []
    
    # Hook function to capture activations
    def hook_fn(module, input, output, name):  # Added output parameter
        if name not in activation_cache:
            activation_cache[name] = []
        # Detach and store input activations
        activation_cache[name].append(input[0].detach().cpu().bfloat16())
    
    # Register hooks for all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n) 
            )
            hooks.append(hook)
    
    # Run calibration data through model
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(calibration_data):
            if i >= num_samples: 
                break
            inputs = inputs.to(model.device)
            model(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate collected activations
    for name in activation_cache:
        activation_cache[name] = torch.cat(activation_cache[name], dim=0)
    
    return activation_cache


def tiled_fake_quant_rectangle(M, fakequant, min_max, block=64, transpose=False, scales=None):
    if transpose:
        M = M.T
    mshape = M.shape
    H,W = M.shape
    assert W%block==0, 'block must divide W'
    n_w = W//block

    M = M.view(H, W//block, block)
    M_batched = M.permute(1,0,2).contiguous().view(n_w, H, block)

    if not (scales is None):
        scales = scales.view(1, n_w, block).permute(1,0,2).contiguous().view(n_w, 1, block)
    def process_block(mat, scales):
        return fakequant(mat, scales, min_max) 
    DQ_batched = torch.vmap(process_block, randomness='different')(M_batched, scales)
    DQ = DQ_batched.view(n_w, H, block).permute(1,0,2).contiguous()
    return DQ.reshape(mshape) if not transpose else DQ.reshape(mshape).T


def compute_awq_scale(weights, activations, min_max, tile=128, num_alphas=20, num_betas=1, use_weightscale=True, method=None):
    """
    Compute AWQ scaling factors with grid search over alpha values.
    Returns best scale factors and best alpha based on reconstruction error.
    """
    torch.cuda.empty_cache()
    alpha_values = np.arange(0, 1, 1./num_alphas) if num_alphas > 1 else [0]
    beta_values = np.arange(0, 1, 1./num_betas) if num_betas > 1 else [0]
    dtype = weights.dtype
    activations = activations.to(weights.device).to(weights.dtype)


    # Compute activation importance - ensure it matches weight dimensions
    mu_tokens = activations.float().abs().mean([0,1])
    mu_weights = weights.float().abs().mean([0])
    std_tokens = activations.float().std([0,1])

    best_scale = torch.ones_like(mu_tokens.view(-1))
    best_error = float('inf')


    torch.cuda.empty_cache()
    for alpha in alpha_values:
        for beta in beta_values:
            scales = (mu_tokens.view(-1).pow(alpha) * std_tokens.view(-1).pow(beta)).clamp(min=1e-4)

            if use_weightscale:
                # scaling with the weights is used in the auto-awq code as well
                scales = (scales / (mu_weights.pow(1-alpha) + 1e-4)).clamp(min=1e-4)

            scales = scales / (scales.max() * scales.min()).sqrt()

            # Calculate reconstruction error
            gt = activations @ weights.T
            fake_quant = sinq_fake_quant if 'sinq' in method else rtn_fake_quant
            Wq = tiled_fake_quant_rectangle(weights, fakequant=fake_quant, min_max=min_max, block=tile, scales=scales)
            pred = activations @ Wq.T.to(weights.dtype)
            if 'l1' in method:   
                reconstruction_error = ((pred - gt)).abs().mean()
            else: 
                reconstruction_error = ((pred - gt)).square().mean()


            # Update best alpha
            if reconstruction_error < best_error:
                best_error = reconstruction_error.item()
                best_scale = scales

    return best_scale.to(dtype)#, best_alpha, best_dq, best_error
