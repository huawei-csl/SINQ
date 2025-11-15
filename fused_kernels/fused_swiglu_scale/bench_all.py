import torch
import triton
from swiglu_triton_impl import swiglu_triton, swiglu_torch
from swiglu_scale_triton_impl import swiglu_scale_triton, swiglu_scale_torch
import matplotlib.pyplot as plt

HIDDEN_DIM = 4096

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch'],
        x_vals=[1, 4, 16, 32, 128, 512, 1024],
        line_arg='provider',
        line_vals=['torch', 'triton', 'scale_torch', 'scale_triton'],
        line_names=['SwiGLU_torch', 'SwiGLU_triton', 'SwiGLU_scale_torch', 'SwiGLU_scale_triton'],
        styles=[('C0', '--'), ('C1', '-.'), ('C2', '--'), ('k', '-.')],
        ylabel="Execution Time (ms)",
        plot_name="Performance",
        args={},
    ),
)
def benchmark(batch, provider):
    hidden = HIDDEN_DIM
    dtype = torch.float16
    device = "cuda:0"
    a = torch.randn(batch, hidden, dtype=dtype, device=device)
    b = torch.randn(batch, hidden, dtype=dtype, device=device)
    s = torch.abs(torch.randn(1, hidden, dtype=dtype, device=device))  # scale can only be postive

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'triton':
        results = triton.testing.do_bench(
            lambda: swiglu_triton(a, b), quantiles=quantiles
            )
    elif provider == 'torch':
        results = triton.testing.do_bench(
            lambda: swiglu_torch(a, b), quantiles=quantiles
            )
    elif provider == 'scale_triton':
        results = triton.testing.do_bench(
            lambda: swiglu_scale_triton(a, b, s), quantiles=quantiles
            )
    elif provider == 'scale_torch':
        results = triton.testing.do_bench(
            lambda: swiglu_scale_torch(a, b, s), quantiles=quantiles
            )
    else:
        raise ValueError("undefined provider")

    return results

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)
    plt.grid(True)
    plt.title(f"shape = [batch, hidden] = [x, {HIDDEN_DIM}]")
    plt.savefig("bench_swiglu_scale.png", dpi=288)
