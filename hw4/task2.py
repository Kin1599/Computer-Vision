import torch
import triton
import triton.language as tl

def layernorm_fwd_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd

    out = x_hat * weight + bias
    return out

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _layernorm_fwd_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    row_start = row_idx * N

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=0) / N

    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N

    rstd = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x_hat = diff * rstd
    y = x_hat * weight + bias

    tl.store(y_ptr + row_start + offsets, y, mask=mask)

    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["N"],
    reset_to_zero=["dweight_ptr", "dbias_ptr"],
)
@triton.jit
def _layernorm_bwd_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    row_start = row_idx * N

    dy = tl.load(dy_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + row_idx).to(tl.float32)
    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)

    x_hat = tl.where(mask, (x - mean) * rstd, 0.0)

    dx_hat = dy * weight

    dx_hat = tl.where(mask, dx_hat, 0.0)

    sum_dx_hat = tl.sum(dx_hat, axis=0)
    sum_dx_hat_x_hat = tl.sum(dx_hat * x_hat, axis=0)

    dx = (dx_hat - sum_dx_hat / N - x_hat * sum_dx_hat_x_hat / N) * rstd

    tl.store(dx_ptr + row_start + offsets, dx, mask=mask)

    dweight = dy * x_hat
    dbias = dy

    tl.atomic_add(dweight_ptr + offsets, dweight, sem="relaxed", mask=mask)
    tl.atomic_add(dbias_ptr + offsets, dbias, sem="relaxed", mask=mask)

class TritonLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        assert x.is_cuda, "x must be CUDA tensor"
        assert weight.is_cuda, "weight must be CUDA tensor"
        assert bias.is_cuda, "bias must be CUDA tensor"

        original_shape = x.shape
        N = x.shape[1]
        M = x.numel() // N
        
        x_2d = x.contiguous().view(M, N)
        weight = weight.contiguous()
        bias = bias.contiguous()

        y = torch.empty_like(x_2d)

        mean = torch.empty((M,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

        block_size = triton.next_power_of_2(N)

        grid = (M,)

        _layernorm_fwd_kernel[grid](
            x_2d,
            weight,
            bias,
            y,
            mean,
            rstd,
            N,
            eps,
            BLOCK_SIZE=block_size,
        )

        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.N = N
        ctx.M = M
        ctx.original_shape = original_shape
        ctx.block_size = block_size

        return y.view(original_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, weight, mean, rstd = ctx.saved_tensors

        N = ctx.N
        M = ctx.M
        original_shape = ctx.original_shape
        block_size = ctx.block_size

        dy_2d = dy.contiguous().view(M, N)

        dx = torch.empty_like(x_2d)
        dweight = torch.zeros_like(weight)
        dbias = torch.zeros_like(weight)

        grid = (M,)

        _layernorm_bwd_kernel[grid](
            dy_2d,
            x_2d,
            weight,
            mean,
            rstd,
            dx,
            dweight,
            dbias,
            N,
            BLOCK_SIZE=block_size,
        )

        return dx.view(original_shape), dweight, dbias, None
    
def layernorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
):
    return TritonLayerNormFunction.apply(x, weight, bias, eps)

def check_correctness():
    torch.manual_seed(0)

    test_shapes = [
        (32, 128),
        (128, 256),
        (512, 512),
        (1024, 1024),
    ]

    for shape in test_shapes:
        print(f"Checking shape: {shape}")

        x_torch = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
        w_torch = torch.randn(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)
        b_torch = torch.randn(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)

        x_triton = x_torch.detach().clone().requires_grad_(True)
        w_triton = w_torch.detach().clone().requires_grad_(True)
        b_triton = b_torch.detach().clone().requires_grad_(True)

        dy = torch.randn(shape, device="cuda", dtype=torch.float32)

        y_torch = layernorm_fwd_torch(x_torch, w_torch, b_torch)
        y_triton = layernorm_triton(x_triton, w_triton, b_triton)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-4, atol=1e-4)

        y_torch.backward(dy)
        y_triton.backward(dy)

        torch.testing.assert_close(x_triton.grad, x_torch.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w_triton.grad, w_torch.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(b_triton.grad, b_torch.grad, rtol=1e-4, atol=1e-4)

    print("All correctness checks passed!")

# BENCHMARK UTILS

def benchmark_cuda(fn, warmup: int = 25, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(repeat):
        fn()

    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / repeat

def benchmark():
    torch.manual_seed(0)

    shapes = [
        (1024, 1024),
        (4096, 1024),
        (4096, 2048),
    ]

    print("\nBenchmark: forward + backward")
    print("-" * 80)
    print(f"{'Shape':>18} | {'PyTorch, ms':>12} | {'Triton, ms':>12} | {'Speedup':>10}")
    print("-" * 80)

    for shape in shapes:
        x_torch = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
        w_torch = torch.randn(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)
        b_torch = torch.randn(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)

        x_triton = x_torch.detach().clone().requires_grad_(True)
        w_triton = w_torch.detach().clone().requires_grad_(True)
        b_triton = b_torch.detach().clone().requires_grad_(True)

        dy = torch.randn(shape, device="cuda", dtype=torch.float32)

        def run_torch():
            x_torch.grad = None
            w_torch.grad = None
            b_torch.grad = None

            y = layernorm_fwd_torch(x_torch, w_torch, b_torch)
            y.backward(dy)

        def run_triton():
            x_triton.grad = None
            w_triton.grad = None
            b_triton.grad = None

            y = layernorm_triton(x_triton, w_triton, b_triton)
            y.backward(dy)

        torch_ms = benchmark_cuda(run_torch)
        triton_ms = benchmark_cuda(run_triton)

        speedup = torch_ms / triton_ms

        print(
            f"{str(shape):>18} | "
            f"{torch_ms:12.4f} | "
            f"{triton_ms:12.4f} | "
            f"{speedup:10.2f}x"
        )


if __name__ == "__main__":
    check_correctness()
    benchmark()

# Output

# Checking shape: (32, 128)
# Checking shape: (128, 256)
# Checking shape: (512, 512)
# Checking shape: (1024, 1024)
# All correctness checks passed!

# Benchmark: forward + backward
# --------------------------------------------------------------------------------
#              Shape |  PyTorch, ms |   Triton, ms |    Speedup
# --------------------------------------------------------------------------------
#       (1024, 1024) |       0.7384 |       0.4843 |       1.52x
#       (4096, 1024) |       2.7523 |       0.5505 |       5.00x
#       (4096, 2048) |       5.3673 |       1.0932 |       4.91x