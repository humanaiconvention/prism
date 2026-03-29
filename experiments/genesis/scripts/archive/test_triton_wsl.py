"""Test if Triton can compile a simple kernel in WSL."""
import torch
import triton
import triton.language as tl
import time

print(f"Triton version: {triton.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
out = torch.empty(1024, device="cuda")

t0 = time.time()
add_kernel[(1,)](x, y, out, 1024, BLOCK=1024)
torch.cuda.synchronize()
t1 = time.time()

print(f"Triton kernel: {(t1-t0)*1000:.0f}ms")
print(f"Correct: {torch.allclose(out, x + y)}")
print("SUCCESS - Triton CUDA compilation works!")
