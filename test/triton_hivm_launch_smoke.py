import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    n_elements = 4096
    x = torch.randn(n_elements, device="npu", dtype=torch.float32)
    y = torch.randn(n_elements, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK=1024)
    print("triton launch smoke complete")


if __name__ == "__main__":
    main()
