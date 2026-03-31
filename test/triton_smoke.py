import triton
import triton.language as tl
from triton.compiler import ASTSource


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    src = ASTSource(
        fn=add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constants={"BLOCK": 1024},
    )
    triton.compile(src, options={"num_warps": 4, "num_stages": 3, "debug": True})
    print("triton smoke compile complete")


if __name__ == "__main__":
    main()
