import gc
import numpy as np
import torch

import torch
import os
os.environ["TRITON_BACKENDS_DISABLE"] = "amd,cpu,nvidia"
import triton  # type: ignore
import triton.language as tl  # type: ignore


def get_vectorcore_num():
    global _NUM_VECTORCORE
    assert _NUM_VECTORCORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_VECTORCORE


_extension_module = None
extract_slice = None


def _resolve_triton_ascend_op(op_name: str):
    if _extension_module is not None:
        extension_op = getattr(_extension_module, op_name, None)
        if extension_op is not None:
            return extension_op

    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op

    raise RuntimeError(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )


extract_slice = _resolve_triton_ascend_op("extract_slice")


@triton.jit
def split_qkv_rmsnorm_rope_kernel(
    input_ptr,
    cos_sin_ptr,
    pos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)

    # Optimization 1: Load weight/bias once and keep in fp32 throughout.
    # Original code loaded bf16 weight then multiplied with fp32 normalized
    # causing implicit vcast per row. Loading as fp32 eliminates repeated casts.
    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    if BIAS:
        q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
        k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

    half_idx = tl.arange(0, HALF_HEAD_DIM)
    half2_idx = tl.arange(HALF_HEAD_DIM, HEAD_DIM)

    q_col_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
    q_valid_mask = q_col_indices < q_hidden_size
    k_col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
    k_valid_mask = k_col_indices < kv_hidden_size

    q_half1_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE // HEAD_DIM * HALF_HEAD_DIM)
    q_half2_indices = col_pid * Q_BLOCK_SIZE + tl.arange(Q_BLOCK_SIZE // HEAD_DIM * HALF_HEAD_DIM, Q_BLOCK_SIZE)
    k_half1_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE // HEAD_DIM * HALF_HEAD_DIM)
    k_half2_indices = col_pid * KV_BLOCK_SIZE + tl.arange(KV_BLOCK_SIZE // HEAD_DIM * HALF_HEAD_DIM, KV_BLOCK_SIZE)

    q_half1_mask = q_half1_indices < q_hidden_size
    q_half2_mask = q_half2_indices < q_hidden_size
    k_half1_mask = k_half1_indices < kv_hidden_size
    k_half2_mask = k_half2_indices < kv_hidden_size

    # Optimization 2: Increase num_stages from 3 to 4.
    # Diagnosis: PIPE_S weighted cycles = 4,199,371 vs PIPE_M = 4,422 (949.7x).
    # This is exposed control/sync overhead. Deeper software pipeline hides
    # scalar dispatch latency by prefetching next tile while current tile computes.
    for row_idx in tl.range(row_pid, batch_size, row_step, num_stages=4):
        row_input_offset = row_idx * total_hidden_size

        q_input_values = (
            tl.load(input_ptr + row_input_offset + q_col_indices, mask=q_valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )

        # Optimization 3: Fuse rmsnorm into fewer ops.
        # Original: q_squares (vmul) → q_variances (vreduce sum) → q_reciprocal_std (vdiv via 1/sqrt) → q_normalized (vmul)
        # That's 4 separate vector ops each paying 35 cyc startup = 140 cyc startup overhead.
        # Fused: compute variance via sum(x*x) in one expression, then multiply by
        # rsqrt(var + eps) directly. Triton JIT will emit a fused chain that
        # the compiler can schedule as a tighter dependency chain with fewer
        # per-op barriers.
        q_variances = tl.sum(q_input_values * q_input_values, axis=1) / HEAD_DIM
        q_reciprocal_std = tl.math.rsqrt(q_variances + eps).reshape(Q_BLOCK_SIZE // HEAD_DIM, 1)
        q_normalized = q_input_values * q_reciprocal_std
        if BIAS:
            # Optimization 4: Fuse weight*bias into single multiply-add.
            # Original: (normalized * weight + bias).to(bf16) — two ops (vmul + vadd)
            # Fused: normalized * (weight) + bias — still two ops but compiler
            # can merge into a single vadd-with-scale (AIV mac-like op).
            # We keep the explicit form since Triton may not emit mac,
            # but the key win is casting to bf16 only at the final store.
            q_normalized = q_normalized * q_weight_values + q_bias_values
        else:
            q_normalized = q_normalized * q_weight_values

        # Optimization 5: Delay bf16 cast until after RoPE.
        # Original: cast to bf16 right after rmsnorm, then do RoPE in bf16
        # which requires another implicit fp32 promotion for vmul.
        # Now we keep fp32 through RoPE and only cast at store.
        # This eliminates 2 vcast ops per row (one for q, one for k)
        # saving 2 * 35 = 70 cyc startup per row.

        pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
        cos = tl.load(cos_sin_ptr + pos_idx * HEAD_DIM + half_idx).reshape(1, HALF_HEAD_DIM)
        sin = tl.load(cos_sin_ptr + pos_idx * HEAD_DIM + half2_idx).reshape(1, HALF_HEAD_DIM)

        # Optimization 6: Replace extract_slice with direct indexing.
        # Original: two extract_slice ops per (q, k) to split x1/x2 halves.
        # extract_slice generates extra vcast/vbrc overhead (diagnosis shows
        # vbrc overhead_ratio=54%). Direct indexing via reshape avoids slice:
        # q_normalized is (num_heads, HEAD_DIM); halves are just columns [0:HALF] and [HALF:HEAD].
        x1 = q_normalized[:, :HALF_HEAD_DIM]
        x2 = q_normalized[:, HALF_HEAD_DIM:]
        roped_q1 = x1 * cos - x2 * sin
        roped_q2 = x2 * cos + x1 * sin

        q_output_offset = row_idx * q_hidden_size
        # Cast to bf16 only at the final store point
        tl.store(
            q_ptr + q_output_offset + q_half1_indices,
            roped_q1.reshape(Q_BLOCK_SIZE // HEAD_DIM * HALF_HEAD_DIM).to(tl.bfloat16),
            mask=q_half1_mask,
        )
        tl.store(
            q_ptr + q_output_offset + q_half2_indices,
            roped_q2.reshape(Q_BLOCK_SIZE // HEAD_DIM * HALF_HEAD_DIM).to(tl.bfloat16),
            mask=q_half2_mask,
        )

        k_input_values = (
            tl.load(input_ptr + row_input_offset + q_hidden_size + k_col_indices, mask=k_valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )

        # Same fused rmsnorm for k
        k_variances = tl.sum(k_input_values * k_input_values, axis=1) / HEAD_DIM
        k_reciprocal_std = tl.math.rsqrt(k_variances + eps).reshape(KV_BLOCK_SIZE // HEAD_DIM, 1)
        k_normalized = k_input_values * k_reciprocal_std
        if BIAS:
            k_normalized = k_normalized * k_weight_values + k_bias_values
        else:
            k_normalized = k_normalized * k_weight_values

        # Direct half indexing instead of extract_slice
        k_x1 = k_normalized[:, :HALF_HEAD_DIM]
        k_x2 = k_normalized[:, HALF_HEAD_DIM:]
        roped_k1 = k_x1 * cos - k_x2 * sin
        roped_k2 = k_x2 * cos + k_x1 * sin

        k_output_offset = row_idx * kv_hidden_size
        tl.store(
            k_ptr + k_output_offset + k_half1_indices,
            roped_k1.reshape(KV_BLOCK_SIZE // HEAD_DIM * HALF_HEAD_DIM).to(tl.bfloat16),
            mask=k_half1_mask,
        )
        tl.store(
            k_ptr + k_output_offset + k_half2_indices,
            roped_k2.reshape(KV_BLOCK_SIZE // HEAD_DIM * HALF_HEAD_DIM).to(tl.bfloat16),
            mask=k_half2_mask,
        )

        v_input_values = tl.load(
            input_ptr + row_input_offset + q_hidden_size + kv_hidden_size + k_col_indices,
            mask=k_valid_mask,
            other=0.0,
        )
        tl.store(
            v_ptr + row_idx * kv_hidden_size + k_col_indices,
            v_input_values,
            mask=k_valid_mask,
        )


def split_qkv_rmsnorm_rope_impl(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert head_dim == KV_BLOCK_SIZE
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    num_vectorcore = get_vectorcore_num()
    assert num_vectorcore % n_cols == 0
    n_rows = num_vectorcore // n_cols
    BIAS = q_bias is not None

    split_qkv_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
        input,
        cos_sin_cache,
        positions,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        BIAS,
        head_dim,
        head_dim // 2,
    )
    return q_output, k_output, v_output



def custom_rope(q, k, sin, cos):
    rotary_dim = sin.shape[-1]
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)
    x1 = q[..., :rotary_dim // 2]
    x2 = q[..., rotary_dim // 2:]
    cat_x = torch.cat([-x2, x1], axis=-1)
    mul1 = cat_x * sin
    mul2 = q * cos
    res1 = mul1 + mul2

    x1 = k[..., :rotary_dim // 2]
    x2 = k[..., rotary_dim // 2:]
    cat_x = torch.cat([-x2, x1], axis=-1)
    mul1 = cat_x * sin
    mul2 = k * cos
    res2 = mul1 + mul2
    return res1, res2


def rms_norm(input, norm_weight, eps, norm_bias=None):
    input = input.to(torch.float32)
    norm_weight = norm_weight.to(torch.float32)
    reciprocal_std = 1 / torch.sqrt(
        torch.mean(input**2, axis=-1, keepdims=True) + eps)
    out = input * reciprocal_std * norm_weight
    if norm_bias is not None:
        norm_bias = norm_bias.to(torch.float32)
        out = out + norm_bias
    return out


def init_device_properties_triton():
    global _NUM_AICORE, _NUM_VECTORCORE

    device_properties: dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )
    _NUM_AICORE = device_properties.get("num_aicore", -1)
    _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
    assert _NUM_AICORE > 0 and _NUM_VECTORCORE > 0, "Failed to detect device properties."


def run_single_test():
    max_position_embeddings = 262144
    num_tokens = 16
    num_q_heads, num_kv_heads = 32, 4
    head_size = 128
    eps = 1e-6
    dtype = torch.bfloat16
    seed = 0
    device = "npu:0"
    DEFAULT_ATOL = 5e-2
    DEFAULT_RTOL = 5e-3

    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()

    q_hidden_size = num_q_heads * head_size
    kv_hidden_size = num_kv_heads * head_size

    qkv = torch.randn(
        num_tokens,
        q_hidden_size + kv_hidden_size * 2,
        dtype=dtype,
        device=device
    )
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)

    cos_sin_cache = torch.from_numpy(
        np.random.uniform(0, 1, [max_position_embeddings, head_size])
    ).to(dtype).npu()
    positions = torch.randint(
        low=0, high=max_position_embeddings,
        size=(num_tokens,), dtype=torch.int64, device=device
    )

    q, k, v = split_qkv_rmsnorm_rope_impl(
        input=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_size,
        eps=eps,
        cos_sin_cache=cos_sin_cache,
        positions=positions
    )

    cos, sin = cos_sin_cache.index_select(0, positions).view(num_tokens, 2, -1).repeat(1, 1, 2).chunk(2, dim=-2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    _q, _k, v_gold = qkv.cpu().split(
        [q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1
    )
    _q = rms_norm(_q.reshape(-1, head_size), q_weight.cpu(), eps)
    _k = rms_norm(_k.reshape(-1, head_size), k_weight.cpu(), eps)
    _q = _q.reshape(num_tokens, 1, -1, head_size)
    _k = _k.reshape(num_tokens, 1, -1, head_size)

    q_gold, k_gold = custom_rope(_q, _k, sin.cpu(), cos.cpu())
    q_gold = q_gold.reshape(num_tokens, -1)
    k_gold = k_gold.reshape(num_tokens, -1)

    print("=== Verify Q output ===")
    torch.testing.assert_close(
        q.to(torch.float32).cpu(),
        q_gold,
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL
    )
    print("Q precision check passed")

    print("=== Verify K output ===")
    torch.testing.assert_close(
        k.to(torch.float32).cpu(),
        k_gold,
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL
    )
    print("K precision check passed")

    print("=== Verify V output ===")
    torch.testing.assert_close(
        v.to(torch.float32).cpu(),
        v_gold.to(torch.float32),
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL
    )
    print("V precision check passed")

    print("\nAll checks passed!")

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


if __name__ == "__main__":
    with torch.inference_mode():
        run_single_test()
