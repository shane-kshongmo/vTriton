#!/usr/bin/env python3
"""
Standalone test wrapper for split_qkv_rmsnorm_mrope kernel.
Works on both CANN 8.5 (built-in tl.extract_slice) and CANN 9 (patched).
"""
import torch
import triton
import triton.language as tl

# ── CANN 9 compat: import vllm-ascend extensions from triton_patch ──────────
if not hasattr(tl, "extract_slice"):
    try:
        from triton.triton_patch.language.core import extract_slice, insert_slice
        tl.extract_slice = extract_slice
        tl.insert_slice  = insert_slice
    except ImportError:
        def _extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None):
            return ful
        def _insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None):
            return ful
        tl.extract_slice = _extract_slice
        tl.insert_slice  = _insert_slice

# ── Stub for vllm_ascend dependency ─────────────────────────────────────────
def get_vectorcore_num():
    return 20  # Ascend 910B / 910B3

# ── Kernel (copied from split_qkv_rmsnorm_mrope.py) ─────────────────────────
@triton.jit(
    do_not_specialize=["num_tokens", "front_core_num", "num_tokens_each_front_core", "num_tokens_each_tail_core"]
)
def split_qkv_rmsnorm_mrope_kernel(
    in_qkv_ptr: torch.Tensor,
    q_weight_ptr: torch.Tensor,
    q_bias_ptr: torch.Tensor,
    k_weight_ptr: torch.Tensor,
    k_bias_ptr: torch.Tensor,
    cos_sin_ptr: torch.Tensor,
    out_q_ptr: torch.Tensor,
    out_k_ptr: torch.Tensor,
    out_v_ptr: torch.Tensor,
    out_gate_ptr: torch.Tensor,
    num_tokens,
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    eps: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    has_bias: tl.constexpr,
    is_interleaved: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    gate_size: tl.constexpr,
):
    block_idx = tl.program_id(0)

    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core

    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num
            + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))

    if has_bias:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))

    for index in range(loop_num):
        in_q_offset = in_qkv_ptr + (block_offset + index) * (q_size + gate_size + 2 * kv_size)
        if gate_size > 0:
            in_q_gate_tensor = (
                tl.load(in_q_offset + tl.arange(0, q_size + gate_size))
                .to(tl.float32)
                .reshape(num_q_heads, head_size * 2)
            )
            in_q_tensor = tl.extract_slice(
                in_q_gate_tensor, offsets=(0, 0), sizes=(num_q_heads, head_size), strides=(1, 1),
            )
            in_gate_tensor = tl.extract_slice(
                in_q_gate_tensor, offsets=(0, head_size), sizes=(num_q_heads, head_size), strides=(1, 1),
            ).reshape(q_size)
        else:
            in_q_tensor = (
                tl.load(in_q_offset + tl.arange(0, q_size)).to(tl.float32).reshape(num_q_heads, head_size)
            )

        in_k_offset = in_q_offset + q_size + gate_size
        in_k_tensor = tl.load(in_k_offset + tl.arange(0, kv_size)).to(tl.float32).reshape(num_kv_heads, head_size)

        in_v_offset = in_k_offset + kv_size
        in_v_tensor = tl.load(in_v_offset + tl.arange(0, kv_size))

        cos_offsets = tl.arange(0, half_rope_dim)
        if is_interleaved:
            h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
            w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
        else:
            t_mask = cos_offsets < mrope_section_t
            h_mask = (mrope_section_t - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h)
            w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (
                cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
            )

        t_cos_offset = cos_sin_ptr + (block_offset + index) * rope_dim
        h_cos_offset = t_cos_offset + num_tokens * rope_dim
        w_cos_offset = h_cos_offset + num_tokens * rope_dim

        t_sin_offset = cos_sin_ptr + (block_offset + index) * rope_dim + half_rope_dim
        h_sin_offset = t_sin_offset + num_tokens * rope_dim
        w_sin_offset = h_sin_offset + num_tokens * rope_dim

        t_cos_tensor = tl.load(t_cos_offset + cos_offsets, mask=t_mask, other=0)
        h_cos_tensor = tl.load(h_cos_offset + cos_offsets, mask=h_mask, other=0)
        w_cos_tensor = tl.load(w_cos_offset + cos_offsets, mask=w_mask, other=0)
        t_sin_tensor = tl.load(t_sin_offset + cos_offsets, mask=t_mask, other=0)
        h_sin_tensor = tl.load(h_sin_offset + cos_offsets, mask=h_mask, other=0)
        w_sin_tensor = tl.load(w_sin_offset + cos_offsets, mask=w_mask, other=0)

        cos_tensor = (t_cos_tensor + h_cos_tensor + w_cos_tensor).to(tl.float32).reshape(1, half_rope_dim)
        cos_tensor = tl.broadcast_to(cos_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

        sin_tensor = (t_sin_tensor + h_sin_tensor + w_sin_tensor).to(tl.float32).reshape(1, half_rope_dim)
        sin_tensor = tl.broadcast_to(sin_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

        # q-rmsnorm
        squares = in_q_tensor * in_q_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_q_heads, 1)
        q_normalized = in_q_tensor * reciprocal_std
        q_normalized = q_normalized * q_rmsnorm_weight
        if has_bias:
            q_normalized = q_normalized + q_bias

        # k-rmsnorm
        squares = in_k_tensor * in_k_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_kv_heads, 1)
        k_normalized = in_k_tensor * reciprocal_std
        k_normalized = k_normalized * k_rmsnorm_weight
        if has_bias:
            k_normalized = k_normalized + k_bias

        # q-mrope
        x1 = tl.extract_slice(q_normalized, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        x2 = tl.extract_slice(q_normalized, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        cat_x = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
        cat_x = tl.insert_slice(cat_x, -x2, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        cat_x = tl.insert_slice(cat_x, x1, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        if IS_PARTIAL_ROPE:
            orig_qk = tl.extract_slice(q_normalized, offsets=(0, 0), sizes=(num_q_heads, rope_dim), strides=(1, 1))
        else:
            orig_qk = q_normalized
        roped_q = cat_x * sin_tensor + orig_qk * cos_tensor

        # k-mrope
        y1 = tl.extract_slice(k_normalized, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        y2 = tl.extract_slice(k_normalized, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        cat_y = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
        cat_y = tl.insert_slice(cat_y, -y2, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        cat_y = tl.insert_slice(cat_y, y1, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        if IS_PARTIAL_ROPE:
            orig_qk = tl.extract_slice(k_normalized, offsets=(0, 0), sizes=(num_kv_heads, rope_dim), strides=(1, 1))
        else:
            orig_qk = k_normalized
        roped_k = cat_y * sin_tensor + orig_qk * cos_tensor

        if IS_PARTIAL_ROPE:
            q_normalized = tl.insert_slice(q_normalized, roped_q, offsets=(0, 0), sizes=(num_q_heads, rope_dim), strides=(1, 1))
            k_normalized = tl.insert_slice(k_normalized, roped_k, offsets=(0, 0), sizes=(num_kv_heads, rope_dim), strides=(1, 1))
        else:
            q_normalized = roped_q
            k_normalized = roped_k

        # store out_q
        out_q_offset = out_q_ptr + (block_offset + index) * q_size
        tl.store(out_q_offset + tl.arange(0, q_size), q_normalized.reshape(q_size))

        # store out_k
        out_k_offset = out_k_ptr + (block_offset + index) * kv_size
        tl.store(out_k_offset + tl.arange(0, kv_size), k_normalized.reshape(kv_size))

        # store out_v
        out_v_offset = out_v_ptr + (block_offset + index) * kv_size
        tl.store(out_v_offset + tl.arange(0, kv_size), in_v_tensor)

        if gate_size > 0:
            out_gate_offset = out_gate_ptr + (block_offset + index) * gate_size
            tl.store(out_gate_offset + tl.arange(0, gate_size), in_gate_tensor)


# ── Build inputs and run (for launcher) ─────────────────────────────────────
def build_inputs():
    num_tokens = 2048
    num_q_heads = 32
    num_kv_heads = 8
    head_size = 128
    rope_dim = 64  # half of head_size -> partial rope
    gate_size = 0  # no gate
    q_size = num_q_heads * head_size   # 4096
    kv_size = num_kv_heads * head_size  # 1024
    has_bias = False

    try:
        device = torch.device("npu:0")
        _ = torch.randn(1, device=device)  # test
    except Exception:
        device = torch.device("cpu")

    in_qkv = torch.randn((num_tokens, q_size + gate_size + 2 * kv_size), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((head_size,), device=device, dtype=torch.bfloat16)
    k_weight = torch.randn((head_size,), device=device, dtype=torch.bfloat16)
    cos_sin = torch.randn((3 * num_tokens * rope_dim,), device=device, dtype=torch.float32)
    out_q = torch.empty((num_tokens, q_size), device=device, dtype=torch.bfloat16)
    out_k = torch.empty((num_tokens, kv_size), device=device, dtype=torch.bfloat16)
    out_v = torch.empty((num_tokens, kv_size), device=device, dtype=torch.bfloat16)
    out_gate = torch.empty((num_tokens, gate_size), device=device, dtype=torch.bfloat16)

    core_num = get_vectorcore_num()
    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num
    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num
    tail_core_num = core_num - front_core_num if num_tokens > core_num else 0
    num_tokens_each_tail_core = num_tokens // core_num
    block_dim = core_num
    total_core = front_core_num + tail_core_num
    if total_core < core_num:
        block_dim = total_core

    eps = 1e-6
    mrope_section = [24, 24, 16]
    is_interleaved = False
    IS_PARTIAL_ROPE = rope_dim != head_size

    return {
        "in_qkv_ptr": in_qkv,
        "q_weight_ptr": q_weight,
        "q_bias_ptr": torch.zeros((1,), device=device),  # dummy, not used
        "k_weight_ptr": k_weight,
        "k_bias_ptr": torch.zeros((1,), device=device),   # dummy, not used
        "cos_sin_ptr": cos_sin,
        "out_q_ptr": out_q,
        "out_k_ptr": out_k,
        "out_v_ptr": out_v,
        "out_gate_ptr": out_gate,
        "num_tokens": num_tokens,
        "front_core_num": front_core_num,
        "num_tokens_each_front_core": num_tokens_each_front_core,
        "num_tokens_each_tail_core": num_tokens_each_tail_core,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "q_size": q_size,
        "kv_size": kv_size,
        "eps": eps,
        "mrope_section_t": mrope_section[0],
        "mrope_section_h": mrope_section[1],
        "mrope_section_w": mrope_section[2],
        "has_bias": has_bias,
        "is_interleaved": is_interleaved,
        "rope_dim": rope_dim,
        "half_rope_dim": rope_dim // 2,
        "IS_PARTIAL_ROPE": IS_PARTIAL_ROPE,
        "gate_size": gate_size,
        "grid": (block_dim,),
    }

# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = build_inputs()
    grid = data.pop("grid")
    split_qkv_rmsnorm_mrope_kernel[grid](**data)
    print(f"Kernel launched: grid={grid}, num_tokens={data['num_tokens']}")
