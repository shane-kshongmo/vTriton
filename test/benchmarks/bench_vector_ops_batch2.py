#!/usr/bin/env python3
"""CCE Benchmarks — remaining opcodes (fixed type issues)."""
import torch, triton, triton.language as tl, json, sys, time
CLOCK = 1.85; SIMD = 128; BLOCKS = 40; N_ITER = 2000; REPS = 3

@triton.jit
def b_div(o, n: tl.constexpr, S: tl.constexpr):
    p = tl.program_id(0); a = tl.full((S,), 1.0, dtype=tl.float32)
    b = tl.full((S,), 2.0, dtype=tl.float32)
    for _ in range(n): a = a / b
    tl.store(o + p * S + tl.arange(0, S), a.to(tl.float16))

@triton.jit
def b_and(o, n: tl.constexpr, S: tl.constexpr):
    p = tl.program_id(0); a = tl.full((S,), 1, dtype=tl.int32)
    b = tl.full((S,), 1, dtype=tl.int32)
    for _ in range(n): a = a & b
    tl.store(o + p * S + tl.arange(0, S), a)

@triton.jit
def b_or(o, n: tl.constexpr, S: tl.constexpr):
    p = tl.program_id(0); a = tl.full((S,), 1, dtype=tl.int32)
    b = tl.full((S,), 2, dtype=tl.int32)
    for _ in range(n): a = a | b
    tl.store(o + p * S + tl.arange(0, S), a)

@triton.jit
def b_not(o, n: tl.constexpr, S: tl.constexpr):
    p = tl.program_id(0); a = tl.full((S,), 0, dtype=tl.int32)
    for _ in range(n): a = ~a
    tl.store(o + p * S + tl.arange(0, S), a)

@triton.jit
def b_arange(o, n: tl.constexpr, S: tl.constexpr):
    p = tl.program_id(0)
    for _ in range(n):
        a = tl.arange(0, S).to(tl.float16)
        tl.store(o + p * S + tl.arange(0, S), a)

TESTS = [("vdiv",b_div),("vand",b_and),("vor",b_or),("vnot",b_not),("varange",b_arange)]

import torch_npu; dev = torch.device("npu:0")
out = torch.empty(BLOCKS * SIMD, device=dev, dtype=torch.float16)
print(f"CCE Benchmarks (batch 2): {len(TESTS)} ops")
print(f"{'op':10s} {'ms':>8s} {'cyc/op':>8s} {'GFLOPS':>8s} {'std%':>6s}")
print("-"*50); sys.stdout.flush()

results = {}
for name, kernel in TESTS:
    times = []
    for r in range(REPS + 1):
        kernel[(BLOCKS,)](out, 10 if r == 0 else N_ITER, SIMD); torch.npu.synchronize()
        if r == 0: continue
        t0 = time.perf_counter()
        kernel[(BLOCKS,)](out, N_ITER, SIMD); torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times)/len(times)
    std = (sum((t-avg)**2 for t in times)/len(times))**0.5
    cyc = avg * CLOCK * 1e9 / (BLOCKS * N_ITER)
    gflops = BLOCKS * N_ITER * SIMD / avg / 1e9
    cv = std/avg*100
    print(f"{name:10s} {avg*1e3:8.2f} {cyc:8.1f} {gflops:8.2f} {cv:6.1f}")
    sys.stdout.flush()
    results[name] = {"cycles_per_128elem": round(cyc,1), "GFLOPS": round(gflops,2), "std%": round(cv,1)}

# Merge with existing results
import os; p="/home/triton_sim/vTriton/test/benchmarks/output/bench_results.json"
if os.path.exists(p):
    existing = json.load(open(p)); existing.update(results)
    json.dump(existing, open(p,"w"), indent=2)
else:
    json.dump(results, open(p,"w"), indent=2)
print(f"Saved {len(results)} results")
