#!/bin/bash
# Batch CCE benchmarks for PIPE_V opcodes.
# Each opcode: run on NPU with msprof, extract per-pipe time, compute cycles.
set -euo pipefail
cd /home/triton_sim/vTriton
mkdir -p test/benchmarks/output
OUT=test/benchmarks/output
rm -rf $OUT/*
mkdir -p $OUT

CLOCK=1.85
SIMD=128
N_ITER=5000

cat > /tmp/bench_kernels.py << 'KERNEL_EOF'
import torch, triton, triton.language as tl, sys
N_ITER = 5000; SIMD = 128

@triton.jit
def bench_add(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1.0, tl.float16)
    for _ in range(n): a = a + a
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_mul(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1.0, tl.float16)
    for _ in range(n): a = a * a
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_sub(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1.0, tl.float16)
    for _ in range(n): a = a - a
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_div(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1.0, tl.float16)
    b = tl.full((SIMD,), 2.0, tl.float16)
    for _ in range(n): a = a / b
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_brc(out_ptr, n: tl.constexpr):
    a = tl.zeros((SIMD,), tl.float16)
    for _ in range(n):
        a = tl.full((SIMD,), 1.0, tl.float16)
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_cast(out_ptr, n: tl.constexpr):
    a = tl.zeros((SIMD,), tl.float16)
    for _ in range(n):
        v = tl.zeros((SIMD,), tl.float32)
        a = v.to(tl.float16)
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_cmp(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1.0, tl.float16)
    b = tl.full((SIMD,), 0.5, tl.float16)
    for _ in range(n):
        m = a > b; a = tl.where(m, a, b)
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_sel(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1.0, tl.float16)
    b = tl.full((SIMD,), 2.0, tl.float16)
    m = tl.full((SIMD,), 1, tl.int32)
    for _ in range(n):
        a = tl.where(m, a, b)
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_reduce(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1.0, tl.float16)
    for _ in range(n):
        s = tl.sum(a)
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_and(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1, tl.int32)
    b = tl.full((SIMD,), 1, tl.int32)
    for _ in range(n): a = a & b
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_or(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 1, tl.int32)
    b = tl.full((SIMD,), 2, tl.int32)
    for _ in range(n): a = a | b
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_not(out_ptr, n: tl.constexpr):
    a = tl.full((SIMD,), 0, tl.int32)
    for _ in range(n): a = ~a
    tl.store(out_ptr + tl.arange(0, SIMD), a)

@triton.jit
def bench_arange(out_ptr, n: tl.constexpr):
    for _ in range(n):
        a = tl.arange(0, SIMD)
    tl.store(out_ptr + tl.arange(0, SIMD), a)

TESTS = {
    "vmul": bench_mul, "vadd": bench_add, "vsub": bench_sub, "vdiv": bench_div,
    "vbrc": bench_brc, "vcast": bench_cast, "vcmp": bench_cmp, "vsel": bench_sel,
    "vreduce": bench_reduce, "vand": bench_and, "vor": bench_or, "vnot": bench_not,
    "varange": bench_arange,
}

name = sys.argv[1]
kernel = TESTS[name]
t = torch.empty(SIMD, device="npu:0", dtype=torch.float16)
kernel[(1,)](t, N_ITER)
torch.npu.synchronize()
print("DONE")
KERNEL_EOF

unset TRITON_COMPILE_ONLY
export TRITON_ENABLE_TASKQUEUE=0
export TRITON_ASCEND_ARCH=Ascend910_9362
rm -rf /root/.triton/cache/*

echo "=== CCE Benchmarks: PIPE_V opcodes ==="
echo "SIMD=128 N_ITER=$N_ITER Clock=$CLOCK GHz"
echo ""

OPS=("vmul" "vadd" "vsub" "vdiv" "vbrc" "vcast" "vcmp" "vsel" "vreduce" "vand" "vor" "vnot" "varange")

for op in "${OPS[@]}"; do
    echo -n "$op: "
    rm -rf /tmp/bench_prof
    mkdir -p /tmp/bench_prof
    rm -rf /root/.triton/cache/*

    msprof --output=/tmp/bench_prof --aic-metrics=PipeUtilization --ai-core=on \
        --aic-mode=task-based --task-time=on \
        python3 /tmp/bench_kernels.py "$op" 2>/tmp/bench_stderr.txt >/tmp/bench_stdout.txt

    # Extract aiv_vec_time from op_summary.csv
    CSV=$(find /tmp/bench_prof -name 'op_summary*.csv' 2>/dev/null | head -1)
    if [ -n "$CSV" ] && [ -s "$CSV" ]; then
        # Get last row, column aiv_vec_time(us)
        LINE=$(tail -1 "$CSV")
        VEC_TIME=$(echo "$LINE" | python3 -c "import sys; r=sys.stdin.read().split(','); print(r[34] if len(r)>34 else '0')" 2>/dev/null)
        if [ "$VEC_TIME" != "0" ] && [ "$VEC_TIME" != "N/A" ] && [ -n "$VEC_TIME" ]; then
            cycles=$(python3 -c "print(round(float($VEC_TIME) * $CLOCK * 1000 / $N_ITER, 1))")
            gflops=$(python3 -c "print(round($N_ITER * $SIMD / (float($VEC_TIME) * 1e6) * 1e3, 2))")
            echo "vec_time=${VEC_TIME}us cycles_per_op=${cycles} GFLOPS=${gflops}"
        else
            echo "NO_VEC_TIME"
        fi
    else
        echo "NO_CSV"
    fi
done

echo ""
echo "[DONE]"
