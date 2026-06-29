# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vTriton (a.k.a. **TritonSim**) is an MLIR-based **performance modelling tool for Ascend NPU** (910B / 910B3, plus A5). It produces a *provably conservative lower bound* on kernel execution time for Triton kernels. The project has **two cooperating layers**:

- **C++ / MLIR layer** (`lib/AscendModel/`, `include/`, `tools/`) — the `AscendModel` dialect, transform passes, and CLI tools (`tritonsim-opt`, `tritonsim-hivm`). Parses TTIR / HIVM IR, classifies ops, estimates cycles, does pipeline/DES scheduling analysis.
- **Python layer** (`perfbound/`) — a zero-MLIR-dependency analytical bound model. This is the **active development focus**; it never compiles or runs kernels — measurement enters only via calibration (M1) and validation (M6).

The two layers communicate via **JSON** emitted by C++ passes and consumed by Python extractors.

### The bound model (core idea)

```
T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
```
- `T_grid_floor` — chip-grid floor (occupancy, load-balance, bandwidth/FLOP ceilings)
- `T_core_floor` — per-core component floor (weighted harmonic-mean Roofline over components)
- `T_serial_irreducible` — unavoidable cross-component handshakes

> Implemented **sound** form: `T_serial` attaches to the Tier-2 term because handshakes are intra-core (Cube↔Vector). The spec prose / `perfbound/__init__.py` use an additive shorthand `max(grid, core)+serial`, which is non-conservative (can violate `T_bound ≤ T_measured`); `bound_combiner.py` implements the form above. See `docs/ARCHITECTURE.md` §1.

`perfbound/` is organized into six stages (M1–M6): `calibration/` → `extract/` (DSL grid + HIVM component) → `model/` → `combine/` → `validate/`. See `docs/ARCHITECTURE.md` for the full deep-dive (it is the authoritative architecture reference; consult it before non-trivial modelling work).

## Build

First build LLVM/MLIR once (the pinned commit is `b5cc222d`, dictated by triton-ascend):

```bash
git submodule update --init thirdparty/triton-ascend
git -C thirdparty/triton-ascend submodule update --init --depth 1 third_party/ascend/AscendNPU-IR
./scripts/apply_patches.sh
./scripts/build_llvm.sh          # ~30–60 min, ~30 GB, one-time
```

Then build TritonSim (CMake auto-discovers `thirdparty/triton-ascend` for Triton dialect support):

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=../thirdparty/llvm-project/build/install/lib/cmake/mlir \
  -DLLVM_DIR=../thirdparty/llvm-project/build/install/lib/cmake/llvm
ninja
```

- **Without Triton** (faster; AscendModel IR input only): add `-DTRITONSIM_ENABLE_TRITON=OFF`.
- **CMake auto-detects** MLIR/LLVM from the AscendNPU-IR fast-build layout if you don't pass `-DMLIR_DIR`/`-DLLVM_DIR`.
- After C++ source changes, rebuild incrementally: `cd build && ninja -j$(nproc)`.
- Binaries land in `build/bin/` (`tritonsim-opt`, `tritonsim-hivm`). `ascend-tiling-opt` exists in source but is not currently built.

Full options in `BUILD.md`; dependency graph in `DEPENDENCIES.md`.

## Running the C++ tools

```bash
# AscendModel dialect MLIR — full pipeline
./build/bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model

# ...with a hardware config (option goes on the pipeline, not as a global flag)
./build/bin/tritonsim-opt test/ascend_ops.mlir \
  -ascend-perf-model="hardware-config=configs/ascend_910b.json"

# Step-by-step passes
./build/bin/tritonsim-opt test/ascend_ops.mlir -assign-op-ids -estimate-cycles -analyze-pipeline -perf-report

# HIVM (.npuir.mlir) direct analysis + Perfetto trace
./build/bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir --perfetto-trace-file /tmp/t.json

# Triton DSL → compile-only HIVM dump → analysis (needs triton-ascend Python env)
./build/bin/tritonsim-hivm --triton-script test/triton_smoke.py --python python3
```

### Two-stage TTIR pipeline (when built without Triton dialect)

Mode-B builds can't parse `.ttir` directly. Pipe `triton-opt` (generic MLIR) into `tritonsim-opt`:

```bash
triton-opt kernel.ttir --allow-unregistered-dialect --mlir-print-op-generic | \
./build/bin/tritonsim-opt - --allow-unregistered-dialect \
  -ascend-perf-model="hardware-config=configs/ascend_910b.json arg-bindings=arg7=4096"
```

`arg-bindings=argN=<value>` binds a function argument (typically the runtime upper bound of an `scf.for` loop). Inspect the `.mlir` to find which `%argN` controls the loop bound. If all loops have static bounds, `arg-bindings` is not needed.

## Running the Python bound model (`perfbound/`)

`perfbound` is imported **in-place** (repo root on `sys.path` via `conftest.py` / each script's own `sys.path` insert) — it is not pip-installed. The end-to-end entry point is `scripts/run_bound.py`:

```
kernel.py → Triton NPUIR dump → cleaned NPUIR → DES graph JSON → perfbound JSON report
```

```bash
# By registry name (see perfbound/experiments/registry.py for built-in kernels:
# vector_add, vector_add_2x, softmax, layernorm, rmsnorm, seeded_gap1/gap2/serial)
python scripts/run_bound.py --kernel seeded_serial --grid 128,32

# By script path
python scripts/run_bound.py --script path/to/kernel.py --grid 128,32 \
  --calibration <calib.json> --measured-us <T_measured>
```

`run_bound.py` shells out to `build/bin/tritonsim-hivm` to produce the DES graph; that binary must exist for a real (non-`--dry-run`) run. Kernel scripts must expose `build_inputs()` and `Model` (see `KernelSpec.validate_interface`).

## Tests

```bash
# Python (perfbound) — run from repo root. conftest.py puts repo root on sys.path.
python -m pytest tests/perfbound/                         # full suite
python -m pytest tests/perfbound/test_bounds.py           # one file
python -m pytest tests/perfbound/test_bounds.py::test_xxx # one test
python -m pytest tests/perfbound/ -k chunk_kda            # by keyword

# HIVM sync/component tests
python -m pytest tests/hivm/

# C++ MLIR tests — enable then ctest
cmake -B build -S . -DASCEND_MODEL_ENABLE_TESTS=ON ...     # add to your cmake line
ctest --test-dir build

# Smoke tests
./build/bin/tritonsim-opt test/ascend_ops.mlir -ascend-perf-model
python3 test/triton_smoke.py
```

Python environment note: the repo ships a **Windows-style** `.venv` (it has `Scripts/*.exe`, no `bin/`). In WSL use a Linux interpreter instead — the local conda env is `vtriton-verify` (`~/miniconda3/envs/vtriton-verify`). Hardware-in-the-loop work (M6 validation, remote benchmarking, calibration microbenches) runs on the remote 910B3 box, not locally.

## Configs & calibration data

- `configs/ascend_910b.json`, `configs/ascend_910b3.json` — hardware parameters (clock, memory hierarchy, compute-unit throughput, data-mover bandwidth). Format documented in `configs/README.md`, validated against `configs/hardware_schema.json`. **Default is 910B.**
- `perfbound/calibration/bench_output/*.csv` — measured sustained rates (bandwidth, cube peak, mandatory handoffs, MTE transfers) from `.cce` AscendC microbenchmarks. Calibration constants live in `perfbound/calibration/data/`. See `docs/CALIBRATION_GUIDE.md`.

## Conventions & gotchas

- **`thirdparty/` is upstream code.** Treat it as read-only unless a task explicitly targets it. Local compatibility patches live in `patches/` and are applied via `scripts/apply_patches.sh`.
- **No regex for MLIR/TTIR parsing.** Extract structure via the C++ passes (`ExtractTTIRInfo`) or `lark`/MLIR APIs — regex silently matches the wrong spans.
- **Git safety in this repo:** do **not** use `git add -A` / `git clean -fd` — WSL git hangs on submodules and this has destroyed work. Stage explicit paths. Recover lost objects via `git fsck`.
- **C++ style:** existing LLVM/MLIR C++17 conventions — 2-space indent for wrapped args, braces on their own lines, grouped includes (LLVM/MLIR before STL). `CamelCase` types/passes, `lowerCamelCase` functions/locals. New MLIR test files named `feature_ascend.mlir`.
- **Bound model invariants:** the model must stay *provably conservative* — use sustained (not peak) rates with CIs, and prefer "report as forced" over under-reporting in serial classification. Measurement never enters the model except through M1/M6.
- When you change analysis logic, pass behaviour, or hardware modelling, reproduce with the shipped tools (add a focused `test/` fixture) rather than eyeballing output.

## WSL invocation (Windows host)

All binaries are Linux ELF and run under WSL (`Ubuntu-24.04`). From an agent, write a script to the WSL filesystem then invoke it — **do not** pass Linux paths as direct args to `wsl.exe` from Git Bash (Git Bash rewrites `/mnt/d/...` → `D:/Program Files/Git/mnt/d/...`):

```bash
# 1. Write the script to //wsl$/Ubuntu-24.04/tmp/run_model.sh (Write tool, UNC path)
# 2. Execute:
powershell.exe -Command "wsl -d Ubuntu-24.04 bash /tmp/run_model.sh"
```

When using the Write tool for scripts, write variable assignments as **plain literals** — `$VAR` and `${VAR}` are shell-expanded at write time. Also prepend the rebuild step when iterating on C++:

```bash
cd /mnt/d/work/git/vTriton/build && ninja -j$(nproc)
```

Key local paths: `tritonsim-opt` → `build/bin/tritonsim-opt`; `triton-opt` → `/mnt/d/work/git/triton-ascend/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt`; default hardware config → `configs/ascend_910b.json`.

## Local environment bootstrap

A `LOCAL_ENVIRONMENT.md` is referenced by tooling but is **not currently present**. Before a fresh dependency bootstrap, first look for an existing venv under `/mnt/c/Users/shane` and a CANN install under `~/Ascend`; only build from scratch (and download/install CANN) if those local installs are absent.
