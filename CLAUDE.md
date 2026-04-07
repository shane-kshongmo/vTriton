# Project Settings

## Project Summary
This is a Triton operator performance modelling project for Ascend NPU(A2/A3, A5 hardware). A tile level white box model, seamlessly adapts to TTIR. The generated tools tritonsim-opt for performance modelling; ascend-tiling-opt for tiling optimization based on the white box model.

## Technology Stack
- CPP
- MLIR
- AI Compiler
- Triton
- Operator Performance Modelling

## Project Structure (Core)
```
├── lib/
|   ├── AscendModel/
|   |   ├── Analysis/
|   |   |   ├── CMakeLists.txt
|   |   |   ├── HardwareConfig.cpp
|   |   |   ├── MemoryTilingOptimizer.cpp
|   |   |   ├── PipelineAnalysis.cpp
|   |   |   ├── RooflineAnalysis.cpp
|   |   |   └── UnifiedTilingCostModel.cpp
|   |   ├── IR/
|   |   |   ├── AscendModelDialect.cpp
|   |   |   ├── AscendModelOps.cpp
|   |   |   └── CMakeLists.txt
|   |   ├── Transforms/
|   |   |   ├── AssignOpIDs.cpp
|   |   |   ├── CMakeLists.txt
|   |   |   ├── ConvertTritonToAscend.cpp
|   |   |   ├── EstimateCycles.cpp
|   |   |   ├── InsertDataTransfers.cpp
|   |   |   ├── PassRegistration.cpp
|   |   |   ├── PerfReportPass.cpp
|   |   |   ├── PipelineAnalysisPass.cpp
|   |   |   └── TilingOptimizationPass.cpp
|   |   └── CMakeLists.txt
|   └── CMakeLists.txt
```
## Important Documentations
- README.md
- BUILD.md

## Notes
- If project is not built, build first according ro BUILD.md
- If the host is Windows, prefered to run the build and tools in WSL.

## Running tritonsim-opt in WSL (Windows Environment)

All binaries (`tritonsim-opt`, `triton-opt`) are Linux ELF binaries and must be run via WSL.

### Key Paths

| Item | Path |
|------|------|
| tritonsim-opt | `/mnt/d/work/git/vTriton/build/bin/tritonsim-opt` |
| triton-opt | `/mnt/d/work/git/triton-ascend/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt` |
| Hardware config | `/mnt/d/work/git/vTriton/configs/ascend_910b.json` |
| WSL distro | Ubuntu-24.04 |

### How to Invoke WSL from Agent

Write a shell script to the WSL filesystem at `//wsl$/Ubuntu-24.04/tmp/run_model.sh` (use the Write tool with that UNC path), then execute via PowerShell:

```bash
powershell.exe -Command "wsl -d Ubuntu-24.04 bash /tmp/run_model.sh"
```

**Important:** Do NOT pass Linux paths as direct arguments to `wsl.exe` from Git Bash — Git Bash converts `/mnt/d/...` to `D:/Program Files/Git/mnt/d/...`. Always use a script file written to the WSL filesystem, or pipe script content via stdin.

**Important:** Do NOT use `$VAR` syntax when writing scripts with the Write tool — the tool expands shell variables at write time. Use `${VAR}` syntax, which is also expanded. Instead, write variables as plain text assignments (no `$` in the value).

### Modelling a TTIR file (Two-Stage Pipeline)

For `.mlir` files containing Triton IR (`tt.*` ops), use the two-stage pipeline:

```bash
#!/bin/bash
TRITON_OPT=/mnt/d/work/git/triton-ascend/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
TRITONSIM_OPT=/mnt/d/work/git/vTriton/build/bin/tritonsim-opt
MLIR_FILE=/path/to/kernel.mlir
HW_CONFIG=/mnt/d/work/git/vTriton/configs/ascend_910b.json

# Bind dynamic kernel arguments (e.g., arg7=4096 for N dimension)
# arg-bindings: use argN=value to bind function argument %argN
# Inspect the .mlir to find which argN controls the loop upper bound

TRITON_OPT_PATH=<value of TRITON_OPT>
TRITONSIM_PATH=<value of TRITONSIM_OPT>

${TRITON_OPT_PATH} ${MLIR_FILE} --allow-unregistered-dialect --mlir-print-op-generic | \
${TRITONSIM_PATH} - --allow-unregistered-dialect \
  -ascend-perf-model="hardware-config=${HW_CONFIG} arg-bindings=arg7=4096" 2>&1
```

Write this as a literal script (no shell variable expansion issues) to `//wsl$/Ubuntu-24.04/tmp/run_model.sh`.

### Determining arg-bindings

Inspect the `.mlir` file for `scf.for` loops with dynamic upper bounds. The upper bound SSA value (e.g., `%arg7`) maps to a function argument. Find the argument index and bind it:
- `%arg7` → `arg-bindings=arg7=<N>`
- Common values: N=4096 for hidden dim in LLMs

If all loops have static bounds, `arg-bindings` is not needed.

### Modelling an AscendModel dialect file directly

```bash
${TRITONSIM_PATH} input.mlir -ascend-perf-model="hardware-config=${HW_CONFIG}" 2>&1
```

### Rebuilding after source changes

Add this before the modelling command in the script:
```bash
cd /mnt/d/work/git/vTriton/build && ninja -j$(nproc)
```

## Local Environment

Prefer the machine-local setup documented in `LOCAL_ENVIRONMENT.md` before attempting any fresh dependency bootstrap. If that file is missing, first search for an existing venv under `/mnt/c/Users/shane` and a CANN install under `~/Ascend`; only build from scratch and download/install CANN if those local installs are not present.
