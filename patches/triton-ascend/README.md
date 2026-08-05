# Three-stage tiling candidate expansion

This patch extends the official Triton-Ascend TileGenerator candidate pool
through three optional cumulative expansion stages.

## Scope

- Preserves the official TileGenerator output as the B0 baseline.
- Adds staged local, resource-aware, and evolutionary candidate expansion.
- Adds conservative hardware-capacity pruning.
- Supports strict B0 baseline and uniform-pruning ablation modes.
- Records candidate origins, stage statistics, predicted costs, and pruning reasons.
- Includes Flash Attention comparison tooling and unit tests.

## Files introduced by the patch

- third_party/ascend/backend/runtime/candidate_expansion.py
- third_party/ascend/backend/runtime/hardware_pruning.py
- third_party/ascend/experiments/fa_official_auto_vs_three_stage.py
- third_party/ascend/experiments/operator_explanations.py
- third_party/ascend/unittest/autotune_ut/test_candidate_expansion.py
- third_party/ascend/unittest/autotune_ut/test_hardware_pruning.py

The patch also integrates the expansion pipeline into:

- third_party/ascend/backend/runtime/autotuner.py

## Compatibility

The patch is generated against the Triton-Ascend submodule commit pinned by
vTriton:

a2b1636dafbf43c658f2fb13b21ca03a403e6e7d

Candidate expansion is disabled by default, preserving existing behavior.

## Applying manually

From the Triton-Ascend source directory:

    git apply /path/to/0001-add-three-stage-tiling-candidate-expansion.patch