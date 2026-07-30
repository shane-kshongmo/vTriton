# perfbound — Two-Tier Analytical Performance Upper-Bound Model
#
# Computes a provably-conservative lower bound on execution time for
# Triton kernels on Ascend NPU (910B3).
#
# Two-tier bound (sound form — see bound_combiner.py and spec §4.1/§A.5):
#   T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
# T_serial attaches INSIDE the max (it is intra-core). The additive shorthand
# max(grid, core) + serial is non-conservative (max(a,b)+c >= max(a, b+c)) and
# must not be used.
#
# Modules:
#   calibration/  — M1: hardware constant calibration database
#   extract/      — M2/M3: DSL grid extractor + HIVM component extractor
#   model/        — M4: grid + component analytical models (pure functions)
#   combine/      — M5: bound combiner + five-way attribution + two-limit
#   validate/     — M6: soundness/tightness validation harness
#
# The model never compiles or runs kernels; measurement enters only
# via calibration (M1) and validation (M6).

__version__ = "0.1.0"
