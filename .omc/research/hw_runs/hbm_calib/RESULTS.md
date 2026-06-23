# Peak-HBM Bandwidth + Contention Calibration — Results (2026-06-23)

Executes `.omc/plans/peak_hbm_contention_calibration.md` on the real **910B3**
(ssh `910B3`, 40 AIV + 20 AIC cores, CANN 9.0.0, conda `triton_hxl`). Closes the
**absolute** Gap-2 gate (US-SB-006) soundly, without mis-binding compute kernels.

Device: `Ascend910B3`, `cube_core_num=20`, `vector_core_num=40`, L2=192 MB.

## M-1 — True peak HBM bandwidth (optimal contiguous copy)

`test/stream_copy_bench.py` (one contiguous slab/program, large-tile streaming,
device-timed warmup=8 iters=40 median; copy `verified=True`). Aggregate R+W BW:

| block (elem) | tile=512 | tile=2048 | tile=8192 | tile=16384 |
|---|---|---|---|---|
| 262144 | 369 | 615 | 668 | 695 |
| 524288 | 420 | 878 | 1140 | **1148** |
| 1048576 | 374 | 741 | 847 | 870 |

**Peak ≈ 1148–1167 GB/s** (vs the overhead-bound `amort` kernel's 324 GB/s). The
1 MB-block row (192 MB footprint ≈ L2) is lower, so the peak is partly
L2-assisted; we adopt the **max** observed (sound: it is the fastest achievable,
so no kernel beats the resulting floor). Canonical peak = **1167.25 GB/s** (η
sweep, below).

## M-2 — Contention curve (per-core BW vs active cores)

block=524288, tile=8192, n_cores=40:

| nprog | 1 | 2 | 4 | 8 | 16 | 20 | 24 | 32 | 40 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agg GB/s | 29 | 59 | 117 | 233 | 464 | 580 | 694 | 921 | **1126** | 1123 | 879 |
| per-core | 29.1 | 29.2 | 29.3 | 29.1 | 29.0 | 29.0 | 28.9 | 28.8 | 28.2 | 28.1 | 22.0 |

**Per-core BW is flat (~29 GB/s) and aggregate scales linearly to ~1126 GB/s at
40 cores, then saturates.** So the occupancy model is the closed form
`per_core_eff(active) = min(single_core_peak, hbm_peak / active)` — no
interpolated contention table needed. Note per-core under full occupancy
(1167/40 ≈ 29.2) equals the measured ~29: the existing single-core constant
(86.5 GB/s) is the *unsaturated* ceiling; HBM contention throttles it to ~29 at
full occupancy. n=64 oversubscription degrades (expected).

## M-3 — Packet-efficiency curve re-anchored to the true peak

Fine tile sweep at the peak config (block=524288, grid=48); η = achieved_agg /
peak (peak=1167.25 GB/s). Bandwidth-bound, so η isolates the small-packet
penalty (unlike the overhead-bound amort curve it supersedes):

| pkt B | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|---|---|---|---|---|
| η | 0.039 | 0.074 | 0.138 | 0.241 | 0.390 | 0.583 | 0.761 | 0.945 | 0.989 | 1.000 |

Shipped in `perfbound/calibration/data/calib_910b3_v1.json` (`memory.pkt_efficiency`,
`memory.hbm_peak_aggregate_bw_gbps`).

## Model change (occupancy-aware MTE bandwidth)

- `MemHierarchy.mte_effective_bw(single_core_bw, active_cores) =
  min(single_core_bw, hbm_peak_aggregate / active_cores)` — applied to
  HBM-touching paths (gm) only; intra-chip paths (l1→l0a) unthrottled.
- `bounds.py`: `active_cores = min(total_programs, n_cores)` threaded into both
  tiers; the MTE_GM grid `i_binding` now derives from the (throttled) component
  floor, so **grid and component tiers use the same achievable rate**
  (supersedes the flat `BW_hbm_allcore_sustained`).
- Both ceilings are peak-achievable ⇒ the floor stays a **sound lower bound**.

### Soundness regression
Full `tests/perfbound/`: **415 passed, 3 skipped, 2 xfailed**. The multi-kernel
validation set (softmax/rmsnorm/layernorm/vector_add/chunk_kda) keeps
`soundness_rate == 1.0` (all `T_bound ≤ T_measured`). chunk_kda re-binds
vector→mte_ub (per-core gm↔ub throttled 86.5→58 GB/s at 20-core occupancy: mte_ub
18.7→26.7 us/prog now exceeds vector 20.6) — tighter, still sound (5466 ≤
104326 us). matmul golden 4.546→5.265 us (steeper true-peak η on 4–8 KB packets).

## US-SB-006 — absolute Gap-2 gate: CLOSED (held-out, non-circular)

Counterfactual = coalesce the transfer (Triton recompile, tile constexpr
512→65536), real launch + **output-verified** on 910B3. Held-out cross-validation
(interior packets 1024 B & 16384 B removed from the curve before interpolation):

| held-out pkt | η measured | η predicted | error |
|---|---|---|---|
| 1024 B | 0.2411 | 0.2315 | **4.0 %** |
| 16384 B | 0.9447 | 0.8677 | **8.2 %** |

**Counterfactual delta** (seed = held-out 1024 B vs coalesced 65536 B, η≡1):
`bound_delta = 572.6 us`, `measured_delta = 543.0 us` →
**quantification_error = 5.4 % < 0.20**. ✅

CI fixture + test: `tests/perfbound/test_gap2_absolute_validation.py` (6 tests),
evidence `tests/perfbound/fixtures/stream_copy_eta_sweep_910b3.json`.

Note: this counterfactual is a *source-level coalescing recompile* (Triton tile
constexpr), not a des.json `HivmEdit`→bishengir edit — but it is the
compiler-reachable hardware recompile/launch/verify the story's notes asked for,
on a bandwidth-bound kernel where the absolute bound is anchorable.

## US-SB-008 — unchanged (documented-miss)
Independent of this plan; see `.omc/plans/peak_hbm_contention_calibration.md` §6.
