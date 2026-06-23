# Plan — Peak-HBM Bandwidth + Contention Calibration (pursue the absolute Gap-2 gate)

**Goal.** Make perfbound's *absolute* memory-time predictions correct so a Gap-2
counterfactual passes the `quantification_error < 0.20` gate (US-SB-006) **without**
mis-attributing compute kernels as memory-bound. The Gap-2 *mechanism* and the
*efficiency ratio* are already validated (η within ~6% on held-out packets); the
blocker is the absolute bandwidth scale.

**Status this unblocks.** US-SB-006 absolute closure; and, more broadly, the
recurring finding that *every* perfbound gap's absolute hardware validation is
gated by the calibration's sustained-vs-peak / contention treatment.

---

## 1. Why the current model is wrong (evidence)

- `gm→ub` / `ub→gm` are calibrated at **86.5 GB/s** in
  `perfbound/calibration/data/bandwidth_910b3.csv`. That is a **single-core**
  rate (one core, no HBM contention).
- The MTE **component floor** (`compute_component_floor` → `_get_mte_throughput_bytes_per_us`
  → `MemHierarchy.lookup_bw`) uses that single-core rate for **every** kernel,
  regardless of how many cores are active.
- A grid-saturating copy actually achieves far less per core. Measured on 910B3
  (`test/amort_sweep_bench.py`, CHUNK=4096, 2048 programs): **~324 GB/s aggregate
  → ~16 GB/s/core combined R+W** — and that kernel is itself sub-peak.
- Two failure modes result:
  1. **Single-core rate → bound too low** (memory time ~10× under measured; the
     coalesced amort copy bound was 21.8 µs vs measured 206.7 µs).
  2. **Naively lowering bw globally → compute kernels mis-bind to memory** (a
     matmul flipped to `mte_gm`-bound; chunk_kda flipped component→grid). This is
     why the 2026-06-23 global bw recalibration was reverted — it calibrated to a
     *sub-peak* kernel and broke attribution.

The correct model needs **occupancy-aware** bandwidth: per-core effective BW is a
function of the number of contending cores, bounded above by the true single-core
peak and below by the all-core contended rate.

This is the **same sustained-vs-peak tension** already seen on the **vector**
rate (see `[[project-vector-rate-soundness-fix]]`): a sound lower bound needs the
*fastest achievable* rate, but "achievable" is contention/occupancy-dependent.

---

## 2. Measurements to take on 910B3 (the calibration microbenches)

All via the proven Triton+`torch.npu.synchronize()` harness
(`test/amort_sweep_bench.py` is the template; device-side timing, warmup=8,
iters≥40, median). Use **deterministic, CPU-reconstructable inputs**
(`arange % p / p`) so outputs are verifiable without RNG (NPU `randn` ≠ CPU).

### M-1 — True peak HBM bandwidth (per direction)
An **optimal** contiguous copy (NOT the interleaved 2048-program amort kernel):
one big contiguous `gm→ub→gm` stream per program, block ≥ 8 KB, grid sized to
saturate but not oversubscribe. Sweep block size up to the UB limit; take the
**max** achieved aggregate BW as the true peak. Expected ≈ HBM datasheet
(~hundreds of GB/s) — must exceed the amort kernel's 324 GB/s.
**Output:** `BW_gm_ub_peak_aggregate` (GB/s), and `_per_core_single` (one program,
one core, no contention).

### M-2 — Contention curve: per-core BW vs active core count
Same optimal copy, vary the number of active programs/cores
`n ∈ {1, 2, 4, 8, 16, 20, 40, …}` (grid = n, one program/core up to 20, then
waves). Measure aggregate BW(n) → per-core BW(n) = BW(n)/min(n, 20).
**Output:** table `active_cores → per_core_bw` (the contention curve). Expect
per-core BW to fall from the single-core peak toward an all-core asymptote as the
HBM controller saturates.

### M-3 — Packet/amortization curve at the *correct* peak
Re-anchor the existing `pkt_efficiency` table (already measured, in
`calib_910b3_v1.json` `memory.pkt_efficiency`) so η = achieved_bw / **M-1 peak**
(currently η is normalized to the sub-peak amort kernel's own 16384-byte point).
Keep the same packet sweep; just renormalize against M-1. Hold out ≥2 packet
sizes for non-circular validation.

**Acceptance for §2:** each constant measured ≥30× (<5% run-to-run variance);
M-1 peak > amort-kernel 324 GB/s; M-2 monotone non-increasing per-core curve.

---

## 3. Model changes

### 3.1 Occupancy-aware MTE bandwidth (core change)
`compute_component_floor` and `bounds.py` already compute `waves` / occupancy.
Thread the **active-core count** (`min(total_programs, n_cores)`) into the MTE
floor and select per-core BW from the **M-2 contention curve** instead of the
flat single-core constant:
- `perfbound/model/component_model.py::_get_mte_throughput_bytes_per_us` — add an
  `active_cores` arg; look up `memory.contention_bw(active_cores)` (new method,
  interpolates the M-2 table), then apply the `pkt_efficiency` η as today.
- `perfbound/model/bounds.py` — pass `active_cores` through; ensure the **grid
  floor** uses the same contention-aware aggregate (it already uses
  `BW_hbm_allcore_sustained` — reconcile that constant with the M-2 all-core
  asymptote so grid and component tiers agree).
- `perfbound/calibration/constants.py::MemHierarchy` — add `contention_bw` table
  + `contention_bw(active_cores)` interpolator (mirror `_packet_efficiency`); load
  it in the active loader (`CalibrationDB.from_dict`, the `"memory"` path — note
  there are TWO loaders; `load_default_calib_db`→`load_calibration`→`from_dict`
  is the live one).

### 3.2 Soundness invariant (must hold)
For **every** validation kernel: `T_bound ≤ T_measured`, AND compute-bound kernels
(matmul, chunk_kda) keep a **compute** binding (cube/vector), not memory. The
contention BW must be the *peak achievable at that occupancy* (so no kernel beats
it) — calibrate from M-1/M-2 maxima, never from a sub-peak kernel.

### 3.3 Keep what already landed
The 2026-06-23 work is in place and correct: `packet_bytes` emission
(`HIVMAnalysis.cpp`, dynamic shape+stride via arg-bindings + `affine.min/max`
resolver), extractor `packet_bytes`, `bound_combiner._compute_gap2` packet-aware
lookup, `pkt_efficiency` interpolation in `lookup_bw`, and the vector-rate
soundness fix. Only the **bandwidth scale/occupancy** piece is missing.

---

## 4. Validation (the absolute gate)

1. **Soundness regression**: full `tests/perfbound/` green; `T_bound ≤ T_measured`
   on the n≥5 validation set; matmul/chunk_kda stay compute-bound. Update goldens
   to the corrected (occupancy-aware) bounds — expect tighter, still sound.
2. **Gap-2 absolute counterfactual** (US-SB-006): the amort kernel
   (`test/amort_sweep_bench.py`) at a **held-out** small packet (e.g. CHUNK=128 →
   packet 512 B) vs coalesced (CHUNK=4096). With M-1/M-2/M-3 in place:
   - `T_bound(coalesced) ≈ measured` (the optimal-copy anchor), and
   - `bound_delta = T_bound(seed) − T_bound(coalesced)` vs
     `measured_delta = t(seed) − t(coalesced)` with
     `quantification_error < 0.20`.
   (On the reverted sub-peak calibration this already reached **11.8%** for
   CHUNK=128 — the occupancy-correct calibration should match or improve it while
   keeping compute kernels compute-bound.)
3. Commit evidence under `.omc/research/hw_runs/` + a CI fixture test; flip
   US-SB-006 `passes:true` in `.omc/prd.json`.

---

## 5. Risks

| Risk | Mitigation |
|------|-----------|
| Contention curve is workload-dependent (copy ≠ matmul access) | Calibrate from the *fastest* observed per-occupancy BW (peak); treat as an upper bound on achievable → bound stays sound, possibly loose |
| η-accuracy margin (interpolated η ~6% off → bound slightly over measured) | Hold out packet points; if a bound exceeds measured, widen η toward the optimistic (higher-BW) neighbor — sound direction |
| Two calibration loaders drift (`calib_loader.py` vs `constants.py`) | Wire `contention_bw` into the live `from_dict` path; add a test asserting both loaders agree |
| Single-core kernels regress | `contention_bw(1)` returns the single-core peak (M-1) — unchanged for non-saturating kernels |

---

## 6. Out of scope / follow-ups
- US-SB-008 (compiler-headroom on hardware): closed as **documented-miss** — the
  bound-moving relaxations (Gap-1/Gap-3) are not realizable as equivalent-output
  hardware counterfactuals (empirically: Gap-3 barrier invisible to
  compiler_headroom; Gap-1 scalar-chain fix is pessimal; Gap-1 type-force breaks
  output equivalence). Independent of this plan.
- A scalar-instruction-issue term for the bound (chunk_kda's real bottleneck) —
  separate work; see `[[project-chunk-kda-headroom-floor]]`.
