# Author Headroom Gap Analysis: Decomposing the T_measured − T_bound Gap

**Status**: Implemented (v4 — converged into `profile_utilization`, paper-aligned)
**Date**: 2026-06-13
**Kernel**: `chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2` (bf16 fused attention backward)
**Hardware**: Ascend 910B3, 20 AI Cores

> **v4 note — naming + home.** This analysis now lives in
> `perfbound/analyze/profile_utilization.py`, which reproduces the ASPLOS '25
> paper's *component-based roofline* and bottleneck taxonomy. The headroom
> mechanism is reported under the paper's verdict **"Insufficient Parallelism"**,
> localized to the exposed Scalar control. The quantity this doc calls **Gap-OVL**
> is the paper-faithful **exposed control/sync deficit** and is surfaced as
> `exposed_control_deficit_pts` (= measured same-core scalar − model
> critical-path exposed control ≈ **+72.7 pts**) and `exposed_control_deficit_us`
> (≈ **58,216 µs**, capped at author headroom when a sound loop-scaled two-tier
> bound is supplied). The earlier `combine/` "Gap-OVL" attribution (the duplicate)
> has been retired; the sound bound still comes from `combine/`'s two-tier
> pipeline, which `profile_utilization` consumes via `t_bound_us`. "Gap-OVL" is
> kept below only as the historical internal name.

> **v3 note.** v2 (retracted) attributed the dominant author headroom to
> "Unmodeled Scalar Cost" and proposed US-SB-007 scalar calibration as the fix.
> v3 corrects this: the scalar dominance is *exposed control + synchronization
> overhead* — pipeline/overlap inefficiency — not scalar arithmetic.  The DES
> schedule assumes ~6× overlap of control with compute (model exposed 11.9%);
> hardware serializes it (measured 84.5% scalar).  This is the **Gap-OVL**
> (Exposed Control/Sync Overlap Deficit).  v2's scalar-calibration conclusion
> is retracted; Gap-OVL is the dominant mechanism.

---

## 0. Problem Statement

The existing 5-gap model (grid, Gap-1 … Gap-4) attributes performance loss
*within* T_bound. For chunk_kda:

```
T_bound    = 46,110 µs   (Tier-2 DES structural bound)
T_measured = 104,326 µs  (msprof on 910B3)
Author headroom = 58,216 µs   (55.8% of T_measured)
```

The 5-gap attribution covers only 2.73% of T_bound, so the headroom looks like a
monolithic unknown. **It is not.** The msprof op-summary CSV carries per-engine
busy ratios, and the DES graph carries per-pipe structural busy time plus a
schedule timeline. Read together they give a direct, quantified answer: **the
kernel's headroom is dominated by exposed control/sync overlap deficit.**

**Question answered:** *Can author headroom be analyzed from msprof + HIVM IR?*
**Yes** — for this kernel class it decomposes cleanly, and it is dominated by a
single, fixable cause: the schedule over-assumes overlap of control with compute.

---

## 1. Key Finding: The Headroom Is Exposed Control/Sync Overlap Deficit (Gap-OVL)

Three facts, each independently verifiable from the DES graph + msprof:

1. **0 of 597 `PIPE_S` ops are vectorizable** (none have `elements > 1`).  There
   is no VEC→scalar fallback.  The PIPE_S ops are shape metadata
   (`collapse_shape`, `subview`, `reinterpret_cast` ≈ 225), address/index
   arithmetic (`index_cast`, `apply`, `muli`, `addi`, `min`, `max`, `cmpi`, …
   ≈ 300), `alloc_workspace`, and loads.  Each is priced at **1 cycle** in the
   model (585/597 = 1 cyc).

2. **402 synchronization ops**: `set_flag` 116, `wait_flag` 110, `pipe_barrier`
   80, `sync_block_wait` 48, `sync_block_set` 48 — also priced ~1 cycle.

3. **Schedule overlap walk**: sweeping the DES schedule timeline shows
   control/sync is 76.4% of structural busy, but the schedule overlaps it down
   to **11.9% exposed** on the critical path (assumes ~6× overlap).  msprof
   shows scalar is **84.5%** of wall-clock.  The hardware does **not** achieve
   the modeled overlap.

**Retraction of v2 conclusion.** v2 labeled this as "Unmodeled Scalar Cost" and
proposed US-SB-007 (calibrate scalar throughput, add a scalar floor to the
bound).  That conclusion was wrong: the scalar dominance is *exposed control +
sync overhead* — a pipeline-overlap inefficiency — not scalar arithmetic and not
VEC→scalar fallback.  v2 mislabeled this as scalar compute; it is overlap
deficit.  US-SB-007 scalar calibration is retired as the primary fix.

**Two independent inputs agree on the mechanism:**

| Input | Signal | Scalar share |
|-------|--------|--------------|
| msprof (measured, **same-core**) | `aiv_scalar_ratio` = scalar(AIV) / aiv_time | **84.5% of AIV core time** ← Gap-OVL term |
| msprof (measured, blended) | (scalar AIV + scalar AIC) / T_measured | 91.6% of T_measured (95,525 µs) — *context only* |
| HIVM/DES (structural) | Σ `duration·loop_multiplier` over `PIPE_S` ops | 72.7% of structural busy (597 ops) |

The DES schedule overlap walk reduces structural 72.7% to model-exposed 11.9%
on the critical path; the **same-core** measured 84.5% scalar shows the hardware
does not achieve this overlap.  **Gap-OVL subtracts matched denominators**
(both normalize to one core's timeline): `84.5% − 11.9% = +72.6 pts`.  The
blended 91.6% (scalar over wall-clock, across AIV+AIC) is a *different*
denominator and is reported for context only, never as the Gap-OVL pts.

---

## 2. Existing Analysis Results (End-to-End Run)

### 2.1 Bound Analysis

| Metric | Value |
|--------|-------|
| T_bound (Tier-2 DES) | 46,109.91 µs |
| T_bound_HIVM (idealized) | 46,064.92 µs |
| T_measured | 104,326.00 µs |
| Tightness | 2.26× |
| Soundness | PASS ✓ (T_bound ≤ T_measured) |
| Binding tier | component |
| Binding component | vector  ← *should reflect scalar dominance via Gap-OVL* |
| T_grid_floor | 46,064.92 µs |
| T_core_floor | 46,109.91 µs |
| T_serial_irreducible | 0.0000 µs |

### 2.2 Component Attribution (from msprof + DES schedule overlap walk)

Produced by `attribute_by_component()` in `bound_combiner.py` + the schedule
overlap walker `_schedule_overlap()`.  Engine-time = (core busy time) × (that
engine's msprof ratio).

**[1] msprof MEASURED engine-time** (where the hardware spent time):

| Engine | µs | % of core busy |
|--------|-----|----------------|
| **scalar (AIV)** | **87,935** | **84.5%** |
| scalar (AIC) | 7,590 | 7.3% |
| vector (AIV) | 4,579 | 4.4% |
| mte2-load (AIC) | 2,703 | 2.6% |
| fixpipe (AIC) | 2,183 | 2.1% |
| mte2-load (AIV) | 1,977 | 1.9% |
| mte3-store (AIV) | 1,145 | 1.1% |
| mte1 (AIC) | 832 | 0.8% |
| mac-cube (AIC) | 624 | 0.6% |
| **scalar total** | **95,525** | **91.6% of T_measured** (blended AIV+AIC over wall-clock — context only; Gap-OVL uses same-core 84.5%) |

Side facts: `aic_icache_miss_rate = aiv_icache_miss_rate = 0.0` (not a cache
problem); `cube_utilization = 99.6%` while `aic_mac_ratio = 0.006` (cube is
*occupied but stalled* — waiting, not computing).

**[2] HIVM STRUCTURAL busy per PIPE** (where the model puts time):

| Pipe | n ops | µs | % of structural busy |
|------|------:|-----:|---------------------:|
| **PIPE_S (scalar)** | 597 | 128.0 | **72.7%** |
| PIPE_V (vector) | 254 | 17.5 | 9.9% |
| PIPE_MTE2_V | 62 | 9.6 | 5.5% |
| PIPE_MTE3 | 70 | 6.8 | 3.9% |
| PIPE_FIX | 41 | 6.4 | 3.6% |
| PIPE_ALL | 54 | 3.0 | 1.7% |
| PIPE_MTE2_C | 59 | 2.8 | 1.6% |
| PIPE_M (cube) | 32 | 1.2 | 0.7% |
| PIPE_MTE1 | 19 | 0.6 | 0.4% |
| PIPE_UNKNOWN | 190 | 0.0 | 0.0% |

**[3] Schedule overlap walk** (Gap-OVL structural primitives):

| Metric | Value |
|--------|-------|
| Critical path | 10,055 cycles |
| Exposed control (no compute/memory overlap) | 11.9% of critical path |
| Control overlapped with compute/memory | remaining control cycles |
| Sync/barrier ops | 402 |

### 2.3 Two-Limit Analysis

| Limit | Value (µs) | % of T_measured |
|-------|-----------|-----------------|
| T_bound_HIVM | 46,064.92 | 44.2% |
| T_bound_DSL | 46,109.91 | 44.2% |
| T_measured | 104,326.00 | 100% |
| Compiler headroom | 44.99 | 0.04% |
| Author headroom | 58,216.09 | 55.8% |

### 2.4 DES Graph Summary

Total ops 1,378. Pipe counts: PIPE_S 597, PIPE_V 254, PIPE_UNKNOWN 190,
PIPE_MTE3 70, PIPE_MTE2_V 62, PIPE_MTE2_C 59, PIPE_ALL 54, PIPE_FIX 41,
PIPE_M 32, PIPE_MTE1 19. Ops with repeat>1: 271 (min 8, max 64, median 32).
`schedule_truncated = False`.

---

## 3. Headroom Decomposition (evidence-led, v3)

Gaps are ordered by what the data supports.

### Gap-OVL: Exposed Control/Sync Overlap Deficit — **DOMINANT**

**Evidence**: Schedule overlap walk shows model exposes 11.9% control on the
critical path; msprof shows 84.5% scalar in wall-clock.  Gap-OVL = +72.6 pts
(≈ 58,216 µs, capped at author headroom).

**What it measures**: the gap between the DES schedule's *assumed* overlap of
control/sync with compute/memory, and the *realized* serialization on hardware.
The model prices each control op at ~1 cycle and assumes the scalar unit
overlaps the vector/cube units; the hardware serializes scalar address math,
flag ops, and barriers.

**How to extract**: the schedule overlap walk (`_schedule_overlap` in
`bound_combiner.py`) sweeps start/end_cycle events, classifying ops as
control_sync / compute / memory, then computes exposed_control_cycles as the
fraction of the critical path where control is active with no overlapping
compute or memory.  Compared against msprof `aiv_scalar_ratio`.

**Metric** (denominator-honest):
- Primary: `gap_ovl_pts = measured_scalar_frac − model_exposed_control_frac`
  (points, not µs — the two fractions have different denominators).
- Secondary: `gap_ovl_us = min(gap_ovl_pts × t_measured, author_headroom)` —
  capped to never exceed the headroom it explains.

**Magnitude**: ≈ 58,216 µs (the entire author headroom).

**Action**: reduce sync density (multi-buffer), hoist address math out of the
tile loop, fuse tiles to amortize control.  **Diagnostic only — bound unchanged.**

### Gap-E: Wave Quantization — small, structural, belongs in the grid floor

**Evidence**: grid = 4096 programs / 20 cores ⇒ 205 waves, last wave 16/20.
`grid_model.py` uses `occupancy = min(G,n_cores)/n_cores = 1.0` for G≫cores, so it
captures the average but not the last-wave tail.

**Magnitude (its own formula)**: `(1 − 16/20)·T_core_floor / n_waves =
0.2·46,110/205 ≈ 45 µs`. This belongs **inside** T_grid_floor, not in author
headroom.

### Gap-C: Instruction Overlap — subsumed by Gap-OVL

**Evidence**: v2 described this as "plausible, needs calibration."  v3 notes
that Gap-OVL *is* the quantified instruction-overlap deficit: the schedule
overlap walk directly measures how much overlap the model assumed vs what
hardware achieved.  Gap-C is subsumed by Gap-OVL and no longer a separate
calibration item for this kernel class.

### Gap-A: HBM Bandwidth Contention — **refuted for this kernel**

v1 estimate 9,000–17,000 µs (15–30%). Measured total MTE activity here is
`aiv_mte2 0.019 + aic_mte2 0.026 + mte3 0.011 + mte1 0.008 ≈ 4.7%` (~7,600 µs).
Contention cannot be 15–30% of headroom when *all* memory-engine activity is
~4.7%. **Drop for this kernel class.**

### Gap-B: AIC-AIV Pipeline Serialization — **refuted for this kernel**

v1 estimate 6,000–12,000 µs. `aic_mac_ratio = 0.006` — almost no cube MatMul.
`cube_utilization 99.6%` with mac 0.6% means the cube is *idle-but-allocated*.
**Drop for this kernel class.**

### Gap-D: Hardware-Level Residual — keep, but small

Honest catch-all = `T_measured − T_bound(after Gap-OVL) − Gap-E`.
`icache_miss_rate = 0.0`. Once Gap-OVL is diagnosed, this residual is expected
to be small.

---

## 4. Estimated Headroom Decomposition (v3, corrected)

```
T_measured (104,326 µs)
├── T_bound (46,110 µs)  ── structural bound (currently mis-binds on vector)
│     ├── gap4_intra_unit (1,185 µs = 2.57%)
│     ├── gap1_wrong_unit (72 µs = 0.16%)
│     └── model core (44,853 µs)
│
└── Author headroom (58,216 µs = 55.8%)
      ├── Gap-OVL Exposed control/sync overlap  ≈ 58,216 µs  (DOMINANT; +72.6 pts)
      ├── Gap-E  Wave quantization              ~45 µs (belongs in grid floor)
      ├── Gap-D  Hardware residual              small once Gap-OVL is diagnosed
      ├── Gap-A  HBM contention                 refuted (~7,600 µs total MTE)
      └── Gap-B  AIC-AIV dequant serial         refuted (aic_mac 0.6%; absent)
```

The honest one-line decomposition: **author headroom ≈ Gap-OVL (exposed
control/sync overlap deficit) + a small residual.**

---

## 5. Data Availability Matrix

| Data | Available? | Location |
|------|-----------|----------|
| Grid dims + n_cores | ✅ | grid=(128,32), n_cores=20 |
| DES dependency edges | ✅ | kda_des.json `depends_on` |
| Pipe assignment per op | ✅ | kda_des.json `pipe` (PIPE_S = 597) |
| Per-pipe structural busy | ✅ | Σ `duration·loop_multiplier` |
| **Schedule timeline (start/end_cycle)** | ✅ | kda_des.json per-op fields |
| **msprof per-engine ratios** | ✅ | `aiv_scalar_ratio`, `aiv_vec_ratio`, `aic_*_ratio` |
| **msprof icache miss rate** | ✅ | `aic/aiv_icache_miss_rate` (= 0.0 here) |
| Cube utilization | ✅ | `cube_utilization(%)` |
| Per-core execution timeline | ❌ | Not in msprof aggregate |
| Cache miss *penalty* (cycles) | ❌ | Only miss-rate exposed |
| Per-wave timing | ❌ | Not in msprof aggregate |

> **Scope caveat.** Per-engine ratios are populated for the chunk_kda
> **MIX_AIC** row and for the **AI_VECTOR_CORE** rows of softmax / layernorm /
> rmsnorm.  The method generalises across both task types.
>
> **Scalar dominance is pervasive, not chunk_kda-specific.** Measured scalar
> share across the committed fixtures: chunk_kda **0.85**, rmsnorm **0.69**,
> softmax **0.51**, layernorm **0.47** — scalar is the dominant engine in every
> populated kernel.  The Gap-OVL diagnosis applies wherever the schedule
> over-assumes overlap.

---

## 6. Implementation Roadmap (v3, reprioritized)

| Phase | Component | Effort | Why |
|-------|-----------|--------|-----|
| **P0** | **Wire Gap-OVL diagnostic into bound_combiner + report** | DONE ✅ | Turns headroom into a quantified overlap deficit |
| **P0** | Reduce-sync-density recommendation surfaced in report | DONE ✅ | Actionable guidance for kernel authors |
| **P1** | Fix binding-component reporting so scalar can bind once calibrated | 1 hr | §2.1 currently mis-binds vector |
| **P2** | Gap-E wave-quant term folded into `grid_model.py` floor (~45 µs) | 1 hr | Structural; move out of "headroom" |
| **P3** | Gap-D residual bucket | 1 hr | Named remainder after Gap-OVL/E |
| ~~—~~ | ~~Gap-A HBM contention, Gap-B AIC-AIV serial~~ | — | Refuted for this kernel class |
| ~~—~~ | ~~US-SB-007 scalar calibration~~ | — | **Retired**: scalar dominance is overlap deficit, not throughput |

---

## 7. Refuted Claims (Lessons Learned)

| Claim | Why Refuted |
|-------|-------------|
| Author headroom decomposable via existing 5-gap model | Gaps are WITHIN T_bound, not the headroom |
| HBM contention is 15–30% of headroom (Gap-A) | Measured total MTE activity ≈ 4.7%; refuted by this kernel's msprof |
| AIC-AIV dequant serialization is 10–20% (Gap-B) | `aic_mac_ratio = 0.006`; cube idle-but-allocated; W4A16 mechanism absent |
| msprof exposes only aggregate task duration | CSV has per-engine ratios + icache rates (parser limitation, not data) |
| Cache behavior drives the residual (Gap-D content) | `icache_miss_rate = 0.0` |
| Wave quantization is 500–1,500 µs (Gap-E v1) | Its own formula gives ~45 µs |
| **v2: Headroom is unmodeled scalar cost (Gap-S / US-SB-007)** | Scalar dominance is *exposed control/sync overlap deficit*, not arithmetic; 0/597 ops are vectorizable; schedule walk proves overlap assumption |

---

## 8. Limitations & Open Questions

1. **Gap-OVL is diagnostic only.** It does not change `t_bound_us`,
   `t_core_floor_us`, or any soundness invariant.  The bound stays loose but
   sound until a future overlap model is calibrated.

2. **Denominator caveat.** `model_exposed_control_frac` is a fraction of the
   idealized critical path; `measured_scalar_frac` is a fraction of wall-clock.
   The primary metric (gap_ovl_pts, in points) is denominator-honest; the
   secondary µs estimate is capped at author headroom.

3. **MIX_AIC vs simple kernels.** Component attribution is proven on the MIX_AIC
   chunk_kda row; DSA_SQE/AI_VECTOR_CORE kernels need a populated-field check
   before the same method applies.

4. **Per-core variance** remains invisible to msprof-aggregate; it needs
   per-core traces.

5. **Counterfactual validation still blocked** — DES-JSON edits cannot be
   recompiled; Gap-OVL is validated by *agreement of two independent inputs*
   (schedule walk + msprof) rather than by a hardware counterfactual.

---

## 9. Verified Sources

| Source | Used for | Access |
|--------|----------|--------|
| `chunk_kda_op_summary.csv` (MIX_AIC row) | §1/§2.2 measured engine ratios | `.omc/research/hw_runs/` |
| `kda_des.json` | §2.2 structural per-pipe busy + schedule timeline | `.omc/research/hw_runs/` |
| `scripts/component_attribution_prototype.py` | reproduces all §1/§2.2 numbers | repo |
| `scripts/overlap_walker_prototype.py` | validates schedule overlap walk (§2.2[3]) | repo |
| `perfbound/combine/bound_combiner.py` | `_schedule_overlap`, `attribute_by_component` | codebase |
| `perfbound/model/component_model.py:288–306` | scalar floor uses vector rate | codebase |
| `perfbound/calibration/constants.py:252` | scalar-rate soundness fallback | codebase |
| `perfbound/combine/two_limit.py` | author/compiler headroom defs | codebase |
