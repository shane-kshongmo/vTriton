# Implementation & Experiment Plan + Paper Outline
# Two-Tier Analytical Performance Bound Model for Triton Kernels on Ascend 910B3

---

# PART A — IMPLEMENTATION PLAN

## A.0 System Overview

The system takes a Triton kernel + problem shape and emits an analytical
performance upper bound with a five-way attribution (grid + four per-core
gaps), plus two bounds (HIVM-reachable, DSL-reachable). No search, no loop.

```
   Triton kernel (.py) + shape
            │
   ┌────────┴─────────┐
   ▼                  ▼
 DSL extractor    HIVM extractor          Calibration DB
 (Tier 1 input)   (Tier 2 input)          (hardware constants,
   │                  │                     measured ONCE)
   ▼                  ▼                          │
 Grid model       Component model  ◄─────────────┘
 (occupancy,      (I_c per comp,
  balance,         mandatory vs
  redundancy)      avoidable serial)
   │                  │
   └────────┬─────────┘
            ▼
     Bound combiner  →  T_bound, binding tier/component,
                         5-way attribution, two-limit gap
```

Five modules, built in dependency order: Calibration → DSL extractor →
HIVM extractor → the two analytical models → the combiner. A sixth module
(validation harness) is built alongside but is not part of the model.

**Stage wiring contract.** A.0 defines the interfaces and dataflow; A.1
populates the measured `CalibrationDB`. The promoted A.1 database is a required
input to A.4, and the same in-memory database must continue through A.5 gap
quantification, A.6 profile-utilization analysis, and A.7 two-limit
recomputation. A report is incomplete if it computes a bound with measured
constants but silently falls back to defaults for attribution or profiling.
Every report must record the calibration path, version, hardware target, and
measurement-quality summary.

A.0 therefore contributes the calibration schema, benchmark protocol,
provenance requirements, and consumer interfaces to A.1. It does not provide
hardware-rate values itself; only A.1 measurements may populate those fields.

## A.1 Module 1 — Calibration Database (Weeks 1–2)

The only place measurement enters the model. Produces a versioned JSON of
hardware constants for one specific 910B3.

**Constants to measure** (microbenchmarks, each in AscendC or hand-HIVM):

| Constant | Benchmark | Priority |
|----------|-----------|----------|
| `P_cube[prec]` for FP16, INT8 (**sustained**) | large square GEMM per precision, steady-state | P0 |
| `P_vector[op]` for add/mul/exp/tanh/gelu/rsqrt (**sustained**) | in-UB element-wise sweep | P0 |
| `P_scalar` | i32 cmp+branch loop | P1 |
| `BW[gm→l1], BW[gm→ub], BW[l1→l0], BW[ub→gm], BW[l0c→gm]` (**sustained**) | aligned DMA sweep, steady-state | P0 |
| `BW_hbm_sustained` (all 20 cores, measured under load) | all 20 cores reading simultaneously | P0 |
| `η_alignment(stride)` | aligned vs 32B-vs-unaligned DMA | P1 |
| `η_amortization(size)` | transfer-size sweep 32B→512KB | P1 |
| `mandatory_handoff_cost` (L0C→GM + GM→UB minimum) | minimal MatMul→Vector | P0 |
| `L2_residency_bytes` (for OPTIONAL redundancy term only) | reuse-distance probe | P2 (deferred) |

**All rates are sustained, directly-measured throughput — never datasheet
peaks.** Recent accelerator-modeling work (microbenchmark-driven models
reaching ~1% MAE where naive roofline exceeds 95% error) shows datasheet
peaks and idealized reuse are the dominant source of model error. The bound's
soundness depends on `I_c` being an *achievable* rate; a too-high peak makes
the time-bound unsound (it could fall below measured). Sustained rates, being
≤ peak, keep the bound conservative.

**Deliverable**: `calib_910b3_vX.json`, plus the benchmark suite so it can be
re-run on other cards. Each constant carries a measured value and a
confidence interval (run each benchmark ≥30× for variance).

**Acceptance**: every P0 constant measured with <5% run-to-run variance;
`mandatory_handoff_cost` separable from K-scaling (vary K, confirm the
intercept is the handoff).

## A.2 Module 2 — DSL Extractor (Tier 1 input) (Weeks 2–3)

Parses the Triton kernel and shape to produce the grid partition.

**Inputs**: the `@triton.jit` function, the launch grid expression, the
problem dimensions, block-size constants.

**Outputs** (the Tier-1 quantities):
- `G` = number of program instances (evaluate the grid lambda on the shape)
- `tile_assignment[p]` = the index range each program id touches (from the
  `tl.program_id` → offset arithmetic; symbolic-execute the address
  computation, do not run the kernel)
- `occupancy = min(G, n_cores)/n_cores`
- `work[p]` per program instance (flops + bytes from its tile)
- `load_balance = mean(work)/max(work)`
- `redundancy(grid)` = total operand bytes read across instances ÷ minimal
  bytes, using `L2_residency_bytes` to decide which re-reads are free

**Method**: parse Triton AST (or TTIR — `tt.get_program_id`, `tt.load`
pointer arithmetic) to recover the affine map from program id to tile.
Most Triton kernels use a small set of grid idioms (1D row-block, 2D tile,
persistent/grouped) — implement these as templates first, general affine
recovery second.

**Affine tiling is necessary but NOT sufficient.** The extractor must also
record the *hardware constraints* that bound which tilings are legal at all:
register/buffer pressure, L0A/L0B/L0C/L1/UB capacity, and integer/tile
divisibility. These are first-class limits (recent Triton-on-NPU work treats
them as such), not afterthoughts — a grid/tile the affine map permits but
that overflows UB or a buffer is not reachable, and must not enter any bound.

**Acceptance**: on 10 reference kernels, the recovered `tile_assignment`
matches a hand-derived map; `occupancy` and `load_balance` match a manual
calculation; tilings that violate a buffer capacity are correctly rejected
as illegal.

## A.3 Module 3 — HIVM Extractor (Tier 2 input) (Weeks 3–5)

Reads the HIVM of one core's program (the busiest, per Tier 1) and extracts
per-component quantities. Uses the open BiShengIR / MLIR Python bindings.

**Outputs** (the Tier-2 quantities, per component):
- `O_prec[component]` — op/byte counts per precision/transfer-type (walk the
  HIVM op list, classify each op to its component and precision)
- `transfer_size[mte]`, `transfer_alignment[mte]` (for Gap 2)
- realized `unit_assignment[op]` (for Gap 1: compare to semantic eligibility)
- `repeat`/`mask`/SIMD-lane params per compute op (for Gap 4)
- handoff list with producer/consumer components (for serialization split)

**Method**: instrument BiShengIR to dump per-op metadata, OR walk the
dumped `.npuir.mlir` with MLIR Python bindings. Build a classifier
`op → (component, precision)` from the HIVM op definitions (`HIVMOps.td`).

**Also build here**: the **semantic eligibility oracle** — from TTIR/Linalg,
the set of units each op *could* run on (the predicates: matmul+FP16/INT8 →
Cube; element-wise/reduction → Vector; type-incompatible → Scalar). Gap 1 is
the diff between this and the realized HIVM assignment.

**Acceptance**: `Σ O_prec` reconciles with the kernel's analytic flop/byte
count within 2%; the eligibility oracle correctly flags a deliberately
seeded i32-compare Scalar fallback.

## A.4 Module 4 — The Two Analytical Models (Weeks 5–7)

Pure functions; no I/O, no compilation.

**Grid model** (Tier 1): consumes Module 2 output + calibration →
`T_grid_floor`, `busiest_core_id`.

**Component model** (Tier 2): consumes Module 3 output + calibration →
`I_c` per component (Eq. 4 harmonic mean), `T_core_floor = max_c(O_c/I_c)`,
and the **serialization split**: for each handoff, classify mandatory
(consumer needs producer's data AND cross-component) vs avoidable; sum the
mandatory minimum costs into `T_serial_irreducible`.

**Acceptance**: on a hand-computed kernel, every intermediate (`I_c`,
`T_core_floor`, `T_serial_irreducible`) matches a spreadsheet to 3 sig figs.

## A.5 Module 5 — Bound Combiner & Attribution (Week 7)

```
    T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
    binding = argmax(grid_floor vs each component) + which tier
    gaps = {gap1, gap2, gap3, gap4} computed from Module 3 deltas
    T_bound_HIVM vs T_bound_DSL (see A.7)
```

**Deliverable**: a report per kernel — bound, binding tier/component, the
five-way attribution (grid + four gaps), the two-limit gap, and the single
recommended action (fix grid / fix DSL types / merge transfers / etc).

## A.6 Module 6 — Validation Harness (Weeks 6–9, parallel; NOT in the model)

The only place compilation + execution happen. Validates the model; the
model never calls this.

**For each validation kernel**: compile via bishengir, run with output
verification against a reference (correctness first), profile with msprof,
and check: **does T_bound ≥ T_measured?** (a bound must hold) and **is the
binding component the model predicted?** Record bound tightness
`T_measured / T_bound`.

**Counterfactual capability**: hand-edit HIVM (e.g. raise `repeat`, insert
ping-pong), recompile through the open compiler, verify correctness, measure
the delta — used to confirm a gap's *quantified* value matches the measured
improvement. This validates the attribution, separately from validating the
bound.

## A.7 The Two-Limit Computation (Week 8)

```
T_bound_HIVM = bound assuming the best legal per-core HIVM + best grid
               (relax avoidable gaps to zero in the model — analytically,
                not by editing)
T_bound_DSL  = bound over HIVM that bishengir actually emits from Triton
               (use the realized HIVM's structural constraints)
gap_attribution = (T_bound_DSL − T_bound_HIVM) attributes headroom to the
                  compiler vs (T_measured − T_bound_DSL) to the kernel author
```

`T_measured − T_bound_DSL` is an **author residual**, not automatically an
attainable optimization estimate. Until a correctness-verified counterfactual
is measured, report any mechanism-specific headroom only as a diagnostic range
or upper bound with explicit confidence and method. Do not present the full
residual as a predicted speedup.

## A.8 Timeline

```
Wk 1–2   Module 1 (calibration) + benchmark suite
Wk 2–3   Module 2 (DSL/grid extractor)
Wk 3–5   Module 3 (HIVM extractor + eligibility oracle)
Wk 5–7   Module 4 (grid + component analytical models)
Wk 7     Module 5 (combiner + attribution)
Wk 6–9   Module 6 (validation harness, parallel)
Wk 8     Two-limit computation
Wk 9–12  Experiments (Part B), iterate calibration where bound is loose
```

---

# PART B — EXPERIMENT PLAN

Experiments are organized to validate, in order: (1) the bound holds,
(2) the bound is tight, (3) the attribution is correct, (4) the two-tier
structure earns its complexity, (5) the model guides real optimization.

## B.1 Kernel Suite

```
Group I  — Compute-bound, regular:    GEMM (square, tall, wide), Conv2D
Group II — Memory-bound, regular:     element-wise chains, LayerNorm, RMSNorm
Group III— Mixed Cube+Vector (Gap 3): MatMul+GELU, MatMul+LayerNorm,
                                       fused attention (FlashAttention-style)
Group IV — Ragged / grid-stressing:   variable-len attention, MoE dispatch,
                                       batch=1 decode (occupancy stress)
Group V  — Gap-seeded (controlled):   kernels with a deliberate i32 compare
                                       (Gap 1), tiny transfers (Gap 2),
                                       missing ping-pong (Gap 3), repeat=1
                                       (Gap 4) — for attribution ground truth
```

Target ≥ 30 kernels total; Groups I–III from real LLM workloads (the kernels
Inductor actually fuses), IV from attention/MoE, V hand-constructed.

## B.2 Experiment 1 — Soundness (the bound holds)

**Claim**: T_bound ≤ T_measured for every kernel — the model is a lower
bound on time (upper bound on performance), and must never overstate
achievable speed.
**Method**: compute T_bound; compile+run+measure; check the inequality.
**Metric**: fraction of kernels where the bound holds (target: 100%; any
case where T_bound > T_measured is a model bug — the bound claimed a speed
the hardware beat, i.e. an over-optimistic `I_c` or a missed mandatory
serialization that should have raised the floor).
**Failure response**: a violated bound localizes to either a too-high `I_c`
(recalibrate to a lower sustained rate) or a missed mandatory handoff (move
it from "avoidable" into T_serial_irreducible).

## B.3 Experiment 2 — Tightness (the bound is useful)

**Claim**: the bound is close enough to guide decisions.
**Method**: for kernels known to be well-optimized (vendor library ops,
hand-tuned), measure `T_measured / T_bound`.
**Metric**: tightness distribution. A well-optimized kernel should approach
the bound (ratio → 1); the paper's Depthwise at 93.5% MTE-GM is the template
— at the bound, ratio ≈ 1.07. Target: median tightness < 1.20 on optimized
kernels.
**Interpretation**: loose bounds on *optimized* kernels mean the model
misses an achievable mechanism (the bound is too optimistic); loose bounds on
*unoptimized* kernels are correct (that's the headroom the model exists to
find).

## B.4 Experiment 3 — Attribution Correctness (Group V)

**Claim**: the five-way attribution correctly identifies and quantifies the
binding cause.
**Method**: on gap-seeded kernels where the dominant gap is known by
construction, check (a) the model names the right gap, (b) the *quantified*
gap value matches the measured improvement when the seed is removed via HIVM
counterfactual.
**Metric**: gap-identification accuracy (target > 90%); quantification error
`|predicted_gap − measured_gap_removal| / measured` (target < 20%).
**This is the core scientific result** — it validates that the gaps are real,
separable, and correctly sized.

## B.5 Experiment 4 — Two-Tier Ablation (does the grid tier earn its keep)

**Claim**: a single-tier (per-core only) model mis-bounds grid-limited
kernels.
**Method**: on Group IV, compare the full two-tier bound against a
per-core-only bound (the naive paper port). Show the per-core model reports
high utilization while the chip is underused, and that the two-tier model
correctly identifies the grid as binding.
**Metric**: for occupancy/balance-limited kernels, the per-core model's
error vs the two-tier model's error against measured time. Expected: the
single-tier model is optimistic by the occupancy/balance factor (e.g. 1.25×
for the 16/20-core example) and points at the wrong optimization.
**This justifies the central architectural choice of the model.**

## B.5b Experiment 4b — Baselines (is two-tier more predictive, not just more complex)

**Claim**: the two-tier model is both more *predictive* and more
*explanatory* than simpler analytical models — not merely more complicated.
**Baselines** (both calibrated on the same sustained constants, for fairness):
- **Sustained-peak roofline**: single `max(compute_time, memory_time)` using
  measured sustained rates (no components, no grid, no serialization).
- **ECM-style staged model**: per-memory-level contributions with
  overlap/no-overlap phases (the established simple analytical baseline),
  single-core, no grid tier.
**Method**: run all three (sustained roofline, ECM-style, two-tier) on the
full suite; compare prediction error against measured, and compare whether
each model points at the correct binding cause.
**Metric**: MAE of each model's predicted time vs measured; binding-cause
identification accuracy. Expected: sustained roofline is sound but loose and
mute on *why*; ECM-style improves on memory-bound kernels but misses the grid
and the Cube↔Vector split; two-tier is tightest AND uniquely explanatory.
**This is the "earns its complexity" experiment** — without it, a reviewer
reasonably asks whether the extra structure pays off.

**Claim**: acting on the model's single recommended action improves
performance, and improves it by roughly the predicted amount.
**Method**: take unoptimized kernels, apply only the model's top
recommendation (fix grid / dtype / merge transfers / repeat param / add
ping-pong), re-measure.
**Metric**: fraction of recommendations that improve performance (target
> 80%); correlation between predicted gap size and realized speedup.
**Comparison**: against autotuning (Triton autotune) — show the model
reaches comparable per-kernel performance with zero search, and additionally
*explains* the bottleneck, which autotuning cannot.

## B.7 Experiment 6 — Two-Limit Validity (compiler vs author attribution)

**Claim**: the HIVM-reachable vs DSL-reachable gap correctly attributes
headroom to the compiler.
**Method**: for kernels where T_bound_DSL > T_bound_HIVM, hand-author the
HIVM the compiler won't emit, compile via the open bishengir, verify, and
measure — confirming the HIVM-reachable bound is attainable and the DSL path
genuinely leaves that headroom.
**Scope guard (critical)**: this comparison is claimed ONLY on kernels that
(a) compile cleanly on both the hand-authored and compiler-emitted paths,
and (b) are verified *semantically equivalent* (same output on a reference
input, bitwise or within numerical tolerance). A faster hand-HIVM that
computes a different result proves nothing. Kernels failing either condition
are excluded from the two-limit claim and reported separately.
**Hardware-limit honesty**: the extractor must treat register/buffer
pressure, L0/L1/UB capacity, and integer/tile divisibility as first-class
constraints — a hand-authored HIVM that violates them is not "reachable,"
so T_bound_HIVM must be computed subject to these limits, not under an
affine-tiling-only idealization.
**Metric**: agreement between predicted compiler-left headroom and the
measured gap between hand-authored HIVM and best DSL-compiled kernel, over
the equivalence-verified subset only.

## B.8 Experiment 7 — Generality

**Claim**: the model transfers across kernels, shapes, and (with
recalibration) Ascend variants.
**Method**: hold the model fixed, sweep shapes per kernel; recalibrate
constants for a second Ascend card (910B4 if available) and re-validate
soundness + tightness.
**Metric**: tightness stability across shapes; soundness preserved after
recalibration.

## B.9 Experiment Matrix

```
Exp  Validates              Kernel groups   Key metric            Target
───  ─────────────────────  ──────────────  ────────────────────  ──────
 1   bound holds (T_bound≤   all             % bound ≤ measured    100%
     T_measured, conservat.)
 2   bound tight            optimized ops   median tightness      <1.20
 3   attribution correct    V (seeded)      id acc / quant err    >90% / <20%
 4   two-tier necessary     IV (ragged)     single vs two-tier    grid-bind found
 4b  beats simple baselines all             MAE & cause-id vs     two-tier best
                                            roofline + ECM-style   on both
 5   guidance works         unoptimized     % improve / corr      >80%
 6   two-limit valid        compiler-gapped agreement             qualitative+quant
 7   generality+L2 stability all + 910B4    tightness; redundancy preserved;
                                            term stability         enable iff stable
```

---

# PART C — PAPER OUTLINE

**Working title**: *A Two-Tier Analytical Performance Bound for Triton
Kernels on Heterogeneous NPUs*

**Venue targets**: ASPLOS / MICRO / CGO / PACT (architecture + compilers);
the ASPLOS'25 Ascend paper is the natural related-work anchor.

## Abstract
Triton lowers to heterogeneous NPUs (Ascend Cube/Vector/Scalar + MTE) through
a multi-stage compiler, leaving developers unable to tell how much
performance a kernel is leaving on the table or why. We present an analytical
*conservative upper bound on performance* (lower bound on time) — no search,
no profiling-in-the-loop — that, given a Triton kernel and shape, computes
the fastest the kernel could run and attributes the gap to five named causes
across two tiers: an inter-core *grid* tier and an intra-core *component*
tier. The bound is provably conservative: it assumes maximal legal overlap
and adds only provably-mandatory serialization, and is calibrated entirely
from *sustained, measured* hardware rates (never datasheet peaks). We show
the bound is sound (holds on 100% of kernels) and tight on optimized kernels,
attribution is accurate on seeded kernels, the grid tier is necessary
(single-tier models mis-bound ragged kernels), and the model is both more
predictive and more explanatory than sustained-roofline and ECM-style
baselines. Acting on the model's single recommendation improves performance
comparably to autotuning while additionally explaining the bottleneck. On a
semantic-equivalence-verified subset we separate the hardware-reachable bound
from the compiler-reachable one, attributing headroom to kernel rewrite vs
compiler limitation.

## 1. Introduction
- Triton on NPUs: portability promise vs the opacity of a deep lowering stack.
- The developer's question is not "is it fast" but "how close to the limit,
  and what's stopping it" — current tools (roofline, autotune, profilers)
  answer neither analytically.
- Naively porting the single-core component roofline misses the grid and
  needs profiling data unavailable before tuning.
- Contributions: (i) a two-tier analytical bound; (ii) a complete five-way
  attribution grounded in the U=E×R decomposition plus a grid axis; (iii) the
  mandatory/avoidable serialization split that makes a bound (not an
  estimate) possible; (iv) a two-limit (HIVM vs DSL) result attributing
  headroom to compiler vs author; (v) evaluation on 30+ kernels.

## 2. Background & Motivation
- Ascend Da Vinci: components, MTE queues, the AIC:AIV split, GM/L2-only
  Cube↔Vector communication.
- Triton → TTIR → Linalg → HIVM (BiShengIR) lowering; where each decision is
  made.
- The component-based roofline (Zhou et al., ASPLOS'25) and its limits for
  the Triton setting: retrospective, single-core, measurement-bound.
- Motivating example: a kernel at 95% per-core utilization with 20% of the
  chip idle — the gap the single-core view cannot see.

## 3. The Two-Tier Bound Model
- 3.0 Direction & conservatism theorem: a lower bound on time; max-overlap
  assumption + mandatory-only serialization ⇒ provably ≤ any real schedule;
  composition by max of two independent lower bounds; calibrated from
  sustained (not peak) rates so `I_c` is achievable.
- 3.1 Tier 1 — Grid: occupancy, load balance, induced traffic; computed from
  the DSL program-id map. (redundancy term default = 1; §3.7.)
- 3.2 Tier 2 — Component: per-component ideal performance (harmonic mean),
  the per-core floor.
- 3.3 Mandatory vs avoidable serialization: why a *bound* needs the split,
  and why the split errs toward "avoidable" to stay conservative.
- 3.4 The four per-core gaps on the U=E×R grid + the grid as the fifth axis.
- 3.5 Composition: T_bound and the five-way attribution.
- 3.6 Two limits: HIVM-reachable vs DSL-reachable, subject to register/
  buffer/capacity/divisibility constraints (not affine-tiling-only).
- 3.7 The redundancy(grid) term: second-order, default-off, enabled only if
  L2 residency proves stable (Exp 7).

## 4. Computing the Bound (System)
- DSL extractor (grid), HIVM extractor (components + eligibility oracle),
  the analytical models, the combiner.
- Calibration: the one-time hardware constants; what is measured vs derived.
- Explicitly: no loop, no search; measurement only calibrates and validates.

## 5. Evaluation
- 5.1 Setup: 910B3, kernel suite (Groups I–V), sustained-rate calibration.
- 5.2 Soundness (Exp 1): the bound is a conservative time lower bound,
  T_bound ≤ T_measured on 100% of kernels; and tightness (Exp 2).
- 5.3 Attribution accuracy on seeded kernels (Exp 3).
- 5.4 Two-tier ablation: the grid tier's necessity (Exp 4).
- 5.5 Baselines (Exp 4b): vs sustained-peak roofline and an ECM-style staged
  model — more predictive (MAE) AND more explanatory (cause-id), not just
  more complex.
- 5.6 Optimization guidance vs autotuning (Exp 5).
- 5.7 Two-limit validity on the semantic-equivalence-verified subset (Exp 6).
- 5.8 Generality across shapes and a second card; L2/redundancy stability
  gate (Exp 7).

## 6. Discussion
- What the bound cannot do: it bounds the *given* computation; algorithmic
  redesign (the bottom-up rewrite axis) lowers the bound itself and is out of
  scope.
- Ragged shapes: the bound is over the worst-case partition; tighter
  distributional bounds are future work.
- The redundancy(grid) term's dependence on an L2-residency model — the one
  analytical assumption not fully measured.

## 7. Related Work
- Component/hierarchical/naive roofline; the ASPLOS'25 Ascend paper (the
  single-core, retrospective anchor we generalize).
- Microbenchmark-driven analytical models: recent GPU work reaching ~1% MAE
  (B200/MI300A) where naive roofline exceeds 95% error — evidence that
  sustained-rate calibration, not peaks, is what makes analytic models
  accurate; ECM as the established staged-analytical exemplar.
- Decoupled-architecture Ascend studies (e.g. W4A16 GEMM) showing the binding
  cost is extra GM traffic from moving data between Vector and Cube, not the
  compute — exactly the behavior our Gap 3 / mandatory-serialization term and
  MTE-GM component capture and a coarse roofline misses.
- Analytical kernel models: tritonBLAS, NeuSight, SynPerf.
- Accelerator mappers/cost models: Timeloop, MAESTRO, CoSA, ZigZag, Stream.
- Autotuning (Ansor, AutoTVM) — contrast: search (lower bound on optimum) vs
  analytical upper bound on performance.
- Triton-on-NPU work treating register pressure, cache sizing, and tile
  divisibility as first-class limits — motivating our capacity-constrained
  reachability in the two-limit analysis.

## 8. Conclusion
An analytical, two-tier, attribution-complete upper bound turns "how fast
could this Triton kernel be, and why isn't it" into a computed answer.

## Appendices
- A: full calibration constant list + benchmark methodology.
- B: the grid affine-recovery for common Triton idioms.
- C: per-kernel bound vs measured table (all 30+).
- D: the eligibility predicates and the HIVM op→component classifier.

---

# PART D — RISKS & MITIGATIONS

```
Risk                                        Mitigation
──────────────────────────────────────────  ──────────────────────────────
Bound violated (T_bound > T_measured;        Exp 1 catches it; sustained (not
model unsound)                               peak) rates keep I_c achievable;
                                            localize to I_c (recalibrate) or
                                            a missed mandatory handoff

Bound too loose to guide                    Exp 2 on optimized kernels; loose
                                            → a missing achievable mechanism;
                                            add it analytically (not a fudge)

Grid affine map not recoverable for         Template the common idioms first;
exotic kernels                              fall back to symbolic execution;
                                            report coverage honestly

redundancy(grid) / L2 model wrong           DEFAULT OFF (redundancy=1, conser-
                                            vative). Enable only after Exp 7
                                            shows L2_residency stable; report
                                            as flagged second-order estimate

Gap 2 vs Gap 4 conflated (both high-R       Key off component TYPE (MTE→Gap2,
low-E)                                       compute→Gap4) in the extractor

Two-limit claim on non-equivalent kernels   Restrict to clean-compile +
                                            output-verified subset; compute
                                            T_bound_HIVM under register/buffer/
                                            capacity/divisibility constraints

Calibration not portable across cards       Versioned calib DB; Exp 7
                                            recalibrates + revalidates
```
