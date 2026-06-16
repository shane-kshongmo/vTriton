# Analytical Performance Upper-Bound Model for Triton Kernels on Ascend 910B3

**What this is**: A model that computes the *upper bound* of performance an
operator can reach — analytically, without searching, compiling, or
iterating. Given a Triton kernel, it states the best latency achievable and
attributes the distance from that bound to specific, named causes.

**What this is not**: An optimizer. There is no compile-run-measure loop. A
loop reports the best point it happened to reach (a lower bound on the
optimum); this model reasons about what is achievable without constructing
it (an upper bound). Those are different epistemic objects. Measurement
appears only twice — to calibrate hardware constants once, and to validate
the finished model on a few kernels — never inside the model.

---

## 0. The Two-Tier Structure

The single most important correction over a naive component-roofline port:
a Triton kernel's performance is bounded at **two levels**, and for the
large kernels that matter, the upper level usually dominates.

```
            ┌──────────────────────────────────────────────────┐
   TIER 1   │  GRID TIER  (inter-core, from the DSL grid)      │
            │  How tl.program_id partitions the tensor across   │
            │  the 20 AI Cores. Sets how much work the BUSIEST  │
            │  core gets, how many cores are idle, and how much │
            │  cross-core traffic the partition induces.        │
            └────────────────────┬─────────────────────────────┘
                                 │ selects the busiest core
                                 ▼
            ┌──────────────────────────────────────────────────┐
   TIER 2   │  COMPONENT TIER  (intra-core, the paper's model) │
            │  For the busiest core: Cube / Vector / Scalar /   │
            │  MTE-GM / MTE-L1 / MTE-UB, each a serial queue    │
            │  running in parallel with the others. U = E × R.  │
            └──────────────────────────────────────────────────┘

   T_bound = T_grid_floor  ⊕  per-core component floor of the busiest core
```

Tier 1 is computed from the **DSL** (`tl.program_id`, block sizes, problem
dims) — the grid is a DSL-level decision that lowering flattens away, so it
is most visible in the Triton source. Tier 2 is computed from the **HIVM**
of one core's program — the realized tiling, unit assignment, transfers, and
instruction parameters. The model reads structural quantities off each level
once; it does not iterate.

---

## 1. Tier 1 — The Grid Floor (Inter-Core Bound)

The grid partitions the problem into `G` program instances mapped onto
`n_cores` AI Cores (20 Cube-capable; 40 Vector lanes via the 1 AIC : 2 AIV
pairing). Three independent effects, none visible to the per-core model:

### 1.1 Occupancy

If the grid launches fewer instances than cores, some cores are idle for the
whole kernel. A core at 95% internal utilization wastes half the chip if the
grid filled only 10 of 20 cores.

```
    occupancy = min(G, n_cores) / n_cores
```

For Vector-only kernels the relevant denominator is 40 (AIV count); for
Cube-bearing kernels it is 20 (AIC count, each dragging its 2 AIV).

### 1.2 Load Balance

Even at full occupancy, the kernel ends when the *slowest* core ends. The
bound is set by `max_core(work)`, not the mean. The grid's mapping of
problem dimensions to program ids determines the spread:

```
    work_busiest = max over program instances p of {
        work(tile assigned to p by the grid)
    }
    load_balance = mean_core(work) / max_core(work)   ∈ (0, 1]
```

Examples of the spread the grid creates:
- 4096 rows over 20 cores → 205 vs 204 rows: balance ≈ 0.995 (benign)
- 33 attention heads over 20 cores → 2 vs 1 head: balance = 0.5 (a 2× cap,
  dominating any intra-core gap)

This is NOT a minor correction factor. For ragged-shape kernels (attention
with variable sequence lengths, mixture-of-experts, any dimension not a
clean multiple of the grid) load balance is frequently *the* binding
constraint, above all four per-core gaps.

### 1.3 Grid-Induced Traffic

How the grid tiles also fixes cross-core data reuse, and therefore total HBM
traffic — which feeds the MTE-GM ceiling that dominates large models (the
GM→UB bottleneck the paper found in PanGu-α). A grid giving each core a
row-block lets cores share a reused operand through L2; a different split
forces redundant GM reads, inflating `bytes_in`:

```
    bytes_in_effective = bytes_in_minimal × redundancy(grid)
    where redundancy ≥ 1, determined by which operands are re-read
    across program instances that don't share an L2 residency window
```

So the grid does not merely distribute fixed work — it *changes* the work,
specifically the memory component most often binding.

### 1.4 The Grid Floor

```
    T_grid_floor = (work_busiest / I_binding_of_busiest_core)
                   × (1 / occupancy effects already in work_busiest)

    equivalently, the chip-level lower bound on time is:
      T_grid_floor = T_total_work / (n_cores × occupancy × load_balance × I_binding)
                     with bytes_in scaled by redundancy(grid) in I_binding
```

All four grid quantities — occupancy, load_balance, redundancy, and which
core is busiest — are computed directly from the DSL grid mapping and the
problem dimensions, before any lowering.

---

## 2. Tier 2 — The Per-Core Component Floor (Intra-Core Bound)

Applied to the **busiest core** identified by Tier 1. This is the paper's
component-based roofline, used unchanged because authoring/reading HIVM puts
us in exactly the paper's position (a known, fixed per-core program).

### 2.1 Components and Ideal Performance

```
    Compute:  Cube, Vector, Scalar
    MTE:      MTE-GM (GM→{L1,L0A/B,UB}), MTE-L1 (L1→L0A/B), MTE-UB (UB→{OUT,L1})

    Ideal performance of a component (paper Eq. 4 — weighted harmonic mean):

                    Σ_prec O_prec
        I_comp = ─────────────────────
                  Σ_prec (O_prec / P_prec)

    where O_prec is the operation/byte count of each precision/transfer-type
    within the component, and P_prec its peak rate. Slower precisions weigh
    more — they limit throughput more.
```

For the *bound*, the per-core floor is the binding component's time:

```
    T_core_floor = max over components c of (O_c / I_c)
```

This assumes the components overlap perfectly (each runs in parallel with
the others). That assumption is what makes it an *upper bound on
performance* (lower bound on time): no real schedule overlaps better than
fully. The distance between this floor and reality is the four gaps.

### 2.2 Irreducible Inter-Component Serialization

Perfect overlap is not always *possible*, even in principle. A matmul whose
result feeds a softmax *must* hand the data Cube→Vector; that handoff is
intrinsic to the computation, not an artifact of a bad schedule. The bound
must therefore add the *irreducible* serialization — the part no schedule
can hide:

```
    T_serial_irreducible = Σ over mandatory handoffs h of {
        min possible cost of h
    }
    where a handoff is mandatory iff the consuming op cannot begin until the
    producing op's data exists, AND producer/consumer are on different
    components that exchange only through memory (Cube↔Vector via GM/L2 on
    the 910B split architecture).
```

The distinction is essential for a *bound*: avoidable serialization (bad
sync, missing ping-pong) is a gap (Section 3), but mandatory serialization
is part of the floor itself. Only the irreducible part enters T_bound.

### 2.3 The Per-Core Bound

```
    T_core_bound = max_c(O_c / I_c) + T_serial_irreducible
```

---

## 3. The Four Gaps — Distance From the Bound, Organized by U = E × R

The bound (Tiers 1+2) is what's *achievable*. A given realized kernel falls
short by losses that map exactly onto the paper's decomposition
`U = E × R`. Crossing the axis (R = idle when it should work; E = active but
below peak) with the component type yields four — and only four — software
gaps:

```
                  │ R-axis loss                 │ E-axis loss
                  │ (idle when should work)     │ (active but below peak)
  ────────────────┼─────────────────────────────┼──────────────────────────
  Placement       │ Gap 1: wrong-unit / Scalar  │
                  │ fallback                     │
  ────────────────┼─────────────────────────────┼──────────────────────────
  MTE component   │ Gap 3: avoidable inter-unit │ Gap 2: coalescing /
                  │ serialization                │ transfer efficiency
  ────────────────┼─────────────────────────────┼──────────────────────────
  Compute comp.   │ (covered by Gap 1 & Gap 3)  │ Gap 4: intra-unit
                  │                             │ execution efficiency
```

The grid is a *fifth*, separate axis — it lives in Tier 1, above all four
per-core gaps, and is the spatial-partitioning dimension the single-core
paper never had to consider.

### Gap 1 — Wrong-Unit Placement (R-axis, placement)

An op that should run on Vector (high P) lands on Scalar (low P) — e.g.
i32/i64 compare. R_scalar high, R_vector low, utilization collapses.
Detect: semantic eligibility (TTIR/Linalg) vs realized assignment (HIVM).
Paper's fix: CT (Computation Transformation).

### Gap 2 — Coalescing (E-axis, MTE)

MTE active but below peak bandwidth from misaligned or small-granularity
transfers. `E_MTE = bytes / (T_MTE × BW_peak)`.
Paper's fixes: ITG (increase granularity), MRT (hoist redundant transfers),
TT (higher-bandwidth path).

### Gap 3 — Avoidable Inter-Unit Serialization (R-axis, MTE)

Pipeline stalls from *avoidable* handoffs — spatial dependency between
MTE-GM and MTE-UB, missing double-buffering, excessive sync. (The
*unavoidable* part of the same handoff is in T_serial_irreducible, not
here.) Paper's fixes: RSD (separate buffers), PP (ping-pong), RUS (remove
unnecessary sync), AIS-reorder.

### Gap 4 — Intra-Unit Execution Efficiency (E-axis, compute)

Right op, right unit, adequate parallelism (R high) — but the compute unit
runs below its ideal (E_compute < 1) from instruction-level inefficiency.
The paper's AvgPool: Vector at R=83.98% but U=13.54% because `repeat`=1
forced 98 loops; one AIP fix → 4.31×. The E-axis mirror of Gap 2.
Paper's fixes: AIP (adjust repeat/mask), AIS-dispatch.

### Not a Gap — Component Bound

When a component's utilization approaches its ideal (the paper's Depthwise
at 93.5% MTE-GM), there is no gap to close — the kernel has hit that
component's hardware ceiling. Mitigations (OP fusion, EA, LC) do not close a
distance to the bound; they *lower the bound itself* by changing the
computation (less traffic, fewer FLOPs, cheaper precision). That is
algorithmic redesign, a separate axis from gap-closing.

---

## 4. The Complete Bound, Stated

### 4.0 Direction and the Conservatism Theorem

The model computes a **lower bound on time** = **upper bound on
performance**. Evaluation checks `T_bound ≤ T_measured` (the bound never
overstates achievable speed). Equivalently the bound is the fastest the
kernel could possibly run; any real execution is at least this slow.

**Why the expression is guaranteed conservative.** The bound assumes the
*most favorable legal schedule*: every pair of components that *can* run in
parallel *does*, perfectly overlapped, and only handoffs that are
*provably mandatory* serialize. Formally, for the busiest core:

```
    T_core_bound = max_c (O_c / I_c)        ← assumes ALL components overlap
                   + T_serial_irreducible    ← adds ONLY provably-mandatory
                                               serialization

  This is a lower bound on the core's time because:
    (a) max_c(O_c/I_c) ≤ any real schedule's time, since no schedule can
        finish a component faster than its own ideal rate, and the real
        time is ≥ the slowest component even under perfect overlap;
    (b) T_serial_irreducible adds only handoffs whose consumer provably
        cannot start before the producer's data exists AND that cross a
        component boundary requiring a memory round-trip (Cube↔Vector via
        GM/L2). Omitting any avoidable serialization keeps the bound below
        all real schedules. Including a non-mandatory handoff would risk
        OVER-stating time (breaking the bound), so the split must err
        toward classifying handoffs as avoidable.
```

The two tiers then compose by a **max**, not a sum, because the grid floor
and the per-core floor are two independent lower bounds on the *same*
wall-clock time — the kernel cannot finish before the busiest core finishes
its component pipeline (Tier 2), and cannot finish before the chip processes
the busiest core's share of work (Tier 1). The true time is ≥ both, so
`T_bound = max(Tier1, Tier2)` is the tightest provable lower bound from the
two arguments. The `+ T_serial_irreducible` attaches to the Tier-2 term
because mandatory serialization is intra-core.

**Stages: which overlap, which serialize.**
```
  Within a core (Tier 2):
    Cube ∥ Vector ∥ Scalar ∥ MTE-GM ∥ MTE-L1 ∥ MTE-UB   — assumed parallel
    mandatory producer→consumer handoffs across components — serialize
  Across cores (Tier 1):
    all cores run in parallel; the kernel ends with the slowest (max work)
```

### 4.1 The Expression

```
    T_bound = max(
        T_grid_floor,                          ← Tier 1: busiest core's share / chip
        T_core_floor + T_serial_irreducible    ← Tier 2: component floor + intra-core serialization
    )

    with work_busiest_core set by occupancy × load_balance (× redundancy,
    see §4.3) from Tier 1, and I_c the per-core ideal performances (Tier 2).

    NOTE (A.5 soundness fix): T_serial_irreducible attaches to the Tier-2
    term inside the max, not outside it. The additive form max(a,b)+c ≥
    max(a, b+c) for c≥0, which can overstate a lower bound and risk
    T_bound > T_measured (violating the §4.0 conservatism theorem).
    The max(a, b+c) form is the tightest provable lower bound.
```

The bound is computed, stated, done. No search.

### 4.2 Attribution of the Realized Gap

A realized kernel's measured time satisfies:

```
    T_measured = T_bound
                 × Gap1_ratio          (R-axis, placement)
                 × Gap2_ratio          (E-axis, MTE)
                 × Gap4_ratio          (E-axis, compute)
                 × (1 + Gap3_fraction) (R-axis, avoidable serialization)
                 × (1 / grid_efficiency_realized)   if the realized grid is
                                                     worse than the optimal
                                                     partition the bound assumes
```

Each factor is attributable and, in principle, independently checkable.

The raw residual `T_measured - T_bound_DSL` is not by itself an attainable
headroom estimate. It includes all unmodeled execution effects and can only be
promoted to a point estimate after a correctness-verified counterfactual
demonstrates that the attributed mechanism is removable. Before that evidence
exists, reports must label it as an author residual and expose only a
confidence-qualified diagnostic interval or upper bound.

### 4.3 The redundancy(grid) Term Is Second-Order and Optional

`redundancy(grid)` — the inflation of HBM traffic from lost cross-core reuse
— depends on `L2_residency_bytes`, an idealized-reuse assumption that recent
accelerator-modeling work shows is often wrong (datasheet peaks and assumed
reuse mislead; sustained, directly-measured throughput is more reliable).

Therefore the **default model sets redundancy = 1** (no assumed cross-core
reuse benefit; each core's traffic counted independently against sustained
measured bandwidth). This keeps the bound conservative (counting *more*
traffic can only *raise* the memory time, i.e. make the time lower-bound
looser, never unsound). The redundancy < 1 refinement is enabled only after
Experiment 7 demonstrates `L2_residency_bytes` is stable across shapes; until
then it is reported as a separate, clearly-flagged second-order estimate, not
part of the headline bound.

---

## 5. What Each Level Is Read From

```
Quantity                        Source              When
──────────────────────────────  ──────────────────  ─────────────────
Grid: occupancy, load balance,  Triton DSL          before lowering
  redundancy, busiest core       (tl.program_id,
                                  block sizes, dims)

Per-core O_prec, transfer        HIVM of one core    after lowering
  sizes, unit assignment,        (or hand-authored
  instruction params             HIVM)

Mandatory vs avoidable           TTIR/Linalg DAG     before lowering
  handoffs (for T_serial and     (data dependencies
  Gap 3 split)                    + producer/consumer
                                   unit)

P_prec peaks, BW peaks,          microbenchmarks     ONCE (calibration)
  alignment/amortization curves  on the 910B3

Bound validation (does T_bound   compile + profile   ONCE per kernel,
  actually upper-bound measured)  a few kernels        as a check — NOT
                                                       part of the model
```

The open `bishengir-compile` is used for the validation row only: compile a
handful of kernels (including, if desired, hand-edited HIVM counterfactuals
to test whether the compiler's lowering is itself leaving headroom) and
confirm the model's bound holds above the measured time. This validates the
model; it is not a step the model performs.

---

## 6. The Two Limits the Model Distinguishes

Because HIVM can be authored beyond what any Triton DSL would compile to,
the model reports two distinct bounds whose difference is itself a result:

```
    T_bound_HIVM  = best achievable by any legal per-core HIVM + best grid
                    → the true hardware/architectural limit

    T_bound_DSL   = best achievable by HIVM that bishengir can actually
                    emit from some Triton kernel
                    → the limit a Triton developer can reach

    T_bound_DSL − T_bound_HIVM  =  performance the compiler's lowering
                                    leaves inaccessible to Triton users
```

This tells the developer whether to rewrite the kernel (gap is above
T_bound_DSL) or file a compiler issue (gap is between the two bounds). The
single-source paper cannot produce this attribution; it follows directly
from treating HIVM as an authorable level while keeping the DSL grid as the
Tier-1 input.

---

## 7. How a Bound Is Computed for One Kernel (Worked Shape)

Fused `MatMul(M,N,K) → Softmax(axis=N)`, FP16, on 910B3:

```
TIER 1 — Grid (from the DSL):
  grid = (ceil(M / BLOCK_M),)              ← one program per row-block
  G = ceil(M / BLOCK_M)
  occupancy   = min(G, 20) / 20
  load_balance= depends on M mod (20·BLOCK_M); for M=2048,BLOCK_M=128 → G=16
                → occupancy = 16/20 = 0.80   ← 4 cores idle! grid too small
  redundancy  = 1.0 (each row-block reads its own A-rows; B reused via L2)
  busiest core work = (2·BLOCK_M·N·K) flops + softmax(BLOCK_M·N) vector ops

TIER 2 — Component floor (from HIVM of one core):
  Cube:    O_cube  = 2·BLOCK_M·N·K            → T = O_cube / I_cube
  Vector:  O_vec   = softmax ops on BLOCK_M·N → T = O_vec  / I_vec
  MTE-GM:  bytes A,B,out                       → T = bytes / I_mte_gm
  T_core_floor = max of the above
  T_serial_irreducible = the mandatory MatMul→Softmax Cube→Vector handoff
                         (one drain L0C→GM + load GM→UB, minimum cost)

BOUND:
  T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
  Here occupancy=0.80 likely makes T_grid_floor bind → the FIRST
  optimization is not any per-core gap, it is the GRID: raise G to ≥20 by
  shrinking BLOCK_M, recovering the 4 idle cores. The model surfaces this
  immediately because Tier 1 sits above Tier 2.
```

The example shows the payoff of the two-tier structure: for this kernel the
binding constraint is the grid (20% of the chip idle), which a single-core
component model would never reveal — it would happily report 95% per-core
utilization while a fifth of the hardware sits unused.

---

## 8. Summary of the Evolution

```
Naive port:        paper's single-core roofline, applied as-is
                   → misses the grid entirely; treats load imbalance as a
                     correction haircut; needs profiling data it won't have
                     for prediction

This model:        TWO TIERS —
                   Tier 1 (grid, from DSL): occupancy, load balance,
                     induced traffic; usually binding for large kernels
                   Tier 2 (component, from HIVM): the paper's U=E×R on the
                     busiest core, with mandatory serialization in the floor
                     and avoidable serialization as Gap 3
                   FOUR per-core gaps on the U=E×R grid + the grid as a
                     fifth, higher axis
                   PURELY ANALYTICAL — bound is computed, not searched;
                     measurement only calibrates constants and validates
                   TWO bounds (HIVM-reachable, DSL-reachable) whose gap
                     attributes headroom to kernel-rewrite vs compiler-fix
```
