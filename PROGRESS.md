# Performance Bound Model — Mainline Progress

**Project**: Two-Tier Analytical Performance Bound Model for Triton/Ascend NPU  
**Spec**: `.omc/specs/performance_bound_model.md` + `.omc/specs/implementation_and_paper_plan.md`  
**Plan**: `.omc/plans/a0_perf_bound_model.md`  
**Detail log**: `.omc/plans/a0_progress.md`

---

## Stage Map (A.0 → A.8 + Part B)

| Stage | Scope | Timeline | Status |
|-------|-------|----------|--------|
| **A.0** | Python `perfbound/` package scaffold (M1–M6 stubs + models) | Wk 1–7 | ✅ **Complete** |
| **A.1** | M1 Calibration — AscendC microbench suite on 910B3 | Wk 1–2 | ✅ **Complete** — 16 P0 constants measured + wired into model (40/40 tests) |
| **A.2** | M2 DSL Extractor — symbolic affine recovery from TTIR | Wk 2–3 | ✅ **Complete** — C++ MLIR pass + 10 reference kernels verified (63/63 tests) |
| **A.3** | M3 HIVM Extractor — C++ JSON round-trip verified | Wk 3–5 | ✅ **Complete** — DES schema fix, MTE byte aggregation, handoff tracing (108/108 tests) |
| **A.4** | M4 Two Analytical Models — units discipline + compute_bounds driver | Wk 5–7 | ✅ **Complete** — flops consumption, distinct-edge serialization, 144/144 tests |
| **A.5** | M5 Combiner — Gap 1/2/4 wired, Gap 3 from real handoffs | Wk 7 | ✅ **Complete** — combiner, 5-way attribution, two-limit, real-kernel milestone (committed `dffda87`) |
| **A.6** | M6 Validation Harness — measurement + counterfactual | Wk 6–9 | ✅ **Complete** — A.6.1 measurement (`174d0ab`) + A.6.2 counterfactual/remote runner; **live 910B3 run done** (real T_measured, evidence `.omc/research/hw_runs/`) |
| **A.7** | Two-limit computation (`T_bound_HIVM` vs `T_bound_DSL`) | Wk 8 | ✅ **Complete** — delivered within A.5; **author_headroom now populated from real msprof data** |
| **A.8** | End-to-end pipeline verified on ≥1 real kernel | Wk 9 | 🟡 Mostly done — mixed-CV pipeline green; chunk_kda **compiles+runs on CANN 9.0.0** and is measured on hardware (bishengir beta.2 crash resolved); remaining gap is offline des.json extraction for chunk_kda (needs NPU host) |
| **Part B** | Experiments, paper writing, iterate calibration | Wk 9–12 | ⛔ Not started |

---

## Current State (2026-06-10)

> A.5 → A.6.2 landed since the snapshot below. The "What's done" list covers
> A.0–A.4; see the **A.5** and **A.6** completion sections lower down for the
> combiner, attribution, two-limit, and validation harness. Full `perfbound/`
> suite: **317 passed, 3 skipped, 2 xfailed**.
>
> **Hardware milestone (2026-06-10):** the remote 910B3 is wired
> (`scripts/remote_bench.py`, ssh host `910B3`, CANN 9.0.0, conda `triton_hxl`).
> chunk_kda **compiles and runs** on the device (the bishengir
> `ConvertLinalgRToBinary` crash was a CANN 9.0.0-**beta.2** bug, fixed in the
> 9.0.0 release). It was profiled under msprof: **T_measured ≈ 104.3 ms**,
> T_bound (HBM floor) ≈ 1.39 ms → soundness **PASS** (75× tightness),
> **author_headroom ≈ 102.9 ms** — the first real, non-synthetic three-level
> measurement. Real-data parser bug fixed (MIX_AIC task type). Evidence +
> reproduce steps: `.omc/research/hw_runs/RESULTS.md`; CI guard:
> `tests/perfbound/test_chunk_kda_hw_validation.py`. The 2 remaining xfails are
> the local milestone compile tests, which now xfail only because WSL has no
> NPU device (not the compiler bug).

### What's done

- **`perfbound/` package** (28 files, pure Python, zero MLIR dependency)
  - M1: `CalibrationDB`, `CalibrationConstant` (value ± CI, source, n_runs), full JSON ser/de
  - M2: `GridInfo`, `DSLExtractor` (C++ MLIR pass + affine idiom recovery), `grid_idioms.py` (1D/2D templates), `mlir_parser.py` (subprocess wrapper), 10 reference kernels verified
  - M3: JSON loaders for both C++ emits, `OpRecord` + `HandoffRecord`, pipe→component classification, eligibility oracle, `repeat`/`mask` fields, C++ DES graph schema fix (`operations` canonical), MTE byte aggregation bugfix, handoff tracing
  - M4: `compute_bounds` driver, `BoundPieces`, units discipline (flops for FLOP/us, bytes for B/us), distinct-edge serialization, grid floor explicit total_work parameter, component model weighted-harmonic mean
  - M5: `T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible`, 5-way attribution, `_wire_gaps` connecting Gap 1/2/4 helpers, binding tier detection, text+JSON report
  - M6: Stubs with correct `NotImplementedError` (hardware-gated)
- **C++ emitter layer** (committed): `emitDESGraph()` JSON, `emitDependencyGraphJSON()`, `PipelineAnalysisPass` wiring
- **A.1 calibration** (`calib_910b3_v1.json`): 16 P0 constants, n=45 each, all CI < 2.5%
  - Cube FP16/INT8/BF16: ~5.16 TFLOPS/core
  - BW GM→UB / UB→GM: ~87 GB/s; GM→L1: ~141 GB/s; L1→L0A: ~452 GB/s
  - Vector add/mul/max/min: 14.6–16.2 GFLOPS; transcendentals: 3.3 GFLOPS
  - Mandatory handoff cost: 7621 ± 82 cycles (~4.1 µs at 1.85 GHz)
- **Tests**: 144/144 passing (26 A.1 + 18 A.2 + 47 A.3 + 53 A.4)

### What's blocked / deferred

| Item | Blocker | Priority |
|------|---------|----------|
| ~~chunk_kda compile crash~~ | ✅ **RESOLVED** — was a CANN 9.0.0-**beta.2** bug; chunk_kda compiles+runs on CANN 9.0.0 release (`.omc/research/hw_runs/`) | — |
| ~~Live 910B3 validation runs~~ | ✅ **DONE** — real T_measured + author_headroom from msprof; soundness PASS (`test_chunk_kda_hw_validation.py`) | — |
| Offline des.json for chunk_kda | Needs `tritonsim-hivm --triton-script` on an NPU host (WSL has no NPU; remote 910B3 is aarch64, the x86-64 binary can't run there) | P2 |
| Counterfactual delta on hardware | Mechanism offline-tested + remote wiring fixed; one live edit→recompile→delta run still pending | P2 |
| Gap 4 from C++ emitter `repeat`/`mask` | C++ emitter still defaults these (model reads them; emitter doesn't populate yet) | P2 |
| Scalar throughput (Vector/20 proxy) | Needs real Scalar calibration (B.4) before trusting as a lower bound | P2 |

**Closed since A.4**: Gap 3 (avoidable serialization, A.5) · Two-limit / A.7 (`two_limit.py`, A.5) · M6 measurement + counterfactual harness (A.6.1/A.6.2).

### Recently closed (A.3/A.4)

| Item | Closed by |
|------|-----------|
| C++/Python DES graph schema mismatch | A.3 |
| MTE byte-vs-element aggregation bug | A.3 |
| Missing transfer metadata (`transfer_sizes`, `transfer_alignments`) | A.3 |
| Non-deterministic C++ output paths (`pipeline_dep_graph.json`) | A.3 |
| Units discipline (flops vs elements) | A.4 |
| Grid floor bypass in bound_from_extract | A.4 |
| T_serial_irreducible flat-cost vs Σ over distinct edges | A.4 |

---

## Completed: A.2 — DSL Extractor (2026-06-08)

| Step | Status |
|------|--------|
| C++ `ExtractTTIRInfoPass` (`--extract-ttir-info`) | ✅ Done — walks AST, emits JSON (grid_axes, persistent_loops, tensor_ptr_shapes, has_dot) |
| `perfbound/extract/mlir_parser.py` | ✅ Done — subprocess wrapper, brace-counted JSON extraction |
| `dsl_extractor.py` refactor (no regex) | ✅ Done — `_extract_from_ttir()` uses `parse_ttir()`, `_persistent_kernel_info()` round-robin |
| `grid_idioms.py` Bug 1 fix | ✅ Done — `parents[2]` correct, `ascend_910b3.json` loads |
| 10 reference kernel parametrized suite | ✅ Done — K1–K10, occupancy/lb/tile_assignment/buffer_ok verified |
| Code review (7 findings) | ✅ All resolved — F1 NameError, F2 pytestmark, F3 paths, F4 C++ filter, F5 inline TTIR, F6 ifdef, F7 dead elif |

---

## Completed: A.3 — HIVM Extractor (2026-06-08)

| Step | Status |
|------|--------|
| C++ DES graph schema fix | ✅ Done — `emitDESGraph()` emits `schema_version: "a3_hivm_des_v1"` and `start_cycle`/`end_cycle` per op |
| C++ deterministic output paths | ✅ Done — `desGraphFile` and `dependencyGraphFile` options, removed hard-coded cwd writes |
| Python MTE byte aggregation | ✅ Done — `load_hivm_desgraph()` uses bytes not elements for MTE ops |
| Python handoff tracing | ✅ Done — canonical compute-to-compute tracing through MTE intermediaries |
| Python transfer metadata | ✅ Done — `transfer_sizes`, `transfer_alignments`, memory space normalization |
| Python required field validation | ✅ Done — validates id, name, pipe on load |
| 22 HIVM extractor tests | ✅ Done — DES schema, field validation, MTE byte aggregation, transfer metadata, handoffs |
| 22 eligibility oracle tests | ✅ Done — Matmul→Cube, elementwise→Vector, i32 compare→Scalar, Gap 1 analysis |
| 3 CLI integration tests (xfail) | ✅ Done — proper xfail on broken fixture |

---

## Completed: A.4 — Two Analytical Models (2026-06-09)

| Step | Status |
|------|--------|
| Component model flops consumption | ✅ Done — uses `op.flops` for FLOP/us rate, not `op.elements` |
| Grid model explicit total_work | ✅ Done — caller-supplied `total_work` parameter matching `i_binding` units |
| Serialization distinct-edge dedup | ✅ Done — sums over DISTINCT mandatory edges by component pair |
| compute_bounds driver | ✅ Done — picks (i_binding, total_work) consistently (memory-bound vs compute-bound) |
| bound_from_extract wiring | ✅ Done — routes through `compute_bounds`, no bypass |
| 53 model tests | ✅ Done — FlashAttention golden test, grid floor unit tests, serialization dedup tests, compute_bounds unit-selection |

---

## Completed: A.6 — Validation Harness (2026-06-10)

**Detail log**: `.omc/plans/a6_progress.md` · **Blocker scoping**: `.omc/plans/a6_2_blockers_scope.md`

M6 splits into measurement (A.6.1) and counterfactual (A.6.2); both software-complete.
The only remaining work is hardware/compiler-gated (a live 910B3 run + the CANN
compiler bug), tracked by xfail/spike tests.

| Part | Step | Status |
|------|------|--------|
| A.6.1 | msprof CSV parser (AiCore filter, invocation grouping, max-per-invocation, warmup discard, median) | ✅ Done (13 tests) |
| A.6.1 | Tri-state harness (`ValidationStatus`; infra errors excluded from soundness) | ✅ Done (7 tests) |
| A.6.1 | Canonical modeling output (HIVM/DSL/profile values + validity gates) | ✅ Done |
| A.6.2 | HIVM edit primitives + no-op guards + extract-reversibility check | ✅ Done (23 tests) |
| A.6.2 | Output correctness verification (numpy allclose) | ✅ Done (10 tests) |
| A.6.2 | Counterfactual orchestration (edit→compile→verify→delta, tri-state infra) | ✅ Done (19 tests) |
| A.6.2 | Fallback-kernel counterfactual (decoupled from chunk_kda compiler bug) | ✅ Done (10 tests) |
| A.6.2 | On-device Triton kernel launcher (load→run→dump `.npy`) | ✅ Done (13 tests) |
| A.6.2 | Remote 910B3 runner (sync/recompile/msprof/fetch, host config, CANN preamble) | ✅ Done (18 tests) |
| A.6 | chunk_kda compile + dump-survives spike | ⛔ xfail — bishengir `ConvertLinalgRToBinary` crash (CANN 9.0.0-beta.2) |

---

## Completed: A.5 — Bound Combiner & Attribution (2026-06-09)

| Step | Status |
|------|--------|
| `T_bound = T_launch + max(T_grid_floor, T_core_floor + T_serial_irreducible)` | ✅ Done |
| Five-way attribution (grid + gap1/2/3/4), Gap 3 from real handoff classification | ✅ Done |
| Binding tier/component detection + text/JSON report | ✅ Done |
| Two-limit (`T_bound_HIVM` vs `T_bound_DSL`) + canonical modeling output | ✅ Done — `two_limit.py`, `report.py` |
| Real-kernel milestone (mixed-CV pipeline end-to-end) | ✅ Done |

---

## Historical: A.5 milestone scope (Wk 7)

**A.5 scope**: Complete the M5 combiner module with full five-way attribution wired from real data:
- Gap 3 (avoidable serialization) computation from handoff classification
- Gap 1/2/4 helpers verified on real HIVM extracts
- End-to-end bound computation on ≥1 real kernel
- Text+JSON report generation with binding tier/component identification

| Step | Status |
|------|--------|
| C++ `ExtractTTIRInfoPass` (`--extract-ttir-info`) | ✅ Done — walks AST, emits JSON (grid_axes, persistent_loops, tensor_ptr_shapes, has_dot) |
| `perfbound/extract/mlir_parser.py` | ✅ Done — subprocess wrapper, brace-counted JSON extraction |
| `dsl_extractor.py` refactor (no regex) | ✅ Done — `_extract_from_ttir()` uses `parse_ttir()`, `_persistent_kernel_info()` round-robin |
| `grid_idioms.py` Bug 1 fix | ✅ Done — `parents[2]` correct, `ascend_910b3.json` loads |
| 10 reference kernel parametrized suite | ✅ Done — K1–K10, occupancy/lb/tile_assignment/buffer_ok verified |
| Code review (7 findings) | ✅ All resolved — F1 NameError, F2 pytestmark, F3 paths, F4 C++ filter, F5 inline TTIR, F6 ifdef, F7 dead elif |

---

## Next Milestone: A.3 — HIVM Extractor (C++ JSON round-trip)

**A.3 scope**: Verify the C++ `emitDESGraph()` + `emitDependencyGraphJSON()` JSON round-trip end-to-end on a real kernel. Python loaders exist (`hivm_extractor.py`, `OpRecord`, `HandoffRecord`) but the C++ JSON output has not been validated against the Python schema.

**Closure targets** (from `open-questions.md §A.3`):
- [ ] DES graph schema compatibility — canonicalize C++/Python contract, required per-op fields, schema version
- [ ] Deterministic artifact handling — remove hard-coded `pipeline_dep_graph.json`/trace writes; use `tmp_path` in tests
- [ ] Tier 2 metadata completeness — `O_prec`, transfer size/alignment, unit assignment, repeat/mask defaults populated from real JSON
- [ ] Semantic eligibility oracle — connect TTIR/Linalg semantics to realized HIVM unit assignment; flag Scalar fallback

**Open item — L2 cache BW model** (deferred to A.5/A.7):
- Real kernels have partial L2 reuse; pure HBM BW is conservative (sound) but loose for tight bounds
- Future: `BW_eff = h × BW_l2 + (1-h) × BW_hbm` where `h = f(working_set / L2_size)`
- Tracked in `open-questions.md §Open Item — L2 Cache Bandwidth Model`

---

## Formula Reference

```
T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible

# Tier 1: Grid floor (A.4)
T_grid_floor = total_work * redundancy / (n_cores * occupancy * load_balance * i_binding)
- Memory-bound: i_binding = BW (B/us), total_work = Σ MTE bytes
- Compute-bound: i_binding = Cube/Vector throughput (FLOP/us), total_work = Σ flops

# Tier 2: Component floor (A.4)
T_core_floor = max_c(O_c / I_c)
I_c = Σ O_prec / Σ (O_prec / P_prec)  [weighted harmonic mean over ops in component c]

# Serialization (A.4)
T_serial_irreducible = |E_mandatory| * (mandatory_handoff_cycles / clock_freq)
where E_mandatory = set of DISTINCT (producer_component, consumer_component) edges

# Five-way attribution (diagnostic, not part of bound):
  grid         = T_grid_floor / T_bound
  gap1         = wrong-unit placement (eligible set vs realized unit)
  gap2         = coalescing loss (actual pkt BW vs ideal large-pkt BW)
  gap3         = avoidable serialization (non-mandatory handoffs)
  gap4         = intra-unit exec inefficiency (repeat > 1, masked lanes)
```
