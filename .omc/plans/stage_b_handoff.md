# Stage-B Execution Handoff — vTriton Performance-Bound Model

**Audience:** an execution agent picking this up cold.
**Written:** 2026-06-11, after commit `d24986c` (A.6 hardware caveat closure).
**Spec:** `.omc/specs/performance_bound_model.md` (read it — it is the source of truth).
**Status doc:** `PROGRESS.md` · **A.6 detail:** `.omc/plans/a6_progress.md` · **HW evidence:** `.omc/research/hw_runs/RESULTS.md`

---

## TL;DR

The analytical bound model (M1–M6) is **code-complete** and now **hardware-validated for soundness on one real kernel** (chunk_kda: `T_bound ≤ T_measured`, PASS). The remote 910B3 is wired and working; the bishengir compiler blocker is **resolved** (it was a CANN 9.0.0-beta.2 bug, fixed in the 9.0.0 release on the box).

**Stage B (experiments + paper) is gated on ONE thing:** getting a **real `des.json`** (HIVM extract) for a live kernel so the *full Tier-2 bound* — not just the analytic grid floor — can be computed and validated. Everything else (multi-kernel validation, counterfactuals) follows quickly once that path works.

Your job: execute the ordered task list below. **Task 1 is the gate — do it first.**

---

## Environment & Access (read before doing anything)

### Local (this WSL box, `DESKTOP-V11OAGE`, x86-64)
- Repo: `/mnt/d/work/git/vTriton` (Windows host, run everything via WSL Ubuntu-24.04).
- Python for tests: `/home/shane/miniconda3/envs/vtriton-verify/bin/python` (has torch; **no NPU**).
- **Always run pytest with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`** — otherwise `torch_npu` auto-load crashes (no device). Example:
  ```bash
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/shane/miniconda3/envs/vtriton-verify/bin/python \
    -m pytest tests/perfbound/ -p no:cacheprovider -q
  ```
- `-p no:cacheprovider` avoids a WSL stale-`.pyc` gotcha (see Gotchas). If an `inspect.getsource` test flakes, `find . -name __pycache__ -type d -exec rm -rf {} +`.
- Local C++ binary: `build/bin/tritonsim-hivm` (x86-64 ELF). `--help` lists its flags. **Rebuild** (after C++ changes) per `CLAUDE.md`: `cd build && ninja -j$(nproc)`.
- CANN locally at `~/Ascend` (calibration only; no device).

### Remote 910B3 (the NPU box — `ssh 910B3`)
- ssh host alias `910B3` (173.147.1.2, root, via local SOCKS proxy already in `~/.ssh/config`). Just `ssh 910B3 '<cmd>'` works.
- **aarch64** — the local x86-64 `tritonsim-hivm` ELF **cannot run there**. (This is why Task 1 needs the NPUIR-dump-then-fetch route.)
- 8× 910B3 NPUs (chip 0 may show "Alarm"; chips 1–7 OK).
- CANN **9.0.0 release** at `/usr/local/Ascend/ascend-toolkit/`; `bishengir-compile` + `msprof` at `/usr/local/Ascend/cann-9.0.0/bin/`.
- triton **3.2.0** in conda env **`triton_hxl`**.
- Remote env preamble (sources CANN + activates conda):
  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate triton_hxl
  ```
- `msprof` has **no `--version`** flag — probe with `command -v msprof`.
- Config for the runner is in `~/.vtriton_remote` (`[remote] host=910B3 path=/root/vTriton`).
- Sync the repo (scoped — never push `.git`/`build`/`thirdparty`, ~24 GB):
  ```bash
  rsync -az --delete --exclude=.git --exclude=build --exclude=thirdparty \
    --exclude=.claude --exclude=.omc --exclude=__pycache__ --exclude='*.pyc' \
    -e 'ssh -o ConnectTimeout=25' ./ 910B3:/root/vTriton/
  ```
  (`scripts/remote_bench.py::sync_to_remote` already encodes these excludes.)

### The remote runner: `scripts/remote_bench.py`
Already fixed and wired this session. `run_remote_bench(remote_host, kernel_name, kernel_script=..., output_csv=..., output_npy=..., hivm_in=...)` does: sync → (optional recompile from edited HIVM) → msprof → fetch `op_summary_*.csv` (+ `kernel_output.npy`). Env-overridable: `VTRITON_REMOTE_CANN_SETENV`, `VTRITON_REMOTE_CONDA_ENV`.

---

## Safety rules (non-negotiable — from project memory)

- **Git:** never `git add -A` / `git add .` / `git clean -fd`. Stage **explicit paths** only; verify zero unexpected `D` deletions before commit. WSL git hangs on submodules — use `--ignore-submodules=all` for status/diff (with `timeout`) and `--recurse-submodules=no` for push. Never run concurrent background git mutations (`.git/index.lock` collisions). Commit footer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Workflow is trunk-based on `master`.
- **Parsing:** never regex TTIR/MLIR — use `lark` or the MLIR API.
- **Microbenchmarks:** use **CCE** language for Ascend microbench (not AscendC / hand-HIVM).
- **Don't commit** OMC runtime state (`.omc/state/`, `.omc/sessions/`, `.omc/project-memory.json`) or `.claude/`.

---

## What's done vs gated (so you don't redo it)

| Area | State |
|---|---|
| M1–M5 model + 5-way attribution + two-limit (`two_limit.py`) | ✅ done, tested |
| M6 measurement harness + msprof parser + 3-level report | ✅ done |
| M6 counterfactual mechanism (edit→recompile→verify→delta) | ✅ offline-proven (`test_counterfactual_fallback.py`); 0 live runs |
| Remote 910B3 wiring (`remote_bench.py`) | ✅ fixed + smoke-tested on hardware |
| bishengir chunk_kda compile | ✅ RESOLVED (CANN 9.0.0); kernel runs on device |
| Real `T_measured` + soundness PASS (chunk_kda) | ✅ but against the **analytic floor**, not a real des-extract bound; **n=1** |
| Real `des.json` for a live kernel | ❌ **THE GATE** — see Task 1 |
| Gap 4 (`repeat`/`mask`) real data | ✅ **done (2026-06-11)** — derived analytically (`repeat=ceil(elements/lanes)` in C++; 271/1378 ops repeat>1) since repeat/mask are absent at npuir level for all kernels and the bishengir codegen stage is unparseable (compile >600s). Per-instruction Gap-4 model; real bound 46,110µs ≤ 104,326µs (2.26×). See RESULTS.md §5. `mask` lane-fill model deferred. |
| Scalar throughput | ❌ Vector/20 proxy, uncalibrated |
| `redundancy(grid)` < 1 | ❌ hardcoded 1 (conservative); needs Experiment 7 |
| Two-limit compiler-headroom (`T_bound_DSL − T_bound_HIVM`) | ✅ **measured on seeded_serial via hand-edited TTAdapter compiler IR (2026-06-24)** — held-out calibration gives `T_bound_HIVM=1819.070us <= T_bound_DSL=1840.007us <= T_measured=3289.604us`; predicted headroom 20.937us vs measured TTAdapter-edit delta 20.860us, output-verified. Evidence: `.omc/research/hw_runs/seeded_serial/RESULTS.md` |

---

## ✅ DECISION RESOLVED (2026-06-11) — Gap-1/3 only

The interpretation below was **resolved by the spec author: keep as-is
(Gap-1/Gap-3 only)**. `two_limit.py` stays unchanged; Task 7 validates against
that definition. Original framing retained for context.

## ⚠️ DECISION (was REQUIRED) — Gap interpretation for `T_bound_HIVM`

`perfbound/combine/two_limit.py` computes `T_bound_HIVM` (the "best legal per-core
HIVM" limit) by relaxing **only Gap-1 (placement) and Gap-3 (avoidable
serialization)** to zero, and **deliberately keeps Gap-2 (coalescing) and Gap-4
(intra-unit execution) as hardware limits** (documented in the module header).
Spec §6 phrases the limit as "best achievable by any legal per-core HIVM"; §3
calls Gap-2 and Gap-4 *software* gaps (E-axis), which arguably should also be
relaxed for a true HIVM-reachable limit.

This is a modelling-semantics decision, **not** an executor decision:
- **Keep as-is (Gap-1/Gap-3 only):** `T_bound_HIVM` = "best *scheduling/placement*
  a compiler could do," treating coalescing/exec efficiency as fixed by the op
  structure. Narrower, more conservative compiler-headroom number.
- **Also relax Gap-2/Gap-4:** `T_bound_HIVM` = "best *any* hand-authored HIVM,"
  matching §3's classification of all four as software gaps. Larger headroom,
  but assumes the author can always fix coalescing/`repeat`.

**Resolve this before Task 7** — it changes what Task 7 must measure. Until
resolved, Task 7 should validate against the *current* (Gap-1/Gap-3) definition
and flag the result as interpretation-dependent.

---

## TASK 1 — Real `des.json` for a live kernel  **(THE GATE — do first)**

**Why:** §5 of the spec reads the per-core Tier-2 quantities (`O_prec`, transfer sizes, unit assignment, `repeat`/`mask`) from the **HIVM of one core**, materialized as `des.json` by `tritonsim-hivm`. Today that path is only exercised on the synthetic `test/hivm_mixed_cv_kernel.npuir.mlir` fixture. Without a real des.json, the chunk_kda bound is just the HBM grid floor — which measured **75× loose** and **mispredicts the binding component** (predicts memory, hardware is compute-core-bound). The tight bound *needs* Tier-2 from real HIVM.

**The constraint:** `tritonsim-hivm --triton-script` JIT-compiles via the installed triton wheel → needs an **NPU device**. WSL has none; the 910B3 is **aarch64** so the x86 `tritonsim-hivm` can't run there. Dead end as-is.

**The route (device-free): `tritonsim-hivm --npuir-file=<chunk_kda.npuir.mlir>`** parses a dumped NPUIR file with **no device**. So:
1. On the **remote**, dump chunk_kda's `*.npuir.mlir` during a normal triton-ascend compile.
2. **Fetch** the `.npuir.mlir` to local.
3. Run **locally**: `build/bin/tritonsim-hivm --npuir-file=<fetched> --hardware-config=configs/ascend_910b.json --des-graph-file=/tmp/kda_des.json --scheduler=des`.
4. Feed `kda_des.json` into `report_from_desgraph(...)` → real Tier-1+Tier-2 bound + two-limit, then pair with the real `T_measured` (104,326 µs) already captured.

**Step 1 is a spike — find how triton-ascend emits `.npuir.mlir`.** The triton cache after a run holds `chunk_kda*.ttir`/`.ttadapter`/`.npubin` but **not** `.npuir.mlir`. Candidate mechanisms to try on the remote (inside the preamble env, `cd /root/vTriton/test`):
- triton dump env vars: `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/root/vTriton/ttdump MLIR_ENABLE_DUMP=1` then `python chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py`, and search the dump dir / cache for `*.npuir.mlir` or a linalg-dialect `.mlir`.
- bishengir IR dump: recompile with `--bishengir-print-ir-after=hivm-inject-sync` (referenced in `test_chunk_kda_milestone.py::TestDumpBeforeCodegen`) and capture stdout/stderr MLIR.
- Inspect the triton-ascend wheel for a dump hook: `python -c "import triton; print(triton.__file__)"` then grep the backend for `npuir`/`dump`.
- Check `tritonsim-hivm --keep-dump-dir` output format (it keeps the temp Triton dump dir on a working NPU host) to learn the exact filename/layout `--npuir-file` expects.

**Acceptance criteria:**
- A `chunk_kda*.npuir.mlir` (or equivalent NPUIR the local binary accepts) is captured on the remote and fetched to `.omc/research/hw_runs/`.
- `build/bin/tritonsim-hivm --npuir-file=... --des-graph-file=/tmp/kda_des.json --hardware-config=configs/ascend_910b.json` produces a non-empty `des.json` with Cube + Vector + MTE ops (verify with `perfbound.extract.hivm_extractor.extract_hivm`).
- `report_from_desgraph(des_json=..., grid_dims=(128,32), n_cores=20, kernel_name="chunk_kda", t_measured_us=104326.0)` yields a `KernelReport` with `t_bound_us > 0`, a three-level reachability hierarchy, and a **tighter** bound than the 1,386 µs HBM floor (expect Tier-2/serialization to bind, narrowing the 75× gap).
- A FLOP reconciliation: extract FLOPs within ±25 % of `HAND_FLOPS_TOTAL = 4.724e10` (see `test_chunk_kda_milestone.py::test_compile_and_emit_des`).
- Add a CI test (extend `tests/perfbound/test_chunk_kda_hw_validation.py` or a sibling) that loads the committed `kda_des.json` and asserts the real-bound soundness (`T_bound ≤ T_measured`).
- If Step 1 proves infeasible after a genuine spike, **document why** in `.omc/research/hw_runs/RESULTS.md` and fall back to building `tritonsim-hivm` natively on the aarch64 remote (CMake/ninja) as Plan B — note that as a finding, don't silently skip.

**Key files:** `perfbound/extract/hivm_extractor.py`, `perfbound/combine/run_report.py` (`report_from_desgraph`), `tests/perfbound/test_chunk_kda_milestone.py` (the `--triton-script` cmd shape at lines ~208–223), `configs/ascend_910b.json`.

---

## TASK 2 — Multi-kernel validation set (n ≥ 5)

**Why:** §5 wants the model validated on "a few kernels"; we have **n=1**. Soundness (`T_bound ≤ T_measured`) needs a spread to be credible for the paper.

**Steps:**
- Pick ≥4 more kernels. Easy runnable candidates: `test/triton_hivm_launch_smoke.py`, a vector add (write a small `build_inputs`+`Model` harness like chunk_kda's), plus layernorm/softmax (see `test/layernorm_ascend.mlir`, `test/softmax_ascend.mlir` for shapes). Each must have a kernel-launcher-compatible entry (`main()` **or** `build_inputs()`+`Model.forward()`; see `scripts/kernel_launcher.py`).
- For each: `scripts/remote_bench.py` → msprof CSV → `parse_kernel_time_us` (real `T_measured`); get its `des.json` via Task 1's route → real bound; run `validate_from_csv` → expect PASS.
- Aggregate via `ValidationSuite` (`perfbound/validate/harness.py`): report `soundness_rate` (must be 1.0 — any `BOUND_VIOLATION` is a model bug to fix) and `median_tightness`.

**Acceptance:** a committed table (CSV/JSON under `.omc/research/hw_runs/`) of ≥5 kernels with `T_bound`, `T_measured`, status, tightness, `component_match`; `soundness_rate == 1.0`; CI test loading the fixtures.

**Gotcha:** any Triton kernel may profile as `MIX_AIC`/`MIX_AIV`/`AI_VECTOR_CORE` — already handled by `msprof_parser._is_aicore_task` (exact-match set). If a new kernel shows an unrecognized compute task-type, **add it to that frozenset** (and a test), don't loosen to substring.

---

## TASK 3 — One live counterfactual (Experiment 3)

**Why:** validates gap **attribution** (not just the bound). Target: `output_verified AND |predicted−measured|/measured < 0.20`.

**Steps:** pick a quantified gap on a real kernel (e.g. Gap 3 avoidable serialization or Gap 4 `repeat`), apply the matching `HivmEdit` (`perfbound/validate/hivm_edits.py`: `raise_repeat`/`insert_pingpong`/`merge_transfers`) to the kernel's `des.json`/HIVM, then `run_counterfactual(...)` (`perfbound/validate/counterfactual.py`) with `remote_host="910B3"`, `remote_bench_script="scripts/remote_bench.py"`, a `reference_fn` for correctness. bishengir recompile works now (`recompile_remote`). Confirm `CounterfactualResult.is_valid`.

**Acceptance:** ≥1 `CounterfactualResult` with `output_verified=True` and `quantification_error < 0.20`, evidence committed. **Depends on Task 1** (need a real HIVM to edit).

---

## TASK 4 — Make Gap 4 (`repeat`/`mask`) real  **(STILL OPEN — current code is a no-op)**

**Why:** the model reads `repeat`/`mask` but they never carry real data, so Gap 4 (intra-unit execution efficiency — the paper's AvgPool `repeat=1` → 4.31× case) is untrustworthy on real kernels.

**Status (code review 2026-06-11):** the committed C++ change added the plumbing — `extractRepeatMask` (`HIVMAnalysis.cpp`), the `repeat`/`mask` struct fields, and JSON emission — but it is a **confirmed no-op**: every one of the 1378 ops in `.omc/research/hw_runs/kda_des.json` is `repeat:1, mask:0`. The acceptance test `test_des_json_has_repeat_mask_fields` only checks that the JSON *keys exist* (the emitter writes them unconditionally), so it passes while extraction is fully broken. Two compounding root causes — split the remediation accordingly:

### Task 4a — extraction-correctness fixes (ready now, no device needed)

- In `extractRepeatMask` (`lib/AscendModel/Analysis/HIVMAnalysis.cpp:~2172`), replace `tail.getAsInteger(10, val)` with `tail.consumeInteger(10, val)`. `StringRef::getAsInteger` returns an *error* whenever any text trails the number (the normal case, e.g. `repeat = 8 : i64`), so the text fallback can **never** fire even when the token is present; `consumeInteger` parses a leading integer and tolerates the trailing `: i64} ...`.
- Anchor the `find("repeat = ")` match so it can't bind a *different* attribute's value (e.g. `loop_repeat = 4`). Require a non-identifier char before `repeat`, or read a named attribute only.
- Verify/clarify `mask` semantics: the struct comment says "0 = all lanes active" but the field is populated from `mask_count`, whose active-vs-disabled meaning is unconfirmed — pin it against a kernel that actually sets it before any consumer relies on the value.
- Replace the tautological `test_des_json_has_repeat_mask_fields` with a **value-asserting** test: add a tiny synthetic `.npuir.mlir` fixture containing an op with an explicit `repeat` attribute (and `mask_count`), run it through `emitDESGraph`, and assert the emitted JSON carries the non-default values. This makes the C++ path verifiable **without** the real device.

### Task 4b — source `repeat`/`mask` from the correct IR (THE open blocker)

- **Evidence:** `grep repeat chunk_kda_kernel_clean.npuir.mlir` → 0 hits; the only `mask` token is `set_mask_norm` (no `mask_count`); 1378/1378 ops default. The IR we parse (post-`GraphSyncSolver` `hivm.hir`) is simply **before** the stage where per-op CCE `repeat`/`mask` are materialized — so even a correct `extractRepeatMask` (Task 4a) finds nothing here.
- **Action:** spike where in the triton-ascend / bishengir lowering the per-op CCE `repeat`/`mask` first appear (later than `hivm.hir` — likely CCE/binary codegen), and either dump *that* IR for `--npuir-file` or extract the params at that stage. Requires a device/IR-stage investigation. **Depends on Task 1** (the real des.json route).
- **Acceptance:** a real kernel's `des.json` shows non-default `repeat`/`mask` on the ops that have them; `_wire_gaps` (`perfbound/combine/bound_combiner.py`) yields a non-trivial Gap 4; the value-asserting test from 4a is green.

---

## TASK 4.5 — Code-review findings (2026-06-11), adjacent risks

These surfaced alongside Task 4 and feed the same evidence pipeline — fix opportunistically.

- **`scripts/clean_npuir.py` fragility (feeds the `kda_des.json` fixture):** it filters MLIR by line-prefix allowlist and drops every `hivm.hir.copy` line. That can delete an SSA def used downstream, or a multi-line op's continuation line, silently altering the graph that becomes `kda_des.json`. Violates the project's no-regex-for-MLIR rule. Re-derive the fixture via the MLIR API / a clean dump rather than text-stripping, and treat the committed `kda_des.json` as **provisional** until Task 4b lands.
- **DES `maxIterations` guard (`HIVMAnalysis.cpp:~2681`):** on a livelock it breaks at `completedCount < numOps` with only an `llvm::errs()` line; the truncated `weightedCycles` then flows downstream as if the schedule completed. Surface an incomplete/error flag on the report so a consumer can refuse the bound instead of silently trusting a partial schedule.

---

## TASK 5 — Scalar throughput calibration (B.4)

**Why:** Scalar uses a `Vector/20` proxy; any Scalar-bound result is currently untrustworthy as a lower bound.

**Steps:** add a **CCE** microbench for Scalar throughput (mirror the Cube/Vector microbench pattern in `perfbound/calibration/`), measure on the 910B3, fit the constant via `fit_constants.py`, wire into the calibration DB (`calib_910b3_v1.json`). Note: `fit_constants.read_msprof_csv` now summarizes skipped `N/A` rows once to stderr.

**Acceptance:** a measured `P_scalar` constant (with CI) replacing the proxy; calibration tests updated.

---

## TASK 6 — Experiment 7 + `redundancy(grid)` refinement

**Why:** §4.3 keeps `redundancy = 1` (conservative, but loose for kernels with cross-core reuse) until Experiment 7 shows `L2_residency_bytes` is stable across shapes.

**Steps:** design Exp 7 (sweep shapes, measure sustained GM bandwidth vs assumed L2 reuse on the 910B3); if `L2_residency_bytes` is stable, enable the `redundancy < 1` path in the grid model (`perfbound/combine/…` grid floor) as a **clearly-flagged** refinement, never the headline bound. Lowest priority.

**Acceptance:** Exp 7 data + a decision (enable refinement or keep `redundancy=1` with evidence).

---

## TASK 7 — Validate the two-limit compiler-headroom on hardware (Spec §6)

**Status update 2026-06-24:** CLOSED via `seeded_serial` using a hand-edited
TTAdapter compiler-IR hardware realization; see
`.omc/research/hw_runs/seeded_serial/RESULTS.md` and
`tests/perfbound/test_seeded_serial_two_limit_hardware.py`. The older steps below
are retained as historical context.

**Why:** §6 is a headline result of the model — `T_bound_DSL − T_bound_HIVM` =
"performance the compiler's lowering leaves inaccessible to Triton users," which
tells a developer whether to **rewrite the kernel** (gap above `T_bound_DSL`) or
**file a compiler issue** (gap between the two limits). `two_limit.py` *computes*
both limits, but **no hardware run has confirmed the `T_bound_HIVM` is actually
reachable** — i.e. that a hand-authored HIVM beating what bishengir emits really
hits the lower limit. Without that, the compiler-headroom number is unvalidated.

**Prerequisite:** resolve the **DECISION REQUIRED** callout above (Gap-1/Gap-3
only, vs also Gap-2/Gap-4). This sets what "reachable" means. **Also depends on
Task 1** (need a real `des.json` + HIVM for a live kernel).

**Steps:**
1. For a real kernel (chunk_kda or a Task-2 kernel), compute `T_bound_HIVM`,
   `T_bound_DSL`, and `T_measured` via `compute_two_limit(...)`
   (`perfbound/combine/two_limit.py`) — the realized bishengir HIVM gives
   `T_bound_DSL`; the idealized extract gives `T_bound_HIVM`.
2. Author a **hand-optimized HIVM** that realizes the relaxations the idealized
   extract assumed (fix the Gap-1 placement / remove the avoidable Gap-3
   handoffs — and Gap-2/Gap-4 too if the decision says so). Use the
   `HivmEdit` primitives (`perfbound/validate/hivm_edits.py`) where they suffice,
   or a hand-written `.npuir.mlir`.
3. Recompile + profile it on the 910B3 (`recompile_remote` / `run_remote_bench`),
   verify output correctness (`perfbound/validate/correctness.py`), and measure
   its time `T_measured_HIVM_opt`.
4. **Check the claim:** `T_measured_HIVM_opt` should approach `T_bound_HIVM`
   (within the §6 sense), and the *realized* speedup `T_measured − T_measured_HIVM_opt`
   should match the predicted compiler-headroom `T_bound_DSL − T_bound_HIVM`
   within ~20 % (same tolerance class as the Exp-3 counterfactual). This is the
   HIVM-level analogue of Task 3, but targeting the **two-limit** rather than a
   single gap.

**Acceptance:**
- A committed result showing, for ≥1 real kernel: `T_bound_HIVM ≤ T_bound_DSL ≤
  T_measured`, plus a hand-optimized HIVM whose measured time validates the
  predicted compiler headroom within tolerance (or a documented explanation if
  it does not — a miss here is a real model finding about §6).
- The result is annotated with which interpretation (Gap-1/3 vs Gap-1/2/3/4) was
  used, per the DECISION callout.
- Evidence under `.omc/research/hw_runs/`; a CI test loading the fixtures.

**Note:** if the hand-optimized HIVM cannot be authored/compiled for the chosen
kernel, fall back to a simpler kernel where a known compiler-suboptimality exists
(e.g. a forced Scalar fallback for Gap-1, or `repeat=1` for Gap-4 à la the
paper's AvgPool), so the headroom is large and measurable.

---

## Verification reference card

```bash
# Full suite (must stay green): 317 passed, 3 skipped, 2 xfailed
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/shane/miniconda3/envs/vtriton-verify/bin/python \
  -m pytest tests/perfbound/ -p no:cacheprovider -q

# Smoke the remote env + tools
ssh 910B3 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate triton_hxl && \
  command -v bishengir-compile && command -v msprof && npu-smi info | head'

# Re-derive the real T_measured from the committed fixture
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/shane/miniconda3/envs/vtriton-verify/bin/python -c "
import sys; sys.path.insert(0,'.')
from perfbound.validate.msprof_parser import parse_kernel_time_us
print(parse_kernel_time_us('tests/perfbound/fixtures/chunk_kda_op_summary_910b3.csv',
      op_name_filter='chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2', n_warmup=0))"
```

The 2 xfails are the local milestone compile tests (`TestChunkKdaCompile`, `TestDumpBeforeCodegen`) — they xfail **only because WSL has no NPU**, not the (now-fixed) compiler bug. If you ever run them on an NPU x86 host they should XPASS.

---

## Gotchas reference card

| Symptom | Cause / fix |
|---|---|
| `torch_npu` import crash in pytest | set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| `inspect.getsource` test returns wrong function | stale `.pyc` on WSL (coarse mtime); `rm -rf` `__pycache__`, use `-p no:cacheprovider` |
| Kernel rows missing from msprof timing | task type is `MIX_AIC`/`MIX_AIV`/`AI_VECTOR_CORE` — add to `_AICORE_TASK_TYPES` frozenset |
| `msprof --version` fails | use `command -v msprof` |
| rsync to remote is huge/slow | excludes `.git`/`build`/`thirdparty` (already in `sync_to_remote`) |
| local `tritonsim-hivm` won't run on remote | it's x86; remote is aarch64 — use `--npuir-file` locally instead |
| git status/push hangs | `--ignore-submodules=all` (status/diff) / `--recurse-submodules=no` (push) + `timeout` |

## Evidence & references
- `.omc/research/hw_runs/RESULTS.md` — full hardware findings + reproduce steps.
- `.omc/research/hw_runs/chunk_kda_op_summary.csv` — raw 910B3 msprof CSV.
- `tests/perfbound/test_chunk_kda_hw_validation.py` — the real-data CI guard (analytic-floor bound).
- `tests/perfbound/test_chunk_kda_milestone.py` — hand-derived FLOP/byte counts + the `--triton-script` cmd shape.
- Project memory: `~/.claude/projects/-mnt-d-work-git-vTriton/memory/project_910b3_hardware.md`.
- Commit `d24986c` — the work this handoff builds on.
