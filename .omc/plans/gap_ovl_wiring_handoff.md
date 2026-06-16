# Wiring Guide: Gap-OVL (Exposed Control/Sync Overlap Deficit)

**Audience**: an implementing agent with no prior context on this thread.
**Type**: precise, self-contained wiring task. Read this whole file first.
**Prereq reading in repo**: `.omc/specs/author_headroom_gap_analysis.md` (v2),
`perfbound/combine/bound_combiner.py`, `perfbound/combine/report.py`,
`perfbound/combine/run_report.py`, `perfbound/validate/msprof_parser.py`.

---

## 0. One-paragraph context

For the `chunk_kda_bwd` kernel the author-headroom gap (T_measured 104,326 µs −
T_bound 46,110 µs = 58,216 µs) was first attributed to "unmodeled scalar
compute." **That conclusion was wrong and has been superseded.** Evidence (below)
shows the scalar dominance is *exposed control + synchronization overhead* —
pipeline/overlap inefficiency — not scalar arithmetic and not VEC→scalar
fallback. Your job is to wire this corrected attribution (**Gap-OVL**) into the
model as a **diagnostic** (it must NOT change the bound), and rewrite spec §3.

---

## 1. The validated finding (do not re-derive; you may re-run to confirm)

Two standalone prototypes already exist and produce the numbers below. Run them
to confirm before changing code:

```bash
python3 scripts/component_attribution_prototype.py \
  --csv .omc/research/hw_runs/chunk_kda_op_summary.csv \
  --des .omc/research/hw_runs/kda_des.json --kernel chunk_kda_bwd --t-bound 46109.91

python3 scripts/overlap_walker_prototype.py \
  --des .omc/research/hw_runs/kda_des.json \
  --csv .omc/research/hw_runs/chunk_kda_op_summary.csv --kernel chunk_kda_bwd
```

Established facts (all reproducible from those two inputs):

1. **0 of 597 `PIPE_S` ops are vectorizable** (none have `elements > 1`). There is
   no VEC→scalar fallback. The PIPE_S ops are shape metadata (`collapse_shape`,
   `subview`, `reinterpret_cast` ≈ 225), address/index arithmetic (`index_cast`,
   `apply`, `muli`, `addi`, `min`, `max`, `cmpi`, … ≈ 300), `alloc_workspace`,
   and loads. Each is priced at **1 cycle** in the model (585/597 = 1 cyc).
2. **402 synchronization ops**: `set_flag` 116, `wait_flag` 110, `pipe_barrier`
   80, `sync_block_wait` 48, `sync_block_set` 48 — also priced ~1 cycle.
3. **Schedule overlap walk**: control/sync is 76.4% of structural busy, but the
   DES schedule overlaps it down to **11.9% exposed** on the critical path
   (assumes ~6× overlap). msprof shows scalar is **84.5%** of wall-clock. The
   hardware does **not** achieve the modeled overlap.
4. Therefore: **Gap-OVL = exposed control/sync overlap deficit** is the dominant
   author-headroom mechanism. The bound (46,110 µs) is set by the *component
   throughput floor*, not the schedule, so reporting Gap-OVL changes **no** bound
   value and carries **no** soundness risk.

**Do NOT**: add a scalar throughput floor, change `compute_bounds` /
`component_model.py`, or revive "US-SB-007 scalar calibration" as the fix. Those
are explicitly retired.

---

## 2. Current code state (already committed at 6acc203)

`perfbound/combine/bound_combiner.py` already has:
- `_PIPE_TO_ENGINE` map.
- `ComponentAttribution` dataclass (structural pipe split + measured engine split
  + `mis_binding`/`note`).
- `attribute_by_component(extract, binding_component=None, measured=None)`.

Its current `note` talks about scalar being modeled at the vector rate and
"calibrate scalar (US-SB-007)". **You will replace that note logic** with the
Gap-OVL framing.

`report.py` has `KernelReport.component_attribution: Optional[dict]` rendered in
`to_text()` / `to_dict()`. `run_report.py` attaches the structural view in
`report_from_desgraph` and the measured view in `_merge_engine_attribution`.

Tests live in `tests/perfbound/test_component_attribution.py` (10 tests). Some
assert the old scalar/US-SB-007 note — you will update those.

---

## 3. What to build

### 3.1 Schedule-overlap walker in `bound_combiner.py`

Port the logic from `scripts/overlap_walker_prototype.py` into a private helper.
**Important constraint:** `OpRecord` (see `perfbound/extract/hivm_extractor.py`)
carries `start_cycle`, `end_cycle`, `pipe`, `op_name`, `duration_cycles`,
`loop_multiplier`, `elements` — but **does NOT carry `is_sync`/`is_barrier`**
(the JSON has them; the loader drops them). So categorize from `pipe` + `op_name`:

```python
_OVL_COMPUTE = {"PIPE_V", "PIPE_M"}
_OVL_MEMORY  = {"PIPE_MTE2_V", "PIPE_MTE2_C", "PIPE_MTE3", "PIPE_MTE1", "PIPE_FIX"}
_OVL_CONTROL = {"PIPE_S", "PIPE_ALL"}
_SYNC_OPS    = {"wait_flag", "set_flag", "pipe_barrier",
                "sync_block_wait", "sync_block_set"}

def _ovl_category(op) -> str:
    if op.op_name in _SYNC_OPS or op.pipe in _OVL_CONTROL:
        return "control_sync"
    if op.pipe in _OVL_COMPUTE:
        return "compute"
    if op.pipe in _OVL_MEMORY:
        return "memory"
    return "other"
```

Implement `_schedule_overlap(operations) -> dict` that sweeps the emitted
schedule using start/end-cycle events (algorithm is in the prototype's `sweep()`):
return `{"critical_path": int, "exposed_control_cycles": int,
"control_overlapped_cycles": int, "n_sync_ops": int,
"control_busy_frac": float}`. **Degenerate-schedule guard**: if
`max(end_cycle) == 0` (synthetic extracts with no schedule), return
`critical_path=0` and let the caller emit `None` for overlap fields — do not
divide by zero.

### 3.2 Extend `ComponentAttribution` + `attribute_by_component`

Add fields (all `Optional`, `None` when unavailable):
```python
critical_path_cycles: Optional[int] = None
model_exposed_control_frac: Optional[float] = None   # of critical path (e.g. 0.119)
control_busy_frac: Optional[float] = None             # structural (e.g. 0.764)
n_sync_ops: Optional[int] = None
gap_ovl_pts: Optional[float] = None                   # measured_scalar - model_exposed
gap_ovl_us: Optional[float] = None                    # µs, capped (see 3.3)
```
Include them in `ComponentAttribution.to_dict()`.

Change the signature to receive the timing anchors:
```python
def attribute_by_component(extract, binding_component=None, measured=None,
                           t_bound_dsl_us=None, t_measured_us=None):
```
Compute the structural split (unchanged) **and** call `_schedule_overlap`. Set
`model_exposed_control_frac = exposed_control_cycles / critical_path` (None if
critical_path == 0).

### 3.3 The Gap-OVL metric (read the caveat — this is the crux)

The naive comparison mixes denominators: `model_exposed_control_frac` is a
fraction of the **idealized critical path**; `measured_scalar_frac` (from
`measured.scalar_frac`, i.e. msprof `aiv_scalar_ratio`) is a fraction of the
**wall-clock**. So:

- **Primary, always reported** (denominator-honest as *points*, not µs):
  - `gap_ovl_pts = measured_scalar_frac - model_exposed_control_frac`
    (e.g. 0.845 − 0.119 = +0.726). This is the overlap deficit.
- **Secondary µs estimate, only when `t_measured_us` and `t_bound_dsl_us` are
  both known, and CAPPED to never exceed the headroom** (prevents double-counting
  / a value larger than the gap it explains):
  ```python
  author_headroom_us = t_measured_us - t_bound_dsl_us
  raw = max(0.0, gap_ovl_pts) * t_measured_us
  gap_ovl_us = min(raw, author_headroom_us) if author_headroom_us > 0 else None
  ```
  Label it in output as **approximate** ("≈ … µs, capped at author headroom").
  Do **not** present it as a precise independent quantity.

`mis_binding` stays, but rewrite `note` to the overlap framing, e.g.:
> "Scalar is 84% of measured wall-clock but is 0/597 vectorizable ops — it is
> exposed control + sync overhead (402 sync/barrier ops). The DES schedule
> assumes the control path overlaps compute (model exposed 12%); the hardware
> serializes it (measured 84%). Gap-OVL ≈ {gap_ovl_us:.0f} µs overlap deficit.
> Action: reduce sync density (multi-buffer), hoist address math out of the tile
> loop, fuse tiles to amortize control. Diagnostic only — bound unchanged."

Remove all "US-SB-007 / scalar throughput / vector rate" wording.

### 3.4 Wire through `run_report.py`

- In `report_from_desgraph`, pass `t_bound_dsl_us=result.t_bound_us` and
  `t_measured_us=t_measured_us` into `attribute_by_component`.
- In `_merge_engine_attribution`, when an `EngineAttribution` is present,
  recompute `gap_ovl_pts`/`gap_ovl_us` now that `measured.scalar_frac` is known
  (the structural pass runs before the CSV is parsed). Simplest: re-call
  `attribute_by_component(extract, …, measured=ea, t_bound_dsl_us=…,
  t_measured_us=ea.t_measured_us)` — but `_merge_engine_attribution` currently
  lacks `extract`. Either (a) thread `extract` into the bridge, or (b) store the
  overlap primitives on the dict in the structural pass and finish the µs math in
  the bridge. Prefer (a): pass `extract` from `report_from_desgraph` →
  `KernelReport` is not the place; instead compute the full attribution
  (structural + overlap) in `report_from_desgraph`, and have the bridge only
  inject `measured.*` + finalize `gap_ovl_*`. Keep it diagnostic and idempotent.

### 3.5 Report rendering (`report.py`)

Extend the "Author-Headroom Component Attribution (A.8)" block in `to_text()` to
print, when present:
```
  overlap: model exposes 11.9% control / measured 84.5% scalar  -> Gap-OVL +72.6 pts (≈ 58,216 µs, capped)
  402 sync/barrier ops; control/sync = 76.4% of structural busy
```
Keep it under the existing structural + measured lines.

---

## 4. Spec rewrite (`.omc/specs/author_headroom_gap_analysis.md`)

This is a v2→v3 edit. Keep the good parts (data-availability matrix, refutations
of Gap-A/Gap-B, the §2.2 component tables). Change:

- **§1**: retitle the dominant mechanism from "Unmodeled Scalar Cost" to
  "Exposed Control/Sync Overlap Deficit (Gap-OVL)". State the three facts from
  §1 of this guide (0/597 vectorizable; 402 sync ops; 11.9% modeled vs 84.5%
  measured overlap). Explicitly retract the scalar-floor / US-SB-007 conclusion
  with one sentence ("v2 mislabeled this as scalar compute; it is overlap
  deficit").
- **§3**: replace "Gap-S" with "Gap-OVL". Define it as exposed control/sync time
  the schedule assumes overlaps but the hardware serializes; extraction = the
  schedule overlap walk vs msprof scalar share; magnitude = the capped µs.
  Keep Gap-E (wave quant, ~45 µs), Gap-C (overlap calibration — note it is now
  subsumed by Gap-OVL), and the Gap-A/Gap-B refutations.
- **§4**: redraw the decomposition with Gap-OVL as the dominant bucket.
- **§6 roadmap**: P0 becomes "wire Gap-OVL (this task) + reduce-sync-density
  recommendation"; drop the scalar-calibration P0.
- Add `scripts/overlap_walker_prototype.py` to §9 sources.

Keep the spec honest about the denominator caveat (§3.3 here).

---

## 5. Tests (`tests/perfbound/test_component_attribution.py`)

Update + add:
- **Update** `test_mis_binding_flagged_…` and any test asserting the old note:
  assert the note now contains "overlap" / "sync" and **not** "US-SB-007".
- **Add** `test_schedule_overlap_on_real_des`: load `.omc/research/hw_runs/
  kda_des.json` (skip if absent), assert `model_exposed_control_frac` ∈
  (0.05, 0.30) and `gap_ovl_pts > 0.5` and `n_sync_ops > 300`.
- **Add** `test_schedule_overlap_synthetic`: build a 3-op `HIVMExtract` with
  explicit `start_cycle`/`end_cycle` where a PIPE_S op runs alone for N cycles
  then a PIPE_V op overlaps a second PIPE_S op; assert exposed vs overlapped
  cycle counts match hand-computed values.
- **Add** `test_degenerate_schedule_no_div_zero`: extract whose ops all have
  `start_cycle == end_cycle == 0` → overlap fields are `None`, no exception.
- **Add** `test_gap_ovl_us_capped_at_headroom`: with synthetic
  `t_measured_us`/`t_bound_dsl_us`, assert `gap_ovl_us <= author_headroom_us`.
- Keep the existing end-to-end test; assert the report text now shows the
  overlap line.

Run with the project convention:
```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest tests/perfbound/ \
  -p no:cacheprovider -q
```
Expected: full suite green (currently 380 passed / 3 skipped / 2 xfailed; your
new tests add to the pass count). Also run the end-to-end report to eyeball:
```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m perfbound.combine.run_report \
  --desgraph .omc/research/hw_runs/kda_des.json --grid 128,32 \
  --kernel-name chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2 \
  --measured-csv .omc/research/hw_runs/chunk_kda_op_summary.csv \
  --measured-op-name chunk_kda_bwd
```
The A.8 block must show the overlap line and the Gap-OVL note.

---

## 6. Constraints (project rules — non-negotiable)

- **Diagnostic only**: never let Gap-OVL alter `t_bound_us`, `t_core_floor_us`,
  the component floor, or any soundness invariant (T_bound ≤ T_measured).
- **No regex for IR/TTIR parsing**: you are reading already-parsed `OpRecord`
  fields and JSON; do not introduce regex over MLIR text.
- **Git safety**: stage explicit paths only — never `git add -A` / `git add .` /
  `git clean -fd`. Verify `git status --ignore-submodules=all --short` shows zero
  unexpected `D` (deletions) and no `thirdparty/` changes before committing.
  Push (if asked) with `--recurse-submodules=no`. Do NOT commit `.omc/state/`,
  `.omc/sessions/`, `.omc/project-memory.json`, or `.claude/`.
- **Commit footer**: end the commit message with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Authoring vs review**: implement, then verify with a fresh test run; do not
  declare done without green output pasted/observed.
- Commit only when the user asks. Suggested message subject if so:
  `refactor(A.8): reshape author headroom to Gap-OVL overlap deficit`.

---

## 7. Definition of done

- [ ] `_schedule_overlap` + extended `attribute_by_component` implemented, with
      degenerate-schedule guard.
- [ ] `ComponentAttribution` carries the 6 new Gap-OVL fields + to_dict.
- [ ] `run_report.py` threads `t_bound_dsl_us`/`t_measured_us`; Gap-OVL finalized
      once measured scalar is known.
- [ ] `report.py` renders the overlap line + rewritten note (no US-SB-007 text).
- [ ] Spec rewritten to v3 (Gap-OVL dominant; scalar-floor retracted).
- [ ] Tests updated + 4 new tests added; full perfbound suite green.
- [ ] End-to-end `run_report` shows the overlap diagnosis on chunk_kda.
- [ ] No bound value changed anywhere (grep the report JSON before/after:
      `t_bound_us`, `t_core_floor_us`, `author_headroom_us` identical).
```
