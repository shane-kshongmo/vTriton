# kda_generate `final_ops` perfbound analysis

## Scope

- Refreshed: 2026-09-03
- Source: `fengrui886/kda_generate`, commit
  `a77b6d430ae2e9e97c1c61d17a09df7d7811a02e`
- Workload: case index 0 from each checked-in workload JSON
- Hardware: Ascend 910B3, 20 AIC / 40 AIV
- Hardware config: `configs/ascend_910b3_v4.json`
- Calibration: `perfbound/calibration/data/calib_910b3_v1.json`
- Model inputs: pre-outline TTAdapter semantic IR, post-GraphSync HIVM IR,
  and hardware msprof task profiles
- Simulator traces: calibration and validation evidence only; never used as
  per-kernel model input

The authoritative refreshed generated set is under
`/tmp/kda_perfbound_refresh_20260903`. Each report records SHA-256 hashes for
its DES graph, hardware profile, hardware config, and calibration database.
The stale-report audit is
`/tmp/kda_perfbound_refresh_20260903/stale_report_audit.json`.

## Profiling Method

Each valid profile was collected in a separate `msprof` invocation on the
configured 910B3 host. The process completed compilation and autotuning before
dynamic profiling attached, then executed ten measured launches. The report
parser discarded the first captured launch and used the median of the remaining
nine. This avoids the 2,940 autotune-contaminated rows observed in the original
`chunk_kda_bwd_dAv` launch-mode profile.

`chunk_kda_fwd_intra_inter_solve_fused` is excluded from headroom evaluation:
its first launch completed, but repeated post-compile execution raised AICore
timeout `507014`. One successful task is not accepted as profiling evidence.

## Model Coverage

All 15 semantic/HIVM DES graphs were regenerated with the current analyzer:

- `schedule_truncated=false` for all 15
- `model_coverage.status=complete` for all 15
- zero unresolved semantic loops and branches
- zero dependency cycles in the three graphs that previously deadlocked
- `trace_timing_status=partial` for all 15 because some aggregate semantic work
  still lacks exact call-level timeline placement

The deadlocks in `chunk_kda_bwd_dAv`,
`chunk_kda_fwd_intra_intra_sub_chunk`, and
`chunk_kda_fwd_intra_token_parallel` were artificial. Static sync flags were
named `flag_N`, while equivalent resolved dynamic values were named `N`; waits
were therefore connected to later loop sets instead of initial pre-signals.
Canonicalizing both forms removed the cycles.

## Refreshed Headroom

`T_DSL` and `T_HIVM` are equal for every case, so the current two-limit model
finds no compiler placement/avoidable-handoff floor shift. `Ceiling` is
`hardware task duration - T_HIVM`; it is emitted only when model coverage,
calibration, task provenance, and bound ordering all pass.

| Op | Task us | T_DSL / T_HIVM us | Ceiling us | Speedup ceiling | Status | Top diagnostic opportunity |
|---|---:|---:|---:|---:|---|---|
| `chunk_gla_fwd_kernel_o` | 82.293 | 42.119 | 40.174 | 1.954x | sound ceiling | Gap 4, intra-unit utilization |
| `chunk_kda_bwd_dAv` | 8.039 | 11.064 | n/a | n/a | bound violation | suppressed |
| `chunk_kda_bwd_intra` | 73.674 | 17.896 | 55.778 | 4.117x | sound ceiling | Gap 4, intra-unit utilization |
| `chunk_kda_bwd_wy_dqkg_fused` | 13.779 | 13.666 | 0.113 | 1.008x | sound ceiling | Gap 4, intra-unit utilization |
| `chunk_kda_fwd_intra_inter_solve_fused` | n/a | 13.511 | n/a | n/a | hardware profile invalid | suppressed |
| `chunk_kda_fwd_intra_intra_sub_chunk` | 43.297 | 12.948 | 30.349 | 3.344x | sound ceiling | Gap 4, intra-unit utilization |
| `chunk_kda_fwd_intra_token_parallel` | 27.018 | 11.112 | 15.906 | 2.431x | sound ceiling | Gap 4, intra-unit utilization |
| `chunk_local_cumsum` | 12.639 | 4.958 | 7.681 | 2.549x | sound ceiling | Gap 4, intra-unit utilization |
| `fused_recurrent_kda_fwd` | 791.237 | 647.406 | 143.831 | 1.222x | sound ceiling | Gap 4, intra-unit utilization |
| `intracard_fwd_h` | 60.296 | 29.202 | 31.094 | 2.065x | sound ceiling | Gap 4, intra-unit utilization |
| `kda_gate_bwd` | 12.619 | 17.606 | n/a | n/a | bound violation | suppressed |
| `kda_gate_chunk_cumsum` | 16.738 | 14.424 | 2.314 | 1.160x | sound ceiling | Gap 4, intra-unit utilization |
| `kda_gate_fwd` | 11.079 | 9.597 | 1.482 | 1.154x | sound ceiling | Gap 4, intra-unit utilization |
| `prepare_wy_repr_bwd` | 19.438 | 13.743 | 5.695 | 1.414x | sound ceiling | Gap 4, intra-unit utilization |
| `recompute_w_u_fwd` | 7.399 | 11.839 | n/a | n/a | bound violation | suppressed |

These are theoretical ceilings, not achievable point estimates. Gap values rank
investigation priority only; they are non-additive and do not prove independent
savings. A correctness-verified counterfactual hardware measurement is still
required to claim achievable headroom.

The largest valid absolute ceilings are fused recurrent (`143.831 us`),
backward intra (`55.778 us`), GLA (`40.174 us`), intracard forward
(`31.094 us`), and intra-sub-chunk (`30.349 us`). The fused backward kernel is
already at its modeled floor within `0.113 us`; gate forward and gate chunk
cumsum have only `1.482 us` and `2.314 us` ceilings.

## Invalid Models

No headroom is reported for the three bound violations:

- `chunk_kda_bwd_dAv`: floor exceeds task by `3.025 us`; the model includes an
  `8.239 us` mandatory serialization term that requires revalidation.
- `kda_gate_bwd`: Vector/grid floor exceeds task by `4.987 us`; semantic Vector
  costing or grid scaling is too pessimistic for this workload.
- `recompute_w_u_fwd`: floor exceeds task by `4.440 us`; the same `8.239 us`
  mandatory serialization term dominates and requires revalidation.

These are model defects or calibration mismatches, not negative headroom. Their
rankings and ceilings are deliberately suppressed.

## Stale-Report Audit

All 15 pre-refresh reports were stale. Their snapshot is
`/tmp/kda_perfbound_stale_snapshot_20260902`. Every old report had all of these
issues:

- missing `modeling_output_v1`
- event elapsed used instead of hardware `msprof_task_duration`
- missing hardware task measurement and profile provenance
- missing input hashes
- older than the current analyzer or an input artifact

Additional non-canonical artifacts:

| Artifact | Audit result |
|---|---|
| `/tmp/kda_perfbound_nooverlay` | stale incomplete intermediate; semantic overlay absent and no report layer |
| `/tmp/chunk_kda_fwd_intra_token_parallel_case0.bound.des.json` | stale legacy one-kernel DES without `model_coverage` |
| `/tmp/kda_sim_trace_summary.json` | validation-only simulator artifact, not a model input |
| `/tmp/kda_final_ops_hw_20260902_csv` | stale launch-mode profile pass contaminated by compilation/autotuning |

Only `/tmp/kda_perfbound_refresh_20260903` is authoritative for this run.
Simulator-derived spans must not be substituted for the hardware task values
recorded in `modeling_output.profile_inputs`.

## Remaining Risks

- All modeled Perfetto traces remain timing-partial, so they cannot support
  exact per-pipe attribution.
- The three bound violations must be fixed before their headroom can be ranked.
- Inter-solve requires a correctness/runtime fix before repeated profiling.
- The 11 valid results are upper ceilings only; none has a measured optimized
  counterfactual, so `achievable_headroom.status=not_established` throughout.
