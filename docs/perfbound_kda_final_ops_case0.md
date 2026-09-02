# kda_generate `final_ops` perfbound analysis

## Scope

- Source: `fengrui886/kda_generate`, commit `a77b6d430ae2e9e97c1c61d17a09df7d7811a02e`
- Workload: case index 0 from each op's checked-in workload JSON
- Hardware: Ascend 910B3, `configs/ascend_910b3_v4.json` (20 AIC / 40 AIV)
- Calibration: `perfbound/calibration/data/calib_910b3_v1.json`
- Screening measurement: `triton.testing.do_bench` event elapsed on the configured Ascend 910B3 host
- Environment: shared host with unrelated profiling jobs; treat timings as screening data

## Result

All case-0 loop bounds and model-relevant branches resolve. The post-GraphSync
NPUIR still omits outlined Vector bodies, so `tritonsim-hivm` now overlays typed
work from the captured pre-outline TTAdapter IR while retaining GraphSync
dependencies. All 15 reports have `model_coverage.status=complete`, zero
unsupported semantic ops, and zero unresolved model-relevant control flow.

The original `triton.testing.do_bench` values are end-to-end event elapsed
times, not the `msprof_task_duration` metric targeted by the model. They remain
visible as screening data but are not used for headroom.

| Op | Event elapsed us | Conservative task floor us |
|---|---:|---:|
| `chunk_gla_fwd_kernel_o` | 594.04 | 42.12 |
| `chunk_kda_bwd_dAv` | 289.42 | 11.06 |
| `chunk_kda_bwd_intra` | 412.46 | 17.90 |
| `chunk_kda_bwd_wy_dqkg_fused` | 345.52 | 13.67 |
| `chunk_kda_fwd_intra_inter_solve_fused` | n/a | 13.51 |
| `chunk_kda_fwd_intra_intra_sub_chunk` | 262.76 | 12.95 |
| `chunk_kda_fwd_intra_token_parallel` | 327.53 | 11.11 |
| `chunk_local_cumsum` | 605.52 | 4.96 |
| `fused_recurrent_kda_fwd` | 790.90 | 647.41 |
| `intracard_fwd_h` | 347.29 | 29.20 |
| `kda_gate_bwd` | 272.47 | 17.61 |
| `kda_gate_chunk_cumsum` | 625.44 | 14.42 |
| `kda_gate_fwd` | 294.28 | 9.60 |
| `prepare_wy_repr_bwd` | 340.24 | 13.74 |
| `recompute_w_u_fwd` | 307.12 | 11.84 |

## Profile Evidence

| Op | msprof task us | Event median us | Main evidence |
|---|---:|---:|---|
| `fused_recurrent_kda_fwd` | 791.80 | 790.84 | 647.41 us floor; 735.10 us Vector active |
| `chunk_gla_fwd_kernel_o` | 88.61 | 565.04 | 42.12 us floor; roughly 500 us event task wait |
| `kda_gate_fwd` | 12.90 | 231.66 | 9.60 us floor; roughly 220 us event task wait |

The previous transcendental calibration assigned one composite
Exp+Ln+Sqrt+Rsqrt duration to each individual opcode. Isolated 45-run FP16 and
FP32 microbenchmarks replace that invalid attribution; all have CV below 0.6%.
The corrected fused floor is within 12% of msprof Vector active time.

## Simulator Trace Evidence

`msprof op simulator` was run with the Ascend910B3 simulator library first in
`LD_LIBRARY_PATH`, the logical runtime `--kernel-name`, `--launch-count=1`, and
`--aic-metrics=PipeUtilization`. BiSheng may report an internal `_mix_aic`
code object after matching a logical MIX kernel; the suffix is not used in the
filter. Full traces remain under `/tmp/msprof_sim_*`; simulator values below
are slowest-core spans. Analytical body excludes the 1 us launch term and is
therefore directly comparable to the simulator kernel span.

| Logical kernel | Analytical body us | Simulator us | Simulator - body us | Status |
|---|---:|---:|---:|---|
| `gla_fwd_o_kernel` | 41.12 | 42.75 | 1.63 | valid |
| `chunk_bwd_dAv_kernel` | 10.06 | 8.32 | -1.75 | model/simulator inversion |
| `chunk_kda_bwd_kernel_intra` | 16.90 | n/a | n/a | invalid simulator LD/ST addresses |
| `_chunk_kda_bwd_wy_dqkg_fused_kernel` | 12.67 | n/a | n/a | simulator-target compiler failure |
| `chunk_kda_fwd_intra_inter_solve_fused` | 12.51 | n/a | n/a | no compiled `npubin` |
| `chunk_intra_sub_chunk_kernel` | 11.95 | 36.71 | 24.76 | valid |
| `intra_token_parallel_kernel` | 10.11 | 40.91 | 30.80 | valid |
| `chunk_local_cumsum_vector_kernel` | 3.96 | 13.72 | 9.76 | valid |
| `fused_recurrent_kernel` | 646.41 | 758.32 | 111.91 | valid |
| `intracard_fwd_h_kernel` | 28.20 | 57.09 | 28.89 | valid |
| `kda_gate_bwd_kernel` | 16.61 | 10.59 | -6.01 | model/simulator inversion |
| `kda_gate_chunk_cumsum_kernel` | 13.42 | 14.97 | 1.55 | valid |
| `kda_gate_fwd_kernel` | 8.60 | 9.26 | 0.66 | valid |
| `prepare_wy_repr_bwd_kernel` | 12.74 | n/a | n/a | invalid simulator LD/ST addresses |
| `recompute_w_u_fwd_kernel` | 10.84 | 3.95 | -6.89 | model/simulator inversion |

| Logical kernel | Dominant non-additive pipe spans |
|---|---|
| `gla_fwd_o_kernel` | Flow-control 27.74 us; MTE2 15.65 us; MTE3 11.03 us |
| `chunk_bwd_dAv_kernel` | MTE3 5.23 us; MTE2 3.03 us; flow-control 1.79 us |
| `chunk_intra_sub_chunk_kernel` | Vector 23.54 us; Scalar-LDST 21.23 us; MTE3 6.09 us |
| `intra_token_parallel_kernel` | Scalar-LDST 33.43 us; MTE3 8.59 us; MTE2 5.29 us |
| `chunk_local_cumsum_vector_kernel` | Vector 7.20 us; MTE2 4.64 us; MTE3 4.16 us |
| `fused_recurrent_kernel` | Vector 742.74 us; Scalar-LDST 498.51 us; MTE2 224.79 us |
| `intracard_fwd_h_kernel` | Flow-control 50.90 us; MTE3 32.94 us; MTE2 29.04 us |
| `kda_gate_bwd_kernel` | Vector 7.50 us; MTE2 2.62 us; Scalar-LDST 2.51 us |
| `kda_gate_chunk_cumsum_kernel` | Vector 11.03 us; Scalar-LDST 5.29 us; MTE3 2.43 us |
| `kda_gate_fwd_kernel` | Vector 5.91 us; MTE2 1.89 us; MTE3 1.89 us |
| `recompute_w_u_fwd_kernel` | Flow-control 2.23 us; MTE2 0.58 us; Scalar-LDST 0.40 us |

Pipe spans overlap and must not be added together.

### Model Perfetto Trace Validation

The generated simulator JSON was also compared directly with each
`tritonsim-hivm --perfetto-trace-file` JSON. Both traces were normalized to
union-busy pipe spans before comparison.

| Kernel | Model trace us | Simulator trace us | Span error |
|---|---:|---:|---:|
| `gla_fwd_o_kernel` | 49.96 | 42.75 | +16.9% |
| `chunk_bwd_dAv_kernel` | 0.56 | 8.32 | -93.2% |
| `chunk_intra_sub_chunk_kernel` | 0.47 | 36.71 | -98.7% |
| `intra_token_parallel_kernel` | 0.59 | 40.91 | -98.6% |
| `chunk_local_cumsum_vector_kernel` | 1.51 | 13.72 | -89.0% |
| `fused_recurrent_kernel` | 791.32 | 758.32 | +4.4% |
| `intracard_fwd_h_kernel` | 36.82 | 57.09 | -35.5% |
| `kda_gate_bwd_kernel` | 17.40 | 10.59 | +64.3% |
| `kda_gate_chunk_cumsum_kernel` | 6.23 | 14.97 | -58.4% |
| `kda_gate_fwd_kernel` | 9.78 | 9.26 | +5.6% |
| `recompute_w_u_fwd_kernel` | 3.84 | 3.95 | -2.8% |

The trace fixes reduce mean absolute span error from 59.1% to 51.6% and median
absolute error from 66.9% to 58.4%. Fused recurrent, gate forward, and
recompute are now within 15%; GLA is within 17%.

Two structural corrections produced the improvement. Vector work recovered
from TTAdapter is costed with the hardware-config opcode microbenchmarks,
including dtype-specific values, conserved exactly, projected onto existing
`vcall` nodes, and rescheduled. Local flags and barriers now use the one-cycle
issue cost observed in the simulator traces, while cross-core waits use a
separate flow-control resource and DES event visibility for their wait time.

Projection from whole-kernel semantic work to calls remains a weighted
heuristic because the post-outline NPUIR does not retain call-body correlation.
Every retained final-ops trace therefore reports `timing_coverage=partial` and
its count of unscheduled semantic aggregates. The remaining misses are
concentrated in MTE work without a call-level carrier, scalar address-generation
expansion, and scan/reduction lowering. A `model_coverage.status=complete`
analytical report must not be interpreted as a complete per-pipe timeline;
check `trace_timing_status` before using the trace for attribution.

The three model/simulator inversions invalidate headroom claims for those
kernels until semantic work accounting is corrected. Among valid traces,
fused recurrent, intra-token parallel, intra-sub-chunk, local cumsum, and
intracard retain material compiled-binary overhead. GLA and both profiled gate
kernels are close to their analytical bodies in simulation.

For the three kernels with hardware task measurements, the remaining
hardware-versus-simulator gaps are 45.86 us for GLA, 33.48 us for fused
recurrent, and 3.64 us for gate forward. These are not compiler headroom by
default. Fused recurrent retains a credible 111.91 us compiler/scheduling
ceiling, dominated by repeated Vector barriers and instruction/control
overhead; GLA and gate do not.

## Task Headroom

| Op | Task us | Floor us | Residual us | Ceiling speedup |
|---|---:|---:|---:|---:|
| `chunk_gla_fwd_kernel_o` | 88.61 | 42.12 | 46.49 | 2.10x |
| `fused_recurrent_kda_fwd` | 791.80 | 647.41 | 144.39 | 1.22x |
| `kda_gate_fwd` | 12.90 | 9.60 | 3.30 | 1.34x |

These residuals are theoretical ceilings, not proven attainable speedups. The
two-limit idealization reports zero compiler-placement headroom for these
graphs, so a correctness-checked counterfactual is still required before
labeling any residual as achievable. Simulator traces indicate a more credible
compiled-binary ceiling of roughly 0.66 us for gate, 1.63 us for GLA, and
111.91 us for fused recurrent. The large event-to-task gaps for GLA and gate
are host/runtime waiting, not kernel optimization headroom.
