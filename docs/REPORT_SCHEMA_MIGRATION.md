# Perfbound report schema migration

Canonical reports use top-level `schema_version: "perfbound_report_v2"` and
store interpreted results under `modeling_output`. The removed v1 aliases are
available only through an opt-in compatibility adapter.

The nested `modeling_output.schema_version` remains `modeling_output_v1`; it
versions that object independently from the top-level report envelope.

## Generate compatibility output

From `perfbound.combine.run_report`:

```bash
python3 -m perfbound.combine.run_report \
  --desgraph kernel.des.json \
  --grid 128,32 \
  --output-json kernel.compat.json \
  --legacy-report-aliases
```

From the Stage-B pipeline:

```bash
python3 scripts/run_bound.py \
  --kernel vector_add \
  --grid 128,32 \
  --legacy-report-aliases
```

Convert an existing canonical report:

```bash
python3 -m perfbound.combine.report_compat \
  kernel.report.json \
  --output kernel.compat.json
```

Python consumers can call:

```python
from perfbound.combine.report_compat import with_legacy_report_aliases

legacy_view = with_legacy_report_aliases(canonical_report)
```

The adapter returns a copy and does not mutate canonical input. Compatibility
output contains:

```json
{
  "schema_version": "perfbound_report_v2",
  "compatibility": {
    "legacy_report_aliases": "v1",
    "deprecated": true
  }
}
```

## Consumer replacements

| Removed v1 path | Canonical v2 path |
|---|---|
| `reachability.t_bound_hivm_us` | `modeling_output.bounds.hivm_floor_us` |
| `reachability.t_bound_dsl_us` | `modeling_output.bounds.dsl_floor_us` |
| `reachability.t_measured_us` | `modeling_output.profile_inputs.task_duration_us` |
| `reachability.msprof_source` | `modeling_output.profile_inputs.msprof_source` |
| `reachability.n_invocations` | `modeling_output.profile_inputs.invocations` |
| `profile.diagnosis` | `modeling_output.profile_inputs.diagnosis` |
| `profile.dominant_component` | `modeling_output.profile_inputs.dominant_component` |
| `profile.exposed_control_frac_measured` | `modeling_output.profile_inputs.exposed_control_fraction_measured` |
| `profile.exposed_control_frac_model` | `modeling_output.profile_inputs.exposed_control_fraction_model` |
| `profile.exposed_control_deficit_pts` | `modeling_output.profile_inputs.exposed_control_deficit_points` |
| `profile.exposed_control_deficit_us` | `modeling_output.profile_inputs.exposed_control_deficit_us` |
| `profile.n_sync_ops` | `modeling_output.profile_inputs.sync_operations` |
| `event_elapsed_us` | `modeling_output.screening_metrics.event_elapsed_us` |
| `event_elapsed_source` | Unavailable in v2; compatibility value is `null` |
| `calibration.derived_constant_count` | Unavailable in v2; compatibility value is `null` |
| `headroom_assessment.upper_us` | No equivalent alias; select the appropriate `modeling_output.theoretical_ceilings.*` ceiling |
| `headroom_assessment.point_estimate_us` | `modeling_output.achievable_headroom.point_estimate_us` |

`headroom_assessment` in compatibility output is intentionally an unavailable
shell with no numeric range. The deleted v1 heuristic was not equivalent to the
v2 sound theoretical ceiling and cannot be reconstructed safely. Consumers
must read `modeling_output.theoretical_ceilings` directly for current ceilings.

Compatibility aliases are transitional. New consumers must use canonical v2
paths and should reject unknown `schema_version` values explicitly.
