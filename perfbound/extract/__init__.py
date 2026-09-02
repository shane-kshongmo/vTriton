# M2/M3 — DSL + HIVM Extractors
#
# M2 (DSL Extractor / Tier 1 input):
#   Parse Triton DSL / TTIR to recover program_id → tile affine map,
#   occupancy, load balance, redundancy(grid).  Hardware-legality
#   constraints (buffer capacity, register pressure, divisibility)
#   are first-class — illegal tilings are rejected.
#
# M3 (HIVM Extractor / Tier 2 input):
#   Consume C++-emitted structural JSON (emitDESGraph + dependency graph)
#   to extract per-component O_prec, transfer sizes, unit assignment,
#   and the handoff/barrier list.  Thin consumer — the heavy MLIR walking
#   is done in C++ HIVMAnalysis / PipelineAnalysis.

from .hivm_extractor import (
    HIVMExtract,
    OpRecord,
    HandoffRecord,
    load_hivm_desgraph,
    load_des_metadata,
    load_pipeline_depgraph,
    extract_hivm,
)
from .hivm_runner import extract_from_npuir
from .op_classifier import classify_op, Component, Precision
from .eligibility_oracle import get_eligibility, compute_gap1
from .semantic_extractor import (
    SemanticOpRecord,
    Gap1Report,
    classify_ttir_op,
    extract_semantic_records,
    analyze_gap1,
    analyze_gap1_from_extract,
)
from .dsl_extractor import GridInfo, extract_grid_info
from .mlir_parser import parse_ttir
