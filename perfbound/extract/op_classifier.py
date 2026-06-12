"""
M3 — Op classifier: op → (component, precision).

Ports the classification logic from tilesim's arc_spec/entity.py but
simplified for the bound model (no cost-model execution, only classification).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple


class Component(str, Enum):
    """Six roofline components from spec §2.1."""
    CUBE = "cube"
    VECTOR = "vector"
    SCALAR = "scalar"
    MTE_GM = "mte_gm"
    MTE_L1 = "mte_l1"
    MTE_UB = "mte_ub"


class Precision(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    INT8 = "int8"
    INT32 = "int32"


# HIVM pipe → Component mapping
# Covers both raw HIVM pipe names (typed ingestion) and C++ stringifyPipe()
# output names (emitDESGraph JSON path).  See HIVMAnalysis::stringifyPipe().
PIPE_TO_COMPONENT = {
    # Raw HIVM names (typed ingestion via BiShengIR)
    "Cube": Component.CUBE,
    "CubeMTE2": Component.MTE_GM,
    "MTE1": Component.MTE_L1,
    "FixPipe": Component.MTE_UB,
    "Vector": Component.VECTOR,
    "VectorMTE2": Component.MTE_GM,
    "MTE3": Component.MTE_UB,
    "Scalar": Component.SCALAR,
    # HIVMAnalysis.cpp internal pipe names (PIPE_* enum)
    "PIPE_CUBE": Component.CUBE,
    "PIPE_M": Component.CUBE,         # Matrix (Cube)
    "PIPE_V": Component.VECTOR,
    "PIPE_S": Component.SCALAR,
    "PIPE_MTE2_C": Component.MTE_GM,  # Cube MTE2: GM → L1
    "PIPE_MTE2_V": Component.MTE_GM,  # Vector MTE2: GM → UB
    "PIPE_MTE1": Component.MTE_L1,    # MTE1: L1 → L0A/B
    "PIPE_MTE3": Component.MTE_UB,    # MTE3: UB → GM
    "PIPE_FIX": Component.MTE_UB,     # FixPipe: L0C → GM
    "PIPE_ALL": Component.SCALAR,     # catch-all for sync/unclassified ops; filtered as sync
}

# PipelineAnalysis hw_unit → Component mapping
HW_UNIT_TO_COMPONENT = {
    "Cube": Component.CUBE,
    "CubeMTE2": Component.MTE_GM,
    "FixPipe": Component.MTE_UB,
    "Vector": Component.VECTOR,
    "VecMTE2": Component.MTE_GM,
    "MTE3": Component.MTE_UB,
    "Scalar": Component.SCALAR,
}

# Element type string → Precision
ELEM_TYPE_TO_PRECISION = {
    "f16": Precision.FP16,
    "bf16": Precision.BF16,
    "f32": Precision.FP32,
    "int8": Precision.INT8,
    "i8": Precision.INT8,
    "i32": Precision.INT32,
}

# Op name patterns → Component (for ops without pipe info)
_OP_PATTERNS = {
    "matmul": Component.CUBE,
    "mm": Component.CUBE,
    "bmm": Component.CUBE,
    "add": Component.VECTOR,
    "sub": Component.VECTOR,
    "mul": Component.VECTOR,
    "div": Component.VECTOR,
    "exp": Component.VECTOR,
    "log": Component.VECTOR,
    "sqrt": Component.VECTOR,
    "rsqrt": Component.VECTOR,
    "tanh": Component.VECTOR,
    "sigmoid": Component.VECTOR,
    "gelu": Component.VECTOR,
    "relu": Component.VECTOR,
    "abs": Component.VECTOR,
    "neg": Component.VECTOR,
    "cast": Component.VECTOR,
    "reduce_sum": Component.VECTOR,
    "reduce_max": Component.VECTOR,
    "reduce_min": Component.VECTOR,
    "broadcast": Component.VECTOR,
    "select": Component.VECTOR,
    "load": Component.MTE_GM,
    "store": Component.MTE_UB,
    "copy": Component.MTE_GM,
    "dma": Component.MTE_GM,
}


def classify_op(
    op_name: str,
    pipe: str = "",
    elem_type: str = "",
) -> Tuple[Component, Optional[Precision]]:
    """Classify an operation into (component, precision).

    Priority: pipe mapping > op name pattern > default (Scalar).
    """
    # Component from pipe
    component = PIPE_TO_COMPONENT.get(pipe)
    if component is None and pipe:
        component = HW_UNIT_TO_COMPONENT.get(pipe)

    # Fallback: component from op name
    if component is None:
        op_lower = op_name.lower()
        for pattern, comp in _OP_PATTERNS.items():
            if pattern in op_lower:
                component = comp
                break
        if component is None:
            component = Component.SCALAR

    # Precision from element type
    precision = ELEM_TYPE_TO_PRECISION.get(elem_type)

    return component, precision
