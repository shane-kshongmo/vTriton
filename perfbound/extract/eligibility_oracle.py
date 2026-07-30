"""
M3 — Eligibility Oracle.

Determines which hardware units each op *could* run on based on TTIR/Linalg
semantics, independent of what HIVM actually assigned. This is the Gap 1 input:

    Gap 1 = realized_assignment(op) ∉ eligibility(op)

Rules:
  - matmul + (FP16|INT8) → {Cube}
  - element-wise / reduction → {Vector}
  - type-incompatible (e.g., i32 compare) → {Scalar}
  - MTE ops are unit-specific (no placement choice)

Hardware-realistic semantics:
  Eligibility reflects what the hardware *can* do, not what would be ideal.
  i32-compare on Scalar is NOT a Gap 1 — Scalar is the only hardware option.
  The seeded Gap 1 case is fp16-compare forced to Scalar (compiler bug).

Conservative: if uncertain, include MORE eligible units (fewer false Gap 1s).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Set

from .op_classifier import Component, Precision


# Eligible components per (op_category, precision)
# Conservative: include more units when uncertain
_ELIGIBILITY_RULES: Dict[str, Dict[str, FrozenSet[Component]]] = {
    "matmul": {
        "fp16": frozenset({Component.CUBE}),
        "bf16": frozenset({Component.CUBE}),
        "int8": frozenset({Component.CUBE}),
        "fp32": frozenset({Component.CUBE, Component.VECTOR}),  # can fall back to vector
    },
    "elementwise": {
        "fp16": frozenset({Component.VECTOR}),
        "bf16": frozenset({Component.VECTOR}),
        "fp32": frozenset({Component.VECTOR}),
        "int8": frozenset({Component.VECTOR}),
        "int32": frozenset({Component.VECTOR, Component.SCALAR}),
    },
    "reduction": {
        "fp16": frozenset({Component.VECTOR}),
        "bf16": frozenset({Component.VECTOR}),
        "fp32": frozenset({Component.VECTOR}),
    },
    "compare": {
        "fp16": frozenset({Component.VECTOR}),
        "bf16": frozenset({Component.VECTOR}),
        "fp32": frozenset({Component.VECTOR}),
        "int32": frozenset({Component.SCALAR}),  # i32 compare → Scalar fallback
    },
    "cast": {
        "default": frozenset({Component.VECTOR}),
    },
}


# Op-name keyword → eligibility category.  Shared by the Gap-1 attribution
# (bound_combiner) and the two-limit idealized-extract reassignment so the
# name→category mapping has a single source of truth.
_MATMUL_KEYWORDS = ("matmul", "mm", "bmm")
_REDUCTION_KEYWORDS = ("reduce", "sum", "max", "min", "arg")
_COMPARE_KEYWORDS = ("cmp", "compare")


def op_category_for_name(op_name: str) -> str:
    """Map an op name to an eligibility-oracle category.

    Returns one of the _ELIGIBILITY_RULES keys that get_eligibility consumes:
    "matmul", "reduction", "compare", else "elementwise".
    """
    lower = op_name.lower()
    if any(k in lower for k in _MATMUL_KEYWORDS):
        return "matmul"
    if any(k in lower for k in _REDUCTION_KEYWORDS):
        return "reduction"
    if any(k in lower for k in _COMPARE_KEYWORDS):
        return "compare"
    return "elementwise"


def get_eligibility(
    op_category: str,
    precision: str | None = None,
) -> FrozenSet[Component]:
    """Get the set of components an op *could* run on.

    Args:
        op_category: "matmul", "elementwise", "reduction", "compare", "cast"
        precision: element precision string ("fp16", "bf16", "fp32", "int8", "int32")

    Returns:
        FrozenSet of eligible Components.
    """
    rules = _ELIGIBILITY_RULES.get(op_category, {})
    if precision and precision in rules:
        return rules[precision]
    if "default" in rules:
        return rules["default"]
    # Conservative: union all eligible sets for this category.
    # If no rules at all, return all compute components (maximally conservative).
    if not rules:
        return frozenset({Component.CUBE, Component.VECTOR, Component.SCALAR})
    return frozenset().union(*rules.values())


def compute_gap1(
    op_id: int,
    op_category: str,
    precision: str | None,
    realized_component: Component,
) -> bool:
    """Whether there is a Gap 1 (wrong-unit placement) for this op.

    Gap 1 exists if the realized unit is NOT in the eligible set.
    """
    eligible = get_eligibility(op_category, precision)
    return realized_component not in eligible
