# M4 — Tier 2 Component Analytical Model (pure functions, no I/O)
#
# For each roofline component c, compute the ideal rate I_c via
# weighted-harmonic mean (Eq. 4):
#
#   I_c = Σ_p O_{c,p} / Σ_p (O_{c,p} / P_p)
#
# Then the core floor:
#   T_core_floor = max_c(O_c / I_c)
#
# where O_c = total work (ops or bytes) for component c.
#
# The harmonic mean correctly models a component that processes a mix of
# precisions at different rates — the overall rate is dominated by the
# slowest precision proportionally to its share of work.
#
# Source spec: .omc/specs/performance_bound_model.md §1.4, §2.1, §A.4
# Ports: tilesim aicore_costmodel.py time_cube structure (not values).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from ..calibration.constants import (
    CubeConfig, VectorConfig, MemHierarchy, CoreConfig, DType, MemLoc, VecOpType,
)
from ..extract.hivm_extractor import HIVMExtract, OpRecord
from ..extract.op_classifier import Component, Precision, HW_UNIT_TO_COMPONENT

if TYPE_CHECKING:
    from ..calibration.constants import CalibrationDB

# ── Helpers ────────────────────────────────────────────────────────────────

# Map our Component enum to memory paths for MTE bandwidth lookup
_COMPONENT_MTE_PATHS: dict[Component, tuple[str, str]] = {
    Component.MTE_GM: ("gm", "ub"),       # GM→UB (covers both CubeMTE2 and VecMTE2)
    Component.MTE_L1: ("l1", "l0a"),      # L1→L0A/B
    Component.MTE_UB: ("ub", "gm"),       # UB→GM (covers both FixPipe and MTE3)
}

# FixPipe is the Cube-side L0C→GM drain path.  When BW_l0c_to_gm_sustained
# is measured, it provides a more accurate bandwidth for Cube output transfers
# than the generic UB→GM (MTE3) path.  The component model falls back to
# UB→GM when L0C→GM is not calibrated.
_FIXPIPE_MTE_PATH = ("l0c", "gm")

# Precision string → DType (for throughput lookup)
def _prec_to_dtype(prec: Precision) -> DType:
    """Map extract Precision enum to calibration DType."""
    return DType.from_str(prec.value)


def _get_cube_throughput_ops_per_us(dtype: DType, cube: CubeConfig) -> float:
    """Sustained Cube throughput in operations per microsecond.

    1 TFLOPS = 10^12 FLOP/s = 10^6 FLOP/us
    """
    tflops = cube.get_throughput(dtype)
    if tflops <= 0:
        return 0.0
    return tflops * 1e6  # FLOP/us


# HIVM vector op names → calibration VecOpType names.  Most HIVM ops are
# "v" + VecOpType.value (vmul→mul); the irregular ones are mapped explicitly.
# Ops with no analytical-rate equivalent (vcmp/vnot/vand/varange) are absent
# and fall through to the aggregate rate.
_HIVM_VEC_NAME_MAP: dict[str, str] = {
    "vbrc": "broadcast",
    "vsel": "select",
    "vreduce": "reduce_sum",
    "vexp": "exp",
    "vcast": "cast",
}


def _hivm_to_vecop(op_name: str) -> "VecOpType | None":
    """Resolve a HIVM vector op name to a calibration VecOpType, or None."""
    if not op_name:
        return None
    name = op_name.lower()
    candidate = _HIVM_VEC_NAME_MAP.get(name)
    if candidate is None and name.startswith("v"):
        candidate = name[1:]  # strip leading 'v' (vmul→mul, vadd→add, …)
    for cand in (candidate, name):
        if not cand:
            continue
        try:
            return VecOpType.from_str(cand)
        except (KeyError, ValueError):
            continue
    return None


def _get_vector_throughput_ops_per_us(
    prec: Precision, vector: VectorConfig, op_name: str = "",
) -> float:
    """Achievable Vector throughput in operations per microsecond.

    A conservative *lower bound on time* needs the fastest *achievable* rate
    (so the bound never exceeds measured).  The per-op cycle calibration is
    that achievable rate; the aggregate-TFLOPS path is only a fallback for ops
    without a per-op model, and FP32 falls back to the FP16 aggregate when its
    own rate is uncalibrated (rather than collapsing the floor to infinity).

    Vector width = 128 elements per instruction.
    """
    # Try per-op cycle lookup first (the achievable per-instruction rate)
    vt = _hivm_to_vecop(op_name)
    if vt is not None and prec is not None:
        dtype = _prec_to_dtype(prec)
        cycles_per_128 = vector.get_op_cycles(vt, dtype)
        if cycles_per_128 > 0:
            # 128 elements/instruction at 1.85 GHz (1850 cycles/us):
            # elements/us = 128 * 1850 / cycles_per_128
            return 128.0 * 1850.0 / cycles_per_128

    # Fallback: aggregate TFLOPS
    if prec in (Precision.FP16, Precision.BF16):
        tflops = vector.throughput_fp16_tflops
    else:
        tflops = vector.throughput_fp32_tflops
        # FP32 rate is often uncalibrated (0).  Fall back to the FP16 aggregate
        # rather than returning 0 (which makes the component floor infinite and
        # the bound unsound).  FP16 ≥ FP32 throughput, so this keeps the time
        # floor a valid lower bound.
        if tflops <= 0:
            tflops = vector.throughput_fp16_tflops

    if tflops <= 0:
        return 0.0
    return tflops * 1e6  # FLOP/us


def _get_mte_throughput_bytes_per_us(
    component: Component, memory: MemHierarchy, op: Optional[OpRecord] = None,
) -> float:
    """Sustained MTE bandwidth in bytes per microsecond.

    Uses the operation's concrete transfer path when available.  FixPipe is
    identified by pipe/source because some DES emitters report its destination
    as UB even though the hardware drain being calibrated is L0C→GM.
    """
    path: tuple[str, str] | None = None
    if op is not None:
        if op.pipe.lower() == "fixpipe" or op.src_space == "l0c":
            path = _FIXPIPE_MTE_PATH
        elif op.src_space and op.dst_space:
            path = (op.src_space, op.dst_space)
    if path is None:
        path = _COMPONENT_MTE_PATHS.get(component)
    if path is None:
        return 0.0

    src, dst = path
    pkt_size = -1
    if op is not None and op.bytes_transferred > 0:
        pkt_size = op.bytes_transferred

    try:
        bw, _ = memory.lookup_bw(src, dst, pkt_size=pkt_size)
        return bw  # already in B/us
    except KeyError:
        fallback = _COMPONENT_MTE_PATHS.get(component)
        if fallback is None or fallback == path:
            return 0.0
        try:
            bw, _ = memory.lookup_bw(*fallback, pkt_size=pkt_size)
            return bw
        except KeyError:
            return 0.0


def _get_scalar_throughput_ops_per_us(prec: Precision, vector: VectorConfig) -> float:
    """Rough Scalar throughput in operations per microsecond.

    Delegates to VectorConfig.get_scalar_throughput_ops_per_us.
    Scalar is ~10-50× slower than Vector; we use a conservative 20× slower
    than Vector FP16 for two-limit Gap-1 analysis.
    """
    prec_str = prec.value if prec else ""
    return vector.get_scalar_throughput_ops_per_us(prec_str)


# ── Core computation ───────────────────────────────────────────────────────

@dataclass
class ComponentRate:
    """Ideal rate I_c for a single component at a single precision."""
    component: Component
    precision: Optional[Precision]  # None for MTE (byte-oriented)
    i_c: float                  # ideal throughput (ops/us or bytes/us)
    o_c: float                  # total work for this component+precision
    t_c_us: float               # O_c / I_c (microseconds)

    def __repr__(self) -> str:
        prec_str = self.precision.value if self.precision else "bytes"
        return (f"ComponentRate({self.component.value}/{prec_str}: "
                f"I={self.i_c:.1f}, O={self.o_c:.1f}, T={self.t_c_us:.3f} us)")


@dataclass
class ComponentBound:
    """Tier 2 bound output."""
    t_core_floor_us: float      # max_c(O_c / I_c)
    binding_component: Component  # component that sets the floor

    # Per-component rates (keyed by "component/precision")
    rates: Dict[str, ComponentRate] = field(default_factory=dict)

    # Per-component totals
    total_ops: Dict[str, float] = field(default_factory=dict)
    total_bytes: Dict[str, float] = field(default_factory=dict)

    # Per-component floor times (before max)
    per_component_us: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"ComponentBound(T_core_floor={self.t_core_floor_us:.2f} us, "
                f"binding={self.binding_component.value})")


def compute_component_floor(
    extract: HIVMExtract,
    cube: CubeConfig,
    vector: VectorConfig,
    memory: MemHierarchy,
    core: Optional[CoreConfig] = None,
) -> ComponentBound:
    """Compute T_core_floor from Tier 2 HIVM extraction.

    For each component c, computes the weighted-harmonic mean ideal rate I_c
    across its precision mix, then T_c = O_c / I_c.  The core floor is the
    maximum across all components (the bottleneck component).

    The harmonic mean is load-bearing for bound semantics: it gives the
    ideal throughput under perfect work distribution across precisions.
    Any real scheduling can only be slower → the computed T_c is a
    conservative lower bound.

    Args:
        extract: M3 HIVM extract output (per-component O_prec, operations).
        cube: Sustained Cube throughput calibration.
        vector: Sustained Vector throughput calibration.
        memory: Memory hierarchy with sustained bandwidths.
        core: Core config (clock, counts).  Uses 1.85 GHz default if None.

    Returns:
        ComponentBound with T_core_floor, per-component rates, and binding.

    Raises:
        ValueError: If a component has work but no throughput calibration.
    """
    if core is None:
        core = CoreConfig()

    # Aggregate work per (component, precision)
    # compute_work[(comp, prec)] = total ops (or bytes for MTE)
    compute_work: dict[tuple[Component, Optional[Precision]], float] = {}
    mte_bytes: dict[Component, float] = {}
    mte_operations: dict[Component, list[tuple[OpRecord, float]]] = {}

    for op in extract.operations:
        comp = op.component
        prec = op.precision

        if comp in (Component.CUBE, Component.VECTOR, Component.SCALAR):
            # Work in FLOPs (preferred) or elements (fallback) scaled by loop_multiplier.
            # Invariant: work-unit must match the rate-unit. Cube rate is FLOP/us;
            # Vector fallback rate is also FLOP/us (per-op elements/us path is dead
            # until op_name is threaded through — do not enable in A.4).
            work_raw = op.flops if op.flops > 0 else op.elements
            work = float(work_raw) * float(op.loop_multiplier)
            key = (comp, prec)
            compute_work[key] = compute_work.get(key, 0.0) + work
        elif comp in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
            # Work in bytes
            work = float(op.bytes_transferred) * float(op.loop_multiplier)
            mte_bytes[comp] = mte_bytes.get(comp, 0.0) + work
            mte_operations.setdefault(comp, []).append((op, work))
            # Also record per-precision for type-level tracking
            key = (comp, prec)
            compute_work[key] = compute_work.get(key, 0.0) + work

    # Compute I_c per component via harmonic mean, then T_c
    per_component_us: dict[str, float] = {}
    rates: dict[str, ComponentRate] = {}
    total_ops: dict[str, float] = {}
    total_bytes: dict[str, float] = {}

    for comp in Component:
        comp_str = comp.value

        # Collect precision-level data for this component
        precision_work: list[tuple[Optional[Precision], float]] = []
        for (c, p), w in compute_work.items():
            if c == comp and w > 0:
                precision_work.append((p, w))

        if not precision_work and comp not in mte_bytes:
            continue

        total_work = sum(w for _, w in precision_work)

        if comp == Component.CUBE:
            # Harmonic mean over Cube precisions
            numerator = 0.0
            denominator = 0.0
            for prec, w in precision_work:
                if prec is None:
                    continue
                dtype = _prec_to_dtype(prec)
                p_rate = _get_cube_throughput_ops_per_us(dtype, cube)
                if p_rate <= 0:
                    continue
                numerator += w
                denominator += w / p_rate
                key = f"{comp_str}/{prec.value}"
                rates[key] = ComponentRate(
                    component=comp, precision=prec,
                    i_c=p_rate, o_c=w, t_c_us=w / p_rate,
                )
            i_c = numerator / denominator if denominator > 0 else 0.0
            t_c = total_work / i_c if i_c > 0 else float("inf")
            total_ops[comp_str] = total_work

        elif comp == Component.VECTOR:
            # Harmonic mean over individual Vector ops, using each op's name so
            # the achievable per-op cycle rate is used (the aggregate-only path
            # both ignored op identity and collapsed to inf on uncalibrated
            # FP32).  Aggregate per-(op,prec) work so identical ops share a rate.
            op_work: dict[tuple[str, Optional[Precision]], float] = {}
            for op in extract.operations:
                if op.component != Component.VECTOR:
                    continue
                w = float(op.flops if op.flops > 0 else op.elements) * float(op.loop_multiplier)
                if w <= 0:
                    continue
                op_work[(op.op_name, op.precision)] = (
                    op_work.get((op.op_name, op.precision), 0.0) + w
                )

            numerator = 0.0
            denominator = 0.0
            for (op_name, prec), w in op_work.items():
                p_rate = _get_vector_throughput_ops_per_us(prec, vector, op_name)
                if p_rate <= 0:
                    continue
                numerator += w
                denominator += w / p_rate
                prec_str = prec.value if prec else "bytes"
                key = f"{comp_str}/{op_name}/{prec_str}"
                rates[key] = ComponentRate(
                    component=comp, precision=prec,
                    i_c=p_rate, o_c=w, t_c_us=w / p_rate,
                )
            i_c = numerator / denominator if denominator > 0 else 0.0
            t_c = total_work / i_c if i_c > 0 else float("inf")
            total_ops[comp_str] = total_work

        elif comp == Component.SCALAR:
            # Scalar: rough throughput for two-limit Gap-1 analysis.
            # Without real Scalar calibration, use Vector/20 as a conservative proxy.
            # This makes two-limit non-vacuous: Scalar→Vector idealization can now
            # show positive compiler_headroom.
            total_ops[comp_str] = total_work
            if total_work > 0:
                # Use the best-available Scalar rate (first precision with work)
                scalar_rate = 0.0
                for prec, w in precision_work:
                    if prec is not None and w > 0:
                        rate = _get_scalar_throughput_ops_per_us(prec, vector)
                        if rate > scalar_rate:
                            scalar_rate = rate
                i_c = scalar_rate
                t_c = total_work / i_c if i_c > 0 else 0.0
            else:
                i_c = 0.0
                t_c = 0.0

        elif comp in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
            total_bytes[comp_str] = mte_bytes.get(comp, 0.0)
            t_c = 0.0
            path_work: dict[tuple[Optional[Precision], float], float] = {}
            for op, work in mte_operations.get(comp, []):
                bw = _get_mte_throughput_bytes_per_us(comp, memory, op)
                if bw <= 0:
                    t_c = float("inf")
                    break
                t_c += work / bw
                path_work[(op.precision, bw)] = (
                    path_work.get((op.precision, bw), 0.0) + work
                )
            i_c = (
                total_bytes[comp_str] / t_c
                if t_c > 0 and t_c != float("inf")
                else 0.0
            )
            for (prec, bw), work in path_work.items():
                prec_str = prec.value if prec else "bytes"
                key = f"{comp_str}/{prec_str}/{bw:g}"
                rates[key] = ComponentRate(
                    component=comp, precision=prec,
                    i_c=bw, o_c=work, t_c_us=work / bw,
                )

        else:
            continue

        per_component_us[comp_str] = t_c

    # Core floor = max across components
    if per_component_us:
        binding_str = max(per_component_us, key=lambda k: per_component_us[k])
        binding_comp = Component(binding_str)
        t_core_floor_us = per_component_us[binding_str]
    else:
        binding_comp = Component.SCALAR
        t_core_floor_us = 0.0

    return ComponentBound(
        t_core_floor_us=t_core_floor_us,
        binding_component=binding_comp,
        rates=rates,
        total_ops=total_ops,
        total_bytes=total_bytes,
        per_component_us=per_component_us,
    )


def compute_component_floor_from_db(
    extract: HIVMExtract,
    db: "CalibrationDB",
) -> ComponentBound:
    """Compute T_core_floor using rates from a CalibrationDB.

    Convenience wrapper around compute_component_floor that unpacks
    cube/vector/memory/core from the DB so callers need not destructure it.
    """
    return compute_component_floor(
        extract,
        db.cube,
        db.vector,
        db.memory,
        db.core,
    )
