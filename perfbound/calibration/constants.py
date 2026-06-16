# M1 — Calibration Constants Dataclasses
#
# Ported from tilesim core/config/arc_spec.py pattern, adapted for the
# two-tier analytical bound model.  Every sustained-rate constant carries
# provenance (value, ci, source, n_runs) — no datasheet peaks enter I_c.
#
# Source spec: .omc/specs/performance_bound_model.md §A.1
# Existing seed: configs/ascend_910b.json (calibration block)

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..extract.op_classifier import Precision

# ── Precision / DType ──────────────────────────────────────────────────────

class DType(Enum):
    """Data types supported by Ascend NPU compute and transfer units."""
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "f64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"

    @property
    def size(self) -> int:
        """Size in bytes."""
        _sizes = {
            DType.FP16: 2, DType.BF16: 2,
            DType.FP32: 4, DType.FP64: 8,
            DType.INT8: 1, DType.INT16: 2,
            DType.INT32: 4, DType.INT64: 8,
        }
        return _sizes[self]

    @classmethod
    def from_str(cls, s: str) -> "DType":
        for dt in cls:
            if dt.value == s.lower():
                return dt
        raise KeyError(f"Unknown DType: {s}")


# ── Memory Locations ───────────────────────────────────────────────────────

class MemLoc(Enum):
    """Memory hierarchy levels on Ascend NPU."""
    GM = "gm"        # Global Memory (HBM)
    L2 = "l2"        # Shared L2 cache
    L1 = "l1"        # Cube data staging buffer (per-AIC)
    L0A = "l0a"      # Left matrix input register file
    L0B = "l0b"      # Right matrix input register file
    L0C = "l0c"      # Cube output accumulator
    UB = "ub"        # Unified Buffer (Vector compute space)

    @classmethod
    def from_str(cls, s: str) -> "MemLoc":
        for ml in cls:
            if ml.value == s.lower():
                return ml
        raise KeyError(f"Unknown MemLoc: {s}")


# ── Vector Op Types ────────────────────────────────────────────────────────

class VecOpType(Enum):
    """Vector operation types with distinct sustained throughput."""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MAX = "max"
    MIN = "min"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"
    NEG = "neg"
    ABS = "abs"
    CAST = "cast"
    REDUCE_SUM = "reduce_sum"
    REDUCE_MAX = "reduce_max"
    REDUCE_MIN = "reduce_min"
    REDUCE_PROD = "reduce_prod"
    BROADCAST = "broadcast"
    SELECT = "select"

    @classmethod
    def from_str(cls, s: str) -> "VecOpType":
        for vt in cls:
            if vt.value == s.lower():
                return vt
        raise KeyError(f"Unknown VecOpType: {s}")


# ── Calibration Constant (provenance-carrying atomic unit) ─────────────────

@dataclass
class CalibrationConstant:
    """A single measured hardware constant with full provenance.

    Every I_c term must reference one of these — datasheet peaks are
    inadmissible as I_c sources (§6 of the A.0 plan).
    """
    name: str                          # e.g. "P_cube_fp16_sustained"
    value: float                       # measured sustained value
    unit: str                          # e.g. "TFLOPS", "GB/s", "cycles"
    ci_95: float                       # 95% confidence interval half-width
    source: str                        # "cce_microbench", "msprof_profile", "datasheet_seed"
    n_runs: int                        # number of measurement runs (≥30 for P0)
    notes: str = ""

    @property
    def ci_rel(self) -> float:
        """Relative confidence interval (coefficient of variation proxy)."""
        return self.ci_95 / self.value if self.value != 0 else float("inf")

    @property
    def is_valid(self) -> bool:
        """P0 constants require ≥30 runs with <5% relative CI."""
        return self.n_runs >= 30 and self.ci_rel < 0.05 and self.source == "cce_microbench"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "ci_95": self.ci_95,
            "source": self.source,
            "n_runs": self.n_runs,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationConstant":
        return cls(**d)


# ── Core Configuration ─────────────────────────────────────────────────────

@dataclass
class CoreConfig:
    """Core topology for Ascend 910B3.

    AIC = AI Core (hosts Cube + CubeMTE2 + FixPipe)
    AIV = AI Vector (hosts Vector + VecMTE2 + MTE3)
    cv_bind: if True, 1 AIC drags 2 AIV in a fixed group.
    """
    aic_core_num: int = 20
    aiv_core_num: int = 40
    cv_bind: bool = True
    clock_freq_ghz: float = 1.85

    @property
    def clock_freq_mhz(self) -> float:
        return self.clock_freq_ghz * 1000.0

    @property
    def cycles_per_us(self) -> float:
        return self.clock_freq_ghz * 1000.0

    @property
    def n_cores_cube(self) -> int:
        """Cores available for Cube-bearing kernels (each AIC drags 2 AIV)."""
        return self.aic_core_num  # 20

    @property
    def n_cores_vector_only(self) -> int:
        """Cores available for Vector-only kernels (AIV can run independently)."""
        return self.aiv_core_num  # 40

    def cycles_to_us(self, cycles: float) -> float:
        return cycles / self.cycles_per_us


# ── Cube Configuration ─────────────────────────────────────────────────────

@dataclass
class CubeConfig:
    """Sustained Cube (matrix engine) throughput per data type.

    All values are measured sustained rates, NOT datasheet peaks.
    The fractal_sizes dict records the hardware M×K×N tile dimensions.
    """
    throughput: dict[DType, float] = field(default_factory=dict)
    # TFLOPS sustained per dtype, e.g. {DType.FP16: 280.0, DType.INT8: 560.0}
    fractal_sizes: dict[DType, tuple[int, int, int]] = field(default_factory=dict)
    # e.g. {DType.FP16: (16,16,16), DType.INT8: (16,32,16)}
    repeat_cycles: dict[DType, float] = field(default_factory=dict)
    # cycles per repeat per dtype (≈ 1.0 for a bound, higher for estimate)

    def get_throughput(self, dtype: DType) -> float:
        """Sustained TFLOPS for dtype."""
        return self.throughput.get(dtype, 0.0)

    def get_tile_ops(self, dtype: DType) -> int:
        """FLOPs per tile: 2 * M * K * N."""
        m, k, n = self.fractal_sizes.get(dtype, (16, 16, 16))
        return 2 * m * k * n


# ── Vector Configuration ───────────────────────────────────────────────────

@dataclass
class VectorConfig:
    """Sustained Vector (SIMD engine) throughput.

    vec_width_elements: SIMD width (128 elements for 910B)
    Per-op cycle costs from profiling calibration (seed from ascend_910b.json).
    """
    vec_width_elements: int = 128
    vec_width_bytes: int = 256
    throughput_fp16_tflops: float = 0.0  # sustained (measure)
    throughput_fp32_tflops: float = 0.0  # sustained (measure)

    # Per-operation cycles per vector instruction (128-element SIMD chunk)
    op_cycles: dict[tuple[VecOpType, DType], float] = field(default_factory=dict)

    # Scalar throughput.
    # NOTE: this is a DERIVED estimate (P_vector / SIMD_width), not a direct CCE
    # measurement — see RESULTS.md §8 (US-SB-007).  It is retained as evidence
    # but MUST NOT tighten the time bound.  Until scalar_throughput_measured is
    # True, get_scalar_throughput_ops_per_us() returns the full measured Vector
    # rate as an optimistic upper-rate fallback (can loosen, never illegitimately
    # tighten, the floor).
    scalar_throughput_fp16_tflops: float = 0.0  # derived estimate (evidence only)
    scalar_throughput_measured: bool = False    # True only after a direct CCE measurement

    def get_op_cycles(self, op: VecOpType, dtype: DType) -> float:
        """Cycles for one vector instruction of this op+dtype."""
        key = (op, dtype)
        if key in self.op_cycles:
            return self.op_cycles[key]
        # Fallback: same op with FP32
        fallback = (op, DType.FP32)
        if fallback in self.op_cycles:
            return self.op_cycles[fallback]
        return 1.0  # conservative default

    def get_scalar_throughput_ops_per_us(self, prec_str: str) -> float:
        """Scalar throughput in operations per microsecond, for the bound.

        SOUNDNESS RULE (US-SB-007, RESULTS.md §8): the stored
        scalar_throughput_fp16_tflops is a DERIVED estimate (P_vector / 128),
        not a direct CCE measurement, so it must NOT be used to tighten the
        time floor.  Unless scalar_throughput_measured is True, this returns the
        full measured Vector rate as an optimistic upper-rate fallback — this
        can only loosen the floor, never illegitimately tighten it.

        Args:
            prec_str: Precision string (e.g., "fp16", "bf16", "fp32")

        Returns 0.0 if Vector throughput is not calibrated.
        """
        prec_lower = prec_str.lower() if prec_str else ""
        if prec_lower in ("fp16", "bf16"):
            vec_tflops = self.throughput_fp16_tflops
        elif prec_lower == "fp32":
            vec_tflops = self.throughput_fp32_tflops
        else:
            return 0.0

        if vec_tflops <= 0:
            return 0.0

        # Only a *measured* scalar rate may tighten the bound.  Otherwise fall
        # back to the full Vector rate (optimistic upper bound on the rate ->
        # lower bound on time stays sound).
        if self.scalar_throughput_measured and self.scalar_throughput_fp16_tflops > 0:
            scalar_tflops = self.scalar_throughput_fp16_tflops
        else:
            scalar_tflops = vec_tflops

        return scalar_tflops * 1e6  # FLOP/us


# ── Memory Bandwidth ───────────────────────────────────────────────────────

@dataclass
class MemBandwidth:
    """Sustained bandwidth for a single memory transfer path.

    Core-num sensitivity: bandwidth may degrade with more cores under load.
    pkt_size sensitivity: small packets incur amortization overhead (Gap 2).
    """
    src_mem: str              # MemLoc value string
    dst_mem: str              # MemLoc value string
    bw_gb_per_s: float        # sustained bandwidth in GB/s
    core_num: int = -1        # -1 = core-count-independent
    pkt_size: int = -1        # -1 = size-independent (use for large xfers)
    alignment_bytes: int = 32 # hardware alignment requirement
    max_burst_bytes: int = 65536  # max single DMA burst

    @property
    def bw_bytes_per_us(self) -> float:
        """Bandwidth in bytes per microsecond (single core)."""
        return self.bw_gb_per_s * 1000.0  # GB/s → B/us  (1 GB/s = 10⁹ B/s = 1000 B/us)

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "MemBandwidth":
        """Parse a row from the bandwidth CSV (tilesim format)."""
        return cls(
            src_mem=row["src_mem"],
            dst_mem=row["dst_mem"],
            bw_gb_per_s=float(row["bandwidth(GB/s)"]),
            core_num=int(row.get("core_num", -1)),
            pkt_size=int(row.get("pkt_size", -1)),
        )


@dataclass
class MemHierarchy:
    """Memory hierarchy sizes and bandwidths (per-core where applicable)."""
    # Sizes in bytes
    gm_size_gb: float = 32.0
    l2_size_mb: float = 192.0
    l1_size_kb: float = 1024.0
    l0a_size_kb: float = 64.0
    l0b_size_kb: float = 64.0
    l0c_size_kb: float = 256.0
    ub_size_kb: float = 256.0

    # Sustained bandwidths indexed by (src, dst, core_num)
    bw: dict[tuple[str, str, int], MemBandwidth] = field(default_factory=dict)

    # Small-packet amortization parameters (pkt_param from tilesim 910B1 yaml)
    pkt_param: dict[str, tuple[float, float]] = field(default_factory=dict)
    # e.g. {"64B": (a, b), "128B": (a, b), "256B": (a, b)}

    def lookup_bw(self, src: str, dst: str, core_num: int = -1,
                  pkt_size: int = -1) -> tuple[float, bool]:
        """Look up sustained bandwidth in B/us for a transfer path.

        Returns (bw_bytes_per_us, is_small_pkt).
        """
        # Try exact match first
        key = (src, dst, core_num)
        if key in self.bw:
            bw = self.bw[key].bw_bytes_per_us
        else:
            # Fall back to core-independent
            default_key = (src, dst, -1)
            if default_key in self.bw:
                bw = self.bw[default_key].bw_bytes_per_us
            else:
                raise KeyError(f"No bandwidth for {src}→{dst} (core_num={core_num})")

        # Small-packet amortization
        is_small = False
        if self.pkt_param and pkt_size > 0:
            thresholds = sorted(
                [(int(k.rstrip("B")), v) for k, v in self.pkt_param.items()],
                key=lambda x: x[0]
            )
            for threshold_bytes, (a, b) in thresholds:
                if pkt_size < threshold_bytes:
                    pkt_size_gb = pkt_size / (1024 ** 3)
                    bw = b / (a * b + pkt_size_gb) * (1024 ** 3) / 1e6
                    is_small = True
                    break

        return bw, is_small


# ── Calibration Database (top-level container) ─────────────────────────────

@dataclass
class CalibrationDB:
    """Versioned calibration database for one Ascend 910B3 chip.

    All sustained-rate constants carry CalibrationConstant provenance.
    Loaded from perfbound/data/calib_910b3_vX.json.
    """
    version: str = "v1"
    hardware_name: str = "Ascend 910B3"
    description: str = ""

    core: CoreConfig = field(default_factory=CoreConfig)
    cube: CubeConfig = field(default_factory=CubeConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    memory: MemHierarchy = field(default_factory=MemHierarchy)

    # All individual calibration constants (flattened for audit)
    constants: dict[str, CalibrationConstant] = field(default_factory=dict)

    # Scalar overhead factor (loop control, pointer arithmetic, barriers)
    scalar_overhead_factor: float = 1.0

    # Startup latencies in cycles
    startup_latency: dict[str, float] = field(default_factory=dict)
    # e.g. {"vector": 35, "mte2": 50, "mte3": 40, "cube": 20, "fixpipe": 30}

    # Mandatory handoff cost (minimal L0C→GM→UB, measured)
    mandatory_handoff_cycles: float = 0.0

    # Pipe barrier cost per inner iteration (for attribution, not I_c)
    pipe_barrier_cycles_per_iter: float = 0.0

    def get_constant(self, name: str) -> Optional[CalibrationConstant]:
        return self.constants.get(name)

    def validate_p0_constants(self) -> list[str]:
        """Return names of P0 constants that fail provenance checks."""
        violations = []
        for name in required_p0_constant_names():
            c = self.constants.get(name)
            if c is None:
                violations.append(f"{name}: missing")
                continue
            if not c.is_valid:
                violations.append(
                    f"{name}: source={c.source}, n_runs={c.n_runs}, ci_rel={c.ci_rel:.3f}"
                )
        return violations

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "version": self.version,
            "hardware_name": self.hardware_name,
            "description": self.description,
            "core": {
                "aic_core_num": self.core.aic_core_num,
                "aiv_core_num": self.core.aiv_core_num,
                "cv_bind": self.core.cv_bind,
                "clock_freq_ghz": self.core.clock_freq_ghz,
            },
            "cube": {
                "throughput": {k.value: v for k, v in self.cube.throughput.items()},
                "fractal_sizes": {k.value: list(v) for k, v in self.cube.fractal_sizes.items()},
            },
            "vector": {
                "vec_width_elements": self.vector.vec_width_elements,
                "throughput_fp16_tflops": self.vector.throughput_fp16_tflops,
                "throughput_fp32_tflops": self.vector.throughput_fp32_tflops,
                "scalar_throughput_fp16_tflops": self.vector.scalar_throughput_fp16_tflops,
                "scalar_throughput_measured": self.vector.scalar_throughput_measured,
            },
            "memory": {
                "gm_size_gb": self.memory.gm_size_gb,
                "l2_size_mb": self.memory.l2_size_mb,
                "l1_size_kb": self.memory.l1_size_kb,
                "l0a_size_kb": self.memory.l0a_size_kb,
                "l0b_size_kb": self.memory.l0b_size_kb,
                "l0c_size_kb": self.memory.l0c_size_kb,
                "ub_size_kb": self.memory.ub_size_kb,
            },
            "constants": {k: v.to_dict() for k, v in self.constants.items()},
            "startup_latency": dict(self.startup_latency),
            "mandatory_handoff_cycles": self.mandatory_handoff_cycles,
            "pipe_barrier_cycles_per_iter": self.pipe_barrier_cycles_per_iter,
            "scalar_overhead_factor": self.scalar_overhead_factor,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationDB":
        """Deserialize from JSON dict."""
        db = cls(
            version=d["version"],
            hardware_name=d.get("hardware_name", "Ascend 910B3"),
            description=d.get("description", ""),
        )
        # Core
        core_d = d.get("core", {})
        db.core = CoreConfig(
            aic_core_num=core_d.get("aic_core_num", 20),
            aiv_core_num=core_d.get("aiv_core_num", 40),
            cv_bind=core_d.get("cv_bind", True),
            clock_freq_ghz=core_d.get("clock_freq_ghz", 1.85),
        )
        # Cube
        cube_d = d.get("cube", {})
        db.cube = CubeConfig(
            throughput={DType.from_str(k): v for k, v in cube_d.get("throughput", {}).items()},
            fractal_sizes={DType.from_str(k): tuple(v) for k, v in cube_d.get("fractal_sizes", {}).items()},
        )
        # Vector
        vec_d = d.get("vector", {})
        db.vector = VectorConfig(
            vec_width_elements=vec_d.get("vec_width_elements", 128),
            throughput_fp16_tflops=vec_d.get("throughput_fp16_tflops", 0.0),
            throughput_fp32_tflops=vec_d.get("throughput_fp32_tflops", 0.0),
            scalar_throughput_fp16_tflops=vec_d.get("scalar_throughput_fp16_tflops", 0.0),
            scalar_throughput_measured=vec_d.get("scalar_throughput_measured", False),
        )
        # Memory hierarchy
        mem_d = d.get("memory", {})
        memory = MemHierarchy(
            gm_size_gb=mem_d.get("gm_size_gb", 32.0),
            l2_size_mb=mem_d.get("l2_size_mb", 192.0),
            l1_size_kb=mem_d.get("l1_size_kb", 1024.0),
            l0a_size_kb=mem_d.get("l0a_size_kb", 64.0),
            l0b_size_kb=mem_d.get("l0b_size_kb", 64.0),
            l0c_size_kb=mem_d.get("l0c_size_kb", 256.0),
            ub_size_kb=mem_d.get("ub_size_kb", 256.0),
        )

        # Load bandwidth entries if present
        bw_d = mem_d.get("bw", {})
        if bw_d:
            for bw_name, bw_data in bw_d.items():
                if isinstance(bw_data, dict):
                    src = bw_data.get("src_mem", "")
                    dst = bw_data.get("dst_mem", "")
                    core_num = bw_data.get("core_num", -1)
                    key = (src, dst, core_num)
                    memory.bw[key] = MemBandwidth(
                        src_mem=src,
                        dst_mem=dst,
                        bw_gb_per_s=bw_data.get("bw_gb_per_s", 0.0),
                        core_num=core_num,
                        pkt_size=bw_data.get("pkt_size", -1),
                        alignment_bytes=bw_data.get("alignment_bytes", 32),
                        max_burst_bytes=bw_data.get("max_burst_bytes", 65536),
                    )

        db.memory = memory
        # Constants
        for name, cd in d.get("constants", {}).items():
            db.constants[name] = CalibrationConstant.from_dict(cd)
        # Other
        db.startup_latency = dict(d.get("startup_latency", {}))
        db.mandatory_handoff_cycles = d.get("mandatory_handoff_cycles", 0.0)
        db.pipe_barrier_cycles_per_iter = d.get("pipe_barrier_cycles_per_iter", 0.0)
        db.scalar_overhead_factor = d.get("scalar_overhead_factor", 1.0)
        return db

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationDB":
        """Load from a calib JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        """Save to a calib JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def required_p0_constant_names() -> list[str]:
    """Soundness-critical A.1 constants that must be measured before completion."""
    return [
        "P_cube_fp16_sustained",
        "P_cube_int8_sustained",
        "P_cube_bf16_sustained",
        "P_vector_add_sustained",
        "P_vector_mul_sustained",
        "P_vector_max_sustained",
        "P_vector_min_sustained",
        "P_vector_exp_sustained",
        "P_vector_log_sustained",
        "P_vector_sqrt_sustained",
        "P_vector_rsqrt_sustained",
        "BW_gm_to_ub_sustained",
        "BW_ub_to_gm_sustained",
        "BW_gm_to_l1_sustained",
        "BW_l1_to_l0a_sustained",
        "BW_l0c_to_gm_sustained",
        "BW_hbm_allcore_sustained",
        "mandatory_handoff_cost_l0c_to_gm_to_ub",
    ]
