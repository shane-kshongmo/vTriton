#!/usr/bin/env python3
"""measure_hbm_bw.py — M-1/M-2 HBM bandwidth + contention calibration on 910B3.

Runs the optimal contiguous stream copy (`test/stream_copy_bench.py`) under
synchronize()-bracketed device timing and emits:

* **M-1** true peak HBM bandwidth: a TILE/BLOCK sweep at a saturating grid; the
  max aggregate GB/s is the peak.
* **M-2** contention curve: a program-count sweep at fixed large BLOCK/TILE;
  per-core BW(n) = aggregate(n) / min(n, n_cores).

Output JSON feeds M-3 (pkt_efficiency re-anchor) and the occupancy-aware MTE
bandwidth model.  Run on the remote (under triton_hxl):

    python scripts/measure_hbm_bw.py --out /tmp/hbm_bw.json --n-cores 20
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_bench(path: str):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[p.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def run_m1(bench, grid: int, blocks, tiles, warmup, iters) -> dict:
    """Peak sweep: at a saturating grid, sweep BLOCK x TILE; take the max."""
    runs = []
    best = None
    for block in blocks:
        for tile in tiles:
            if block % tile != 0:
                continue
            r = bench.run_config(grid, block, tile, warmup=warmup, iters=iters)
            runs.append(r)
            print(
                f"[M-1] grid={grid} block={block} tile={tile} "
                f"pkt={r['packet_bytes']}B -> {r['agg_gbps']:.1f} GB/s "
                f"({r['median_us']:.1f} us)",
                file=sys.stderr,
            )
            if best is None or r["agg_gbps"] > best["agg_gbps"]:
                best = r
    return {"grid": grid, "runs": runs, "peak": best}


def run_m2(bench, block, tile, ncounts, n_cores, warmup, iters) -> dict:
    """Contention sweep: vary program count; per-core BW = agg / min(n, n_cores)."""
    runs = []
    for n in ncounts:
        r = bench.run_config(n, block, tile, warmup=warmup, iters=iters)
        active = min(n, n_cores)
        r["active_cores"] = active
        r["per_core_gbps"] = r["agg_gbps"] / active
        runs.append(r)
        print(
            f"[M-2] nprog={n} active={active} -> agg={r['agg_gbps']:.1f} GB/s "
            f"per_core={r['per_core_gbps']:.2f} GB/s ({r['median_us']:.1f} us)",
            file=sys.stderr,
        )
    return {"block": block, "tile": tile, "n_cores": n_cores, "runs": runs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bench",
        default=str(Path(__file__).parent.parent / "test" / "stream_copy_bench.py"),
    )
    ap.add_argument("--out", default="/tmp/hbm_bw.json")
    ap.add_argument("--n-cores", type=int, default=40,
                    help="physical core count for per-core normalization (M-2); "
                         "910B3 has 40 AIV (vector/MTE) cores, 20 AIC (cube)")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--m1-grid", type=int, default=48)
    args = ap.parse_args()

    bench = load_bench(args.bench)

    # M-1: peak. Large per-program slab so DMA dominates launch overhead.
    # tiles span 2KB..64KB packets (UB-bounded); blocks 256KB..1MB per program.
    m1 = run_m1(
        bench,
        grid=args.m1_grid,
        blocks=[262144, 524288, 1048576],
        tiles=[512, 2048, 8192, 16384],
        warmup=args.warmup,
        iters=args.iters,
    )

    # M-2: contention. Fixed large slab + large tile (peak packet), sweep cores.
    m2 = run_m2(
        bench,
        block=524288,
        tile=8192,
        ncounts=[1, 2, 4, 8, 16, 20, 24, 32, 40, 48, 64],
        n_cores=args.n_cores,
        warmup=args.warmup,
        iters=args.iters,
    )

    # Sanity verify one config end-to-end (correctness of the copy).
    verify = bench.run_config(args.m1_grid, 262144, 8192, warmup=2, iters=2, verify=True)

    result = {
        "host": "910B3",
        "n_cores": args.n_cores,
        "warmup": args.warmup,
        "iters": args.iters,
        "verify_copy_correct": verify["verified"],
        "M1_peak": m1,
        "M2_contention": m2,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))

    peak = m1["peak"]
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"M-1 PEAK aggregate BW: {peak['agg_gbps']:.1f} GB/s "
          f"(block={peak['block']} tile={peak['tile']} "
          f"pkt={peak['packet_bytes']}B)", file=sys.stderr)
    sc = m2["runs"][0]
    print(f"M-2 single-core BW (n=1): {sc['agg_gbps']:.1f} GB/s", file=sys.stderr)
    print(f"copy verified: {verify['verified']}", file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
