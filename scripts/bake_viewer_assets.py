#!/usr/bin/env python3
"""Pre-bake all D4RT viewer assets into a self-contained ``viewer_results/`` dir.

Every bit of view-time CPU work is moved AHEAD of time here: GLB meshing for demo
packages, and PnP/Sim3 + plot + ``.rrd`` for COLMAP trajectory comparisons. The
result is a portable directory of static files plus ``viewer_index.json``.
``scripts/demo_gradio.py --prebaked <dir>`` then displays them with zero compute.

Example::

    uv run --extra vis python scripts/bake_viewer_assets.py \\
        --results-root tmp/ --out viewer_results \\
        --trajectory run400_12f recon/run400/d4rt_pred_12.npz recon/run400/sparse_final_txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._gradio_helpers import (  # noqa: E402
    build_glb_from_demo_data,
    discover_demo_packages,
    meta_table_rows,
)
from vis.rerun_visualize import save_trajectory_comparison_to_rrd  # noqa: E402

VIEWER_INDEX = "viewer_index.json"


def bake_demo_package(
    pkg_dir: str | Path,
    out_root: str | Path,
    *,
    show_dynamic: bool = True,
    max_points: int = 200_000,
) -> dict[str, Any]:
    """Bake one demo package: GLB + copied video/poster + meta rows."""
    pkg_dir = Path(pkg_dir)
    data = json.loads((pkg_dir / "assets" / "demo_data.json").read_text())
    manifest = json.loads((pkg_dir / "manifest.json").read_text())
    name = pkg_dir.name
    dst = Path(out_root) / name
    dst.mkdir(parents=True, exist_ok=True)

    build_glb_from_demo_data(
        data, dst / "scene.glb", show_dynamic=show_dynamic, max_points=max_points
    )
    entry: dict[str, Any] = {
        "name": name,
        "kind": "demo_package",
        "glb": f"{name}/scene.glb",
        "meta_rows": meta_table_rows(data["meta"]),
        "video": None,
        "poster": None,
    }
    for field, manifest_key, fname in (
        ("video", "video_copy", "input_video.mp4"),
        ("poster", "video_poster", "video_poster.jpg"),
    ):
        src = pkg_dir / manifest.get(manifest_key, f"assets/{fname}")
        if src.exists():
            shutil.copy(src, dst / fname)
            entry[field] = f"{name}/{fname}"
    return entry


def bake_trajectory(
    name: str,
    npz: str | Path,
    colmap: str | Path,
    out_root: str | Path,
    *,
    use_ransac: bool = True,
) -> dict[str, Any]:
    """Bake one trajectory comparison: report JSON + plot PNG + ``.rrd``."""
    from scripts.check_colmap_trajectory_consistency import (  # noqa: PLC0415
        _public_report,
        _write_plot,
        compute_consistency,
        load_prediction,
        read_colmap_model,
    )

    dst = Path(out_root) / name
    dst.mkdir(parents=True, exist_ok=True)
    report = compute_consistency(
        load_prediction(Path(npz)),
        read_colmap_model(Path(colmap)),
        use_ransac=use_ransac,
    )
    public = _public_report(report)
    (dst / "report.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_plot(report, dst / "traj.png")
    save_trajectory_comparison_to_rrd(
        Path(npz), Path(colmap), dst / "traj.rrd", use_ransac=use_ransac
    )
    rows = [
        [k, json.dumps(v) if isinstance(v, dict) else str(v)] for k, v in public.items()
    ]
    return {
        "name": name,
        "kind": "trajectory",
        "metrics_rows": rows,
        "report": f"{name}/report.json",
        "plot": f"{name}/traj.png",
        "rrd": f"{name}/traj.rrd",
    }


def bake_all(
    results_root: str | Path,
    out_root: str | Path,
    *,
    trajectories: tuple[tuple[str, str, str], ...] = (),
    show_dynamic: bool = True,
    max_points: int = 200_000,
) -> Path:
    """Bake every demo package under ``results_root`` plus the given trajectories."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    packages = [
        bake_demo_package(
            pkg, out_root, show_dynamic=show_dynamic, max_points=max_points
        )
        for pkg in discover_demo_packages(results_root)
    ]
    trajs = [
        bake_trajectory(name, npz, colmap, out_root)
        for name, npz, colmap in trajectories
    ]
    index = {"demo_packages": packages, "trajectories": trajs}
    index_path = out_root / VIEWER_INDEX
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return index_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--results-root", required=True, help="Root dir to scan for demo packages."
    )
    parser.add_argument("--out", default="viewer_results", help="Output bake dir.")
    parser.add_argument(
        "--trajectory",
        nargs=3,
        action="append",
        metavar=("NAME", "NPZ", "COLMAP"),
        help="A trajectory comparison to bake (repeatable).",
    )
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument(
        "--no-highlight-dynamic",
        action="store_true",
        help="Do not recolor dynamic points.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index_path = bake_all(
        args.results_root,
        args.out,
        trajectories=tuple(tuple(t) for t in (args.trajectory or [])),
        show_dynamic=not args.no_highlight_dynamic,
        max_points=args.max_points,
    )
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
