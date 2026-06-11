"""Rerun visualization for D4RT inference artifacts.

Three artifact families produced elsewhere in this repo can be visualized:

* **demo package** — the directory written by ``scripts/build_demo_from_video.py``
  (``manifest.json`` + ``assets/demo_data.json`` + ``assets/input_video.mp4``).
  Logs the per-frame point cloud (static/dynamic split), the predicted tracks as
  3D polylines, the ref0 camera frustum, and the source video with the 2D point
  overlay.
* **static-tracks npz** — the dump written by
  ``scripts/dump_static_tracks_for_trajectory.py``. Logs the per-frame
  static/dynamic point clouds.
* **trajectory comparison** — a static-tracks npz plus a COLMAP model. Reuses
  ``scripts.check_colmap_trajectory_consistency`` to derive a PnP camera
  trajectory and Sim3-align it to COLMAP, then overlays both trajectories and
  prints the metrics.

Each artifact has a ``visualize_*`` entrypoint (spawns the local viewer) and a
``save_*_to_rrd`` entrypoint (writes a ``.rrd`` for headless / web inspection).
The ``.rrd`` form is the one to use on machines without a display.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]

DEFAULT_APP_ID = "open-d4rt"
DYNAMIC_COLOR = (255, 60, 60)
TRACK_COLOR = (255, 190, 0)
COLMAP_COLOR = (60, 130, 255)
D4RT_COLOR = (255, 140, 0)


# ---------------------------------------------------------------------------
# Blueprint + emit scaffolding (shared by every artifact).
# ---------------------------------------------------------------------------


def make_blueprint() -> rrb.Blueprint:
    """3D scene on the left, source image / metrics panels on the right."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="3D"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="frame/image", name="image"),
                rrb.TextDocumentView(origin="metrics", name="metrics"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )


def _emit(
    log_fn: Callable[[], None],
    *,
    app_id: str,
    rrd_path: Path | None,
    spawn: bool,
) -> None:
    """Run ``log_fn`` against a fresh recording, either spawned or saved."""
    rr.init(app_id, spawn=spawn)
    blueprint = make_blueprint()
    if rrd_path is not None:
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(rrd_path), default_blueprint=blueprint)
    else:
        rr.send_blueprint(blueprint)
    log_fn()
    if rrd_path is not None:
        rr.disconnect()


def _subsample(points: FloatArray, colors: NDArray[Any], max_points: int) -> tuple:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[idx], colors[idx]


def _log_point_cloud(
    entity: str,
    xyz: FloatArray,
    colors: NDArray[Any],
    *,
    is_dynamic: NDArray[Any] | None,
    max_points: int,
    static: bool,
) -> None:
    """Log a finite-filtered point cloud, splitting static vs dynamic points."""
    finite = np.isfinite(xyz).all(axis=1)
    if is_dynamic is None:
        is_dynamic = np.zeros(len(xyz), dtype=bool)
    static_mask = finite & ~is_dynamic
    dynamic_mask = finite & is_dynamic
    s_pts, s_rgb = _subsample(xyz[static_mask], colors[static_mask], max_points)
    if len(s_pts):
        rr.log(entity, rr.Points3D(s_pts, colors=s_rgb, radii=0.0005), static=static)
    if dynamic_mask.any():
        d_pts = xyz[dynamic_mask]
        rr.log(
            f"{entity}_dynamic",
            rr.Points3D(d_pts, colors=DYNAMIC_COLOR, radii=0.0009),
            static=static,
        )


def _log_ref0_camera(k: FloatArray, width: int, height: int) -> None:
    rr.log("world/ref0_camera", rr.Transform3D(), static=True)
    rr.log(
        "world/ref0_camera",
        rr.Pinhole(
            image_from_camera=k, width=width, height=height, image_plane_distance=0.1
        ),
        static=True,
    )


# ---------------------------------------------------------------------------
# Artifact A: demo package directory.
# ---------------------------------------------------------------------------


def _read_demo_data(pkg_dir: Path) -> dict[str, Any]:
    data_json = pkg_dir / "assets" / "demo_data.json"
    if not data_json.is_file():
        raise FileNotFoundError(f"demo_data.json not found under {pkg_dir}")
    return json.loads(data_json.read_text(encoding="utf-8"))


def _video_frames(pkg_dir: Path, count: int) -> list[NDArray[np.uint8]]:
    video_path = pkg_dir / "assets" / "input_video.mp4"
    if not video_path.is_file():
        return []
    cap = cv2.VideoCapture(str(video_path))
    frames: list[NDArray[np.uint8]] = []
    try:
        while len(frames) < count:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8))
    finally:
        cap.release()
    return frames


def _log_demo_package(pkg_dir: Path, max_points: int) -> None:
    data = _read_demo_data(pkg_dir)
    meta = data["meta"]
    points = data["points"]
    tracks = data["tracks"]

    num_frames = int(meta["numFrames"])
    point_xyz = np.asarray(points["xyzRef0"], dtype=np.float64)  # [F, P, 3]
    point_rgb = np.asarray(points["rgb"], dtype=np.uint8)  # [F, P, 3]
    point_vis = np.asarray(points["visibility"], dtype=bool)  # [F, P]
    point_uv = np.asarray(points["uvPx"], dtype=np.float64)  # [F, P, 2]
    is_dynamic = np.asarray(points["isDynamic"], dtype=bool)  # [P]

    _log_ref0_camera(
        np.asarray(meta["ref0K"], dtype=np.float64),
        int(meta["videoWidth"]),
        int(meta["videoHeight"]),
    )

    track_xyz = np.asarray(tracks["xyzRef0"], dtype=np.float64)  # [T, F, 3]
    strips = [t[np.isfinite(t).all(axis=1)] for t in track_xyz]
    strips = [s for s in strips if len(s) >= 2]
    if strips:
        rr.log("world/tracks", rr.LineStrips3D(strips, colors=TRACK_COLOR), static=True)

    video = _video_frames(pkg_dir, num_frames)
    for f in range(num_frames):
        rr.set_time("frame", sequence=f)
        vis = point_vis[f]
        _log_point_cloud(
            "world/points",
            point_xyz[f][vis],
            point_rgb[f][vis],
            is_dynamic=is_dynamic[vis],
            max_points=max_points,
            static=False,
        )
        if f < len(video):
            rr.log("frame/image", rr.Image(video[f]))
            rr.log(
                "frame/image/points",
                rr.Points2D(point_uv[f][vis], colors=point_rgb[f][vis], radii=1.5),
            )


def visualize_demo_package(
    pkg_dir: str | Path, *, max_points: int = 500_000, app_id: str = DEFAULT_APP_ID
) -> None:
    """Stream a demo package to a locally-spawned rerun viewer."""
    pkg_dir = Path(pkg_dir)
    _read_demo_data(pkg_dir)
    _emit(
        lambda: _log_demo_package(pkg_dir, max_points),
        app_id=app_id,
        rrd_path=None,
        spawn=True,
    )


def save_demo_package_to_rrd(
    pkg_dir: str | Path,
    rrd_path: str | Path,
    *,
    max_points: int = 500_000,
    app_id: str = DEFAULT_APP_ID,
) -> Path:
    """Write a demo package visualization to an ``.rrd`` file (headless)."""
    pkg_dir = Path(pkg_dir)
    rrd_path = Path(rrd_path)
    _read_demo_data(pkg_dir)  # fail fast on a missing/invalid package
    _emit(
        lambda: _log_demo_package(pkg_dir, max_points),
        app_id=app_id,
        rrd_path=rrd_path,
        spawn=False,
    )
    return rrd_path


# ---------------------------------------------------------------------------
# Artifact B: static-tracks npz.
# ---------------------------------------------------------------------------


def _log_static_tracks(npz_path: Path, max_points: int) -> None:
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    xyz = np.asarray(data["point_xyz_ref0"], dtype=np.float64)  # [F, P, 3]
    uv = np.asarray(data["point_uv_px"], dtype=np.float64)  # [F, P, 2]
    is_dynamic = np.asarray(data["point_is_dynamic"], dtype=bool)  # [P]
    # point_visibility is optional (matches check_colmap_trajectory_consistency).
    if "point_visibility" in data:
        vis = np.asarray(data["point_visibility"], dtype=bool)  # [F, P]
    else:
        vis = np.ones(xyz.shape[:2], dtype=bool)
    k = np.asarray(data["ref0_K"], dtype=np.float64)
    grey = np.full((xyz.shape[1], 3), 180, dtype=np.uint8)

    width = int(np.nanmax(uv[..., 0])) + 1 if np.isfinite(uv).any() else 1
    height = int(np.nanmax(uv[..., 1])) + 1 if np.isfinite(uv).any() else 1
    _log_ref0_camera(k, width, height)

    for f in range(xyz.shape[0]):
        rr.set_time("frame", sequence=f)
        mask = vis[f]
        _log_point_cloud(
            "world/points",
            xyz[f][mask],
            grey[mask],
            is_dynamic=is_dynamic[mask],
            max_points=max_points,
            static=False,
        )


def visualize_static_tracks(
    npz_path: str | Path, *, max_points: int = 500_000, app_id: str = DEFAULT_APP_ID
) -> None:
    """Stream a static-tracks npz to a locally-spawned rerun viewer."""
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    _emit(
        lambda: _log_static_tracks(npz_path, max_points),
        app_id=app_id,
        rrd_path=None,
        spawn=True,
    )


def save_static_tracks_to_rrd(
    npz_path: str | Path,
    rrd_path: str | Path,
    *,
    max_points: int = 500_000,
    app_id: str = DEFAULT_APP_ID,
) -> Path:
    """Write a static-tracks npz visualization to an ``.rrd`` file (headless)."""
    npz_path = Path(npz_path)
    rrd_path = Path(rrd_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    _emit(
        lambda: _log_static_tracks(npz_path, max_points),
        app_id=app_id,
        rrd_path=rrd_path,
        spawn=False,
    )
    return rrd_path


# ---------------------------------------------------------------------------
# Artifact C: trajectory comparison (npz + COLMAP model).
# ---------------------------------------------------------------------------


def _log_trajectory_comparison(
    npz_path: Path, colmap_model: Path, use_ransac: bool
) -> None:
    # Imported lazily so the lib is usable without the checker's deps loaded.
    from scripts.check_colmap_trajectory_consistency import (  # noqa: PLC0415
        compute_consistency,
        load_prediction,
        read_colmap_model,
    )

    pred = load_prediction(npz_path)
    colmap = read_colmap_model(colmap_model)
    report = compute_consistency(pred, colmap, use_ransac=use_ransac)

    aligned = np.asarray(report["_aligned_pred_centers"], dtype=np.float64)
    gt = np.asarray(report["_gt_centers"], dtype=np.float64)
    rr.log(
        "world/colmap_trajectory",
        rr.LineStrips3D([gt], colors=COLMAP_COLOR),
        static=True,
    )
    rr.log(
        "world/colmap_trajectory/points",
        rr.Points3D(gt, colors=COLMAP_COLOR, radii=0.01),
        static=True,
    )
    rr.log(
        "world/d4rt_trajectory",
        rr.LineStrips3D([aligned], colors=D4RT_COLOR),
        static=True,
    )
    rr.log(
        "world/d4rt_trajectory/points",
        rr.Points3D(aligned, colors=D4RT_COLOR, radii=0.01),
        static=True,
    )

    metrics = {k: v for k, v in report.items() if not k.startswith("_")}
    rr.log("metrics", rr.TextDocument(json.dumps(metrics, indent=2)), static=True)


def visualize_trajectory_comparison(
    npz_path: str | Path,
    colmap_model: str | Path,
    *,
    use_ransac: bool = True,
    app_id: str = DEFAULT_APP_ID,
) -> None:
    """Stream a D4RT-vs-COLMAP trajectory comparison to a spawned viewer."""
    npz_path = Path(npz_path)
    colmap_model = Path(colmap_model)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not colmap_model.exists():
        raise FileNotFoundError(colmap_model)
    _emit(
        lambda: _log_trajectory_comparison(npz_path, colmap_model, use_ransac),
        app_id=app_id,
        rrd_path=None,
        spawn=True,
    )


def save_trajectory_comparison_to_rrd(
    npz_path: str | Path,
    colmap_model: str | Path,
    rrd_path: str | Path,
    *,
    use_ransac: bool = True,
    app_id: str = DEFAULT_APP_ID,
) -> Path:
    """Write a trajectory comparison visualization to an ``.rrd`` file."""
    npz_path = Path(npz_path)
    colmap_model = Path(colmap_model)
    rrd_path = Path(rrd_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not colmap_model.exists():
        raise FileNotFoundError(colmap_model)
    _emit(
        lambda: _log_trajectory_comparison(npz_path, colmap_model, use_ransac),
        app_id=app_id,
        rrd_path=rrd_path,
        spawn=False,
    )
    return rrd_path
