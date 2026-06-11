#!/usr/bin/env python3
"""Check how consistent a D4RT-inferred camera trajectory is with COLMAP.

This OpenD4RT build does not predict camera extrinsics directly
(``vis.build_like_demo._export_demo_data`` returns ``pred_camera_* = None``).
Instead, the model yields per-frame 3D points in the frame-0 reference frame
(``point_xyz_ref0``) together with their 2D image tracks (``point_uv_px``) and a
per-point dynamic mask (``point_is_dynamic``). We *derive* a camera trajectory
from those 2D-3D correspondences via PnP (restricted to static points), align it
to the COLMAP trajectory with a similarity (Sim3) transform, and report standard
trajectory-consistency metrics (ATE / RPE / scale / rotation / intrinsics).

Inputs:
  --pred           A ``.npz`` prediction dump with keys:
                     point_xyz_ref0  float [F, P, 3]   (ref0-frame 3D points)
                     point_uv_px     float [F, P, 2]   (per-frame pixel tracks)
                     point_is_dynamic bool [P]         (True = moving point)
                     ref0_K          float [3, 3]      (D4RT intrinsics)
                     frame_names     str   [F]         (COLMAP image file names)
                   (Produce it with ``build_demo_from_video.py --dump-pred-npz``.)
  --colmap-model   A COLMAP model directory. Text models (cameras.txt +
                   images.txt) are read with no extra deps; binary models are
                   read via ``pycolmap`` if installed, otherwise export text with
                   ``colmap model_converter --output_type TXT``.

Output: a JSON report (``--output``) and an optional top-down trajectory plot
(``--plot``).
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


# === Data containers ===


@dataclass(frozen=True)
class Prediction:
    """D4RT static-point correspondences used to derive a camera trajectory."""

    xyz_ref0: FloatArray  # [P, 3] static-point 3D positions in the ref0 frame
    uv_per_frame: FloatArray  # [F, P, 2] static-point pixel tracks per frame
    visible: FloatArray  # [F, P] bool mask of usable point observations per frame
    K: FloatArray  # [3, 3] D4RT intrinsics
    frame_names: list[str]  # [F] COLMAP image file names


@dataclass(frozen=True)
class ColmapCamera:
    """A COLMAP image pose (world->camera) plus its intrinsics."""

    R_wc: FloatArray  # [3, 3] world->camera rotation
    t_wc: FloatArray  # [3] world->camera translation
    K: FloatArray  # [3, 3] intrinsics


# === COLMAP model reading ===


def _quat_wxyz_to_rotation(qw: float, qx: float, qy: float, qz: float) -> FloatArray:
    """Convert a COLMAP (w, x, y, z) quaternion to a rotation matrix."""
    n = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _intrinsics_from_params(model: str, params: list[float]) -> FloatArray:
    """Build a 3x3 K from COLMAP camera ``MODEL`` + ``PARAMS``."""
    if model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _iter_data_lines(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            yield line


def _parse_cameras_txt(path: Path) -> dict[int, FloatArray]:
    cameras: dict[int, FloatArray] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        cam_id, model = int(parts[0]), parts[1]
        params = [float(x) for x in parts[4:]]
        cameras[cam_id] = _intrinsics_from_params(model, params)
    return cameras


def _parse_images_txt(
    path: Path, cameras: dict[int, FloatArray]
) -> dict[str, ColmapCamera]:
    # COLMAP images.txt stores two lines per image: a pose line then a POINTS2D
    # line (which may be empty). Drop only comment lines so the pairing holds,
    # then read every even-indexed line as a pose line.
    data = [
        ln
        for ln in path.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    ]
    images: dict[str, ColmapCamera] = {}
    for pose_line in data[0::2]:
        parts = pose_line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = (float(parts[i]) for i in range(1, 5))
        tx, ty, tz = (float(parts[i]) for i in range(5, 8))
        cam_id = int(parts[8])
        name = parts[9]
        if cam_id not in cameras:
            continue
        images[name] = ColmapCamera(
            R_wc=_quat_wxyz_to_rotation(qw, qx, qy, qz),
            t_wc=np.array([tx, ty, tz], dtype=np.float64),
            K=cameras[cam_id],
        )
    return images


def _read_colmap_via_pycolmap(model_dir: Path) -> dict[str, ColmapCamera]:
    import pycolmap  # noqa: PLC0415 # ty: ignore[unresolved-import]

    rec = pycolmap.Reconstruction(str(model_dir))
    images: dict[str, ColmapCamera] = {}
    for image in rec.images.values():
        cam = rec.cameras[image.camera_id]
        k = cam.calibration_matrix()
        rigid = image.cam_from_world()
        images[image.name] = ColmapCamera(
            R_wc=np.asarray(rigid.rotation.matrix(), dtype=np.float64),
            t_wc=np.asarray(rigid.translation, dtype=np.float64),
            K=np.asarray(k, dtype=np.float64),
        )
    return images


def read_colmap_model(model_dir: Path) -> dict[str, ColmapCamera]:
    """Read a COLMAP model directory into ``{image_name: ColmapCamera}``.

    Prefers the dependency-free text model; falls back to ``pycolmap`` for binary
    models. Raises with actionable guidance when neither is available.
    """
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"
    if cameras_txt.exists() and images_txt.exists():
        return _parse_images_txt(images_txt, _parse_cameras_txt(cameras_txt))
    try:
        return _read_colmap_via_pycolmap(model_dir)
    except ImportError:
        raise RuntimeError(
            f"No text model (cameras.txt/images.txt) in {model_dir} and pycolmap "
            "is not installed. Either export a text model with "
            "`colmap model_converter --input_path <bin_dir> --output_path <dir> "
            "--output_type TXT`, or run `uv add pycolmap`."
        ) from None


# === Prediction loading ===


def load_prediction(npz_path: Path) -> Prediction:
    """Load a prediction dump and keep only the static-point correspondences."""
    data = np.load(npz_path, allow_pickle=True)
    xyz = np.asarray(data["point_xyz_ref0"], dtype=np.float64)  # [F, P, 3]
    uv = np.asarray(data["point_uv_px"], dtype=np.float64)  # [F, P, 2]
    is_dynamic = np.asarray(data["point_is_dynamic"], dtype=bool)  # [P]
    K = np.asarray(data["ref0_K"], dtype=np.float64)  # [3, 3]
    frame_names = [str(n) for n in data["frame_names"].tolist()]
    n_frames, n_points = uv.shape[0], uv.shape[1]
    if "point_visibility" in data:
        visible = np.asarray(data["point_visibility"], dtype=bool)  # [F, P]
    else:
        visible = np.ones((n_frames, n_points), dtype=bool)

    # Keep static points; drop any observation that is invisible or non-finite.
    static = ~is_dynamic
    visible = visible & np.isfinite(uv).all(axis=2) & np.isfinite(xyz).all(axis=2)
    visible[:, ~static] = False

    # Robust canonical 3D per point: median over the frames where it is visible.
    xyz_masked = np.where(visible[:, :, None], xyz, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        xyz_canonical = np.nanmedian(xyz_masked, axis=0)  # [P, 3]
    keep = static & np.isfinite(xyz_canonical).all(axis=1)
    if int(keep.sum()) < 4:
        raise ValueError(
            f"Need >=4 usable static points for PnP; got {int(keep.sum())}. "
            "Check the dynamic mask / visibility or the upstream dynamic threshold."
        )
    return Prediction(
        xyz_ref0=xyz_canonical[keep],
        uv_per_frame=uv[:, keep, :],
        visible=visible[:, keep],
        K=K,
        frame_names=frame_names,
    )


# === Geometry ===


def camera_center(R_wc: FloatArray, t_wc: FloatArray) -> FloatArray:
    """World-frame camera center C = -R_wc^T t_wc."""
    return -R_wc.T @ t_wc


def solve_pnp_pose(
    object_pts: FloatArray, image_pts: FloatArray, K: FloatArray, use_ransac: bool
) -> tuple[FloatArray, FloatArray, int] | None:
    """Recover a world->camera pose from 2D-3D correspondences via PnP.

    Returns ``(R_wc, t_wc, n_inliers)`` or ``None`` when PnP fails.
    """
    obj = np.ascontiguousarray(object_pts.reshape(-1, 1, 3), dtype=np.float64)
    img = np.ascontiguousarray(image_pts.reshape(-1, 1, 2), dtype=np.float64)
    if obj.shape[0] < 4:
        return None
    if use_ransac and obj.shape[0] >= 6:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, None, reprojectionError=4.0, flags=cv2.SOLVEPNP_ITERATIVE
        )
        n_inliers = 0 if inliers is None else int(len(inliers))
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        n_inliers = obj.shape[0]
    if not ok:
        return None
    R_wc, _ = cv2.Rodrigues(rvec)
    return R_wc.astype(np.float64), tvec.reshape(3).astype(np.float64), n_inliers


def umeyama_sim3(
    src: FloatArray, dst: FloatArray
) -> tuple[float, FloatArray, FloatArray]:
    """Least-squares similarity (Umeyama 1991) mapping ``src`` onto ``dst``.

    Returns ``(scale, R, t)`` such that ``dst ~= scale * R @ src + t``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s_diag = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_diag[-1] = -1.0
    R = u @ np.diag(s_diag) @ vt
    var_src = (src_c**2).sum() / n
    scale = float((d * s_diag).sum() / var_src) if var_src > 0 else 1.0
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def rotation_geodesic_deg(R1: FloatArray, R2: FloatArray) -> float:
    """Geodesic angle (degrees) between two rotation matrices."""
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def average_rotation(rotations: list[FloatArray]) -> FloatArray:
    """Chordal-L2 mean rotation (project the matrix sum onto SO(3))."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    return u @ np.diag(d) @ vt


# === Metrics ===


_MIN_PNP_POINTS = 6


def _matched_frames(
    pred: Prediction, colmap: dict[str, ColmapCamera], use_ransac: bool
) -> tuple[list[str], FloatArray, FloatArray, list[FloatArray], list[FloatArray]]:
    """Return frame names plus aligned pred/gt centers and rotations."""
    names: list[str] = []
    pred_centers: list[FloatArray] = []
    gt_centers: list[FloatArray] = []
    pred_R: list[FloatArray] = []
    gt_R: list[FloatArray] = []
    for f, name in enumerate(pred.frame_names):
        cam = colmap.get(name)
        if cam is None:
            continue
        mask = pred.visible[f]
        if int(mask.sum()) < _MIN_PNP_POINTS:
            continue
        pose = solve_pnp_pose(
            pred.xyz_ref0[mask], pred.uv_per_frame[f][mask], pred.K, use_ransac
        )
        if pose is None:
            continue
        R_wc, t_wc, _ = pose
        names.append(name)
        pred_centers.append(camera_center(R_wc, t_wc))
        pred_R.append(R_wc)
        gt_centers.append(camera_center(cam.R_wc, cam.t_wc))
        gt_R.append(cam.R_wc)
    return (
        names,
        np.asarray(pred_centers, dtype=np.float64).reshape(-1, 3),
        np.asarray(gt_centers, dtype=np.float64).reshape(-1, 3),
        pred_R,
        gt_R,
    )


def compute_consistency(
    pred: Prediction, colmap: dict[str, ColmapCamera], use_ransac: bool
) -> dict[str, Any]:
    """Align the derived trajectory to COLMAP and compute consistency metrics."""
    names, pred_c, gt_c, pred_R, gt_R = _matched_frames(pred, colmap, use_ransac)
    if len(names) < 3:
        raise RuntimeError(
            f"Only {len(names)} frames matched COLMAP+PnP; need >=3 to align."
        )

    scale, R_align, t_align = umeyama_sim3(pred_c, gt_c)
    aligned = (scale * (R_align @ pred_c.T)).T + t_align
    residuals = np.linalg.norm(aligned - gt_c, axis=1)

    # Estimate the world-frame rotation from the camera orientations rather than
    # the centres: gt_R = pred_R @ Rg^T, so Rg = average(gt_R^T @ pred_R). This is
    # well-defined even when the trajectory is near-collinear (where the
    # centre-based Umeyama rotation is under-determined).
    rg = average_rotation([gr.T @ pr for pr, gr in zip(pred_R, gt_R)])
    rot_errors = [rotation_geodesic_deg(pr @ rg.T, gr) for pr, gr in zip(pred_R, gt_R)]

    # RPE: consecutive position-step error between the aligned and COLMAP tracks.
    step_err = np.linalg.norm(np.diff(aligned, axis=0) - np.diff(gt_c, axis=0), axis=1)

    gt_extent = float(np.linalg.norm(gt_c.max(axis=0) - gt_c.min(axis=0))) or 1.0
    return {
        "matched_frames": len(names),
        "total_pred_frames": len(pred.frame_names),
        "sim3_scale": scale,
        "ate_rmse": float(np.sqrt((residuals**2).mean())),
        "ate_median": float(np.median(residuals)),
        "ate_max": float(residuals.max()),
        "ate_rmse_normalized": float(np.sqrt((residuals**2).mean()) / gt_extent),
        "rotation_error_deg_mean": float(np.mean(rot_errors)),
        "rotation_error_deg_median": float(np.median(rot_errors)),
        "rpe_translation_rmse": float(np.sqrt(np.mean(np.square(step_err))))
        if step_err.size
        else 0.0,
        "intrinsics_pred": _k_to_dict(pred.K),
        "intrinsics_colmap": _k_to_dict(next(iter(colmap.values())).K),
        "_aligned_pred_centers": aligned.tolist(),
        "_gt_centers": gt_c.tolist(),
        "_frame_names": names,
    }


def _k_to_dict(K: FloatArray) -> dict[str, float]:
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


# === Reporting ===


def _write_plot(report: dict[str, Any], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aligned = np.asarray(report["_aligned_pred_centers"])
    gt = np.asarray(report["_gt_centers"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt[:, 0], gt[:, 2], "-o", label="COLMAP", ms=3)
    ax.plot(aligned[:, 0], aligned[:, 2], "-x", label="D4RT (PnP, aligned)", ms=4)
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Trajectory (ATE RMSE={report['ate_rmse']:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    """Strip the internal plotting arrays (prefixed with ``_``) from the report."""
    return {k: v for k, v in report.items() if not k.startswith("_")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="D4RT prediction .npz dump.")
    parser.add_argument("--colmap-model", required=True, help="COLMAP model directory.")
    parser.add_argument("--output", help="Write the JSON report to this path.")
    parser.add_argument("--plot", help="Write a top-down trajectory plot (PNG).")
    parser.add_argument(
        "--ransac", action="store_true", help="Use RANSAC PnP (robust to outliers)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred = load_prediction(Path(args.pred).resolve())
    colmap = read_colmap_model(Path(args.colmap_model).resolve())
    report = compute_consistency(pred, colmap, use_ransac=args.ransac)

    public = _public_report(report)
    if args.output:
        Path(args.output).resolve().write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.plot:
        _write_plot(report, Path(args.plot).resolve())
    print(json.dumps(public, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
