from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import check_colmap_trajectory_consistency as ck


def _look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV-convention world->camera pose (x right, y down, z forward)."""
    up = np.array([0.0, 1.0, 0.0])
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R_cw = np.stack([r, d, f], axis=1)  # cam axes expressed in world
    R_wc = R_cw.T
    return R_wc, -R_wc @ eye


def _rot_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return (
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return (
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return (
        (R[1, 0] - R[0, 1]) / s,
        (R[0, 2] + R[2, 0]) / s,
        (R[1, 2] + R[2, 1]) / s,
        0.25 * s,
    )


def _project(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, pts: np.ndarray
) -> np.ndarray:
    cam = (R @ pts.T).T + t
    uv = (K @ (cam / cam[:, 2:3]).T).T
    return uv[:, :2]


def _build_synthetic_scene():
    """A ref0 camera arc viewing a static point cloud, plus a known Sim3 to world."""
    rng = np.random.default_rng(0)
    K = np.array([[480.0, 0.0, 320.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]])
    n_points, n_frames = 40, 8

    # A compact static point cloud near the origin.
    pts_ref0 = rng.uniform(-1.5, 1.5, size=(n_points, 3))
    target = pts_ref0.mean(axis=0)

    # Cameras orbit the cloud on a non-collinear arc, each looking at it.
    ref0_R, ref0_t, frame_names = [], [], []
    for i in range(n_frames):
        a = np.deg2rad(-28 + i * 8.0)
        eye = np.array([6.0 * np.sin(a), 0.4 * np.sin(2 * a), -6.0 * np.cos(a)])
        R, t = _look_at(eye, target)
        ref0_R.append(R)
        ref0_t.append(t)
        frame_names.append(f"frame_{i + 1:05d}.jpg")

    uv = np.stack(
        [_project(K, ref0_R[i], ref0_t[i], pts_ref0) for i in range(n_frames)], axis=0
    )
    # Sanity: all points project in front of all cameras.
    assert all(
        ((ref0_R[i] @ pts_ref0.T).T + ref0_t[i])[:, 2].min() > 0
        for i in range(n_frames)
    )

    # Known Sim3 mapping ref0 -> COLMAP world: X_world = s * Rg @ X_ref0 + tg.
    s, Rg, tg = (
        2.5,
        cv2.Rodrigues(np.array([0.10, 0.20, -0.15]))[0],
        np.array([3.0, -1.0, 4.0]),
    )
    colmap_R, colmap_t = [], []
    for i in range(n_frames):
        colmap_R.append(ref0_R[i] @ Rg.T)
        colmap_t.append(s * ref0_t[i] - ref0_R[i] @ Rg.T @ tg)
    return K, pts_ref0, uv, frame_names, ref0_R, ref0_t, (s, Rg, tg), colmap_R, colmap_t


def _write_colmap_text(model_dir: Path, K, names, colmap_R, colmap_t) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.txt").write_text(
        f"# camera\n1 PINHOLE 640 480 {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n"
    )
    lines = ["# images"]
    for name, R, t in zip(names, colmap_R, colmap_t):
        qw, qx, qy, qz = _rot_to_quat_wxyz(R)
        lines.append(f"1 {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}")
        lines.append("")  # empty POINTS2D line
    (model_dir / "images.txt").write_text("\n".join(lines) + "\n")


def _write_pred_npz(npz_path: Path, K, pts_ref0, uv, names) -> None:
    n_frames, n_points = uv.shape[0], uv.shape[1]
    np.savez(
        npz_path,
        point_xyz_ref0=np.tile(pts_ref0[None], (n_frames, 1, 1)).astype(np.float32),
        point_uv_px=uv.astype(np.float32),
        point_is_dynamic=np.zeros((n_points,), dtype=bool),
        ref0_K=K.astype(np.float32),
        frame_names=np.array(names),
    )


def test_umeyama_recovers_known_sim3() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    s_true, R_true = 1.7, cv2.Rodrigues(np.array([0.3, -0.2, 0.5]))[0]
    t_true = np.array([1.0, 2.0, -3.0])
    dst = (s_true * (R_true @ src.T)).T + t_true

    s, R, t = ck.umeyama_sim3(src, dst)

    assert s == np.float64(s) and abs(s - s_true) < 1e-9
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)


def test_quat_rotation_roundtrip() -> None:
    R = cv2.Rodrigues(np.array([0.4, -0.1, 0.9]))[0]
    qw, qx, qy, qz = _rot_to_quat_wxyz(R)
    assert np.allclose(ck._quat_wxyz_to_rotation(qw, qx, qy, qz), R, atol=1e-9)


def test_colmap_text_parse_reads_pose_and_intrinsics(tmp_path: Path) -> None:
    K, _, _, names, _, _, _, colmap_R, colmap_t = _build_synthetic_scene()
    _write_colmap_text(tmp_path, K, names, colmap_R, colmap_t)

    images = ck.read_colmap_model(tmp_path)

    assert set(images) == set(names)
    cam0 = images[names[0]]
    assert np.allclose(cam0.R_wc, colmap_R[0], atol=1e-6)
    assert np.allclose(cam0.t_wc, colmap_t[0], atol=1e-6)
    assert ck._k_to_dict(cam0.K) == {
        "fx": K[0, 0],
        "fy": K[1, 1],
        "cx": K[0, 2],
        "cy": K[1, 2],
    }


def test_end_to_end_recovers_consistent_trajectory(tmp_path: Path) -> None:
    K, pts_ref0, uv, names, _, _, sim3, colmap_R, colmap_t = _build_synthetic_scene()
    s_true = sim3[0]
    _write_colmap_text(tmp_path / "model", K, names, colmap_R, colmap_t)
    _write_pred_npz(tmp_path / "pred.npz", K, pts_ref0, uv, names)

    pred = ck.load_prediction(tmp_path / "pred.npz")
    colmap = ck.read_colmap_model(tmp_path / "model")
    report = ck.compute_consistency(pred, colmap, use_ransac=False)

    assert report["matched_frames"] == len(names)
    # Exact (noise-free) correspondences -> near-perfect recovery.
    assert report["ate_rmse_normalized"] < 1e-4
    assert abs(report["sim3_scale"] - s_true) < 1e-3
    assert report["rotation_error_deg_mean"] < 1e-2
    assert report["intrinsics_colmap"]["fx"] == K[0, 0]


def test_load_prediction_requires_enough_static_points(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "bad.npz",
        point_xyz_ref0=np.zeros((2, 3, 3), dtype=np.float32),
        point_uv_px=np.zeros((2, 3, 2), dtype=np.float32),
        point_is_dynamic=np.ones((3,), dtype=bool),  # all dynamic -> 0 static
        ref0_K=np.eye(3, dtype=np.float32),
        frame_names=np.array(["a.jpg", "b.jpg"]),
    )
    try:
        ck.load_prediction(tmp_path / "bad.npz")
        raise AssertionError("expected ValueError for too few static points")
    except ValueError:
        pass


def test_handles_pixel_noise_with_ransac(tmp_path: Path) -> None:
    K, pts_ref0, uv, names, _, _, sim3, colmap_R, colmap_t = _build_synthetic_scene()
    rng = np.random.default_rng(7)
    noisy_uv = uv + rng.normal(scale=0.4, size=uv.shape)
    _write_colmap_text(tmp_path / "model", K, names, colmap_R, colmap_t)
    _write_pred_npz(tmp_path / "pred.npz", K, pts_ref0, noisy_uv, names)

    pred = ck.load_prediction(tmp_path / "pred.npz")
    colmap = ck.read_colmap_model(tmp_path / "model")
    report = ck.compute_consistency(pred, colmap, use_ransac=True)

    assert report["matched_frames"] == len(names)
    # Sub-pixel noise -> still tightly consistent after Sim3 alignment.
    assert report["ate_rmse_normalized"] < 0.05
    assert abs(report["sim3_scale"] - sim3[0]) < 0.1
    assert report["rotation_error_deg_mean"] < 2.0
