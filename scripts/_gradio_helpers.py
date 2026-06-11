"""Pure helpers for the D4RT gradio viewer (no gradio import, unit-testable).

Keeping the GLB building, package discovery, and metadata formatting out of the
UI module lets them be tested without launching a server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

DYNAMIC_RGB = (255, 60, 60)


def discover_demo_packages(root: str | Path) -> list[Path]:
    """Return demo-package directories (those containing ``manifest.json``)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted({p.parent for p in root.rglob("manifest.json")})


def meta_table_rows(meta: dict[str, Any]) -> list[list[str]]:
    """Flatten a demo-package ``meta`` dict to ``[[key, value], ...]`` rows."""
    rows: list[list[str]] = []
    for key, value in meta.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        rows.append([str(key), str(value)])
    return rows


def _frustum_mesh(
    k: Any, width: int, height: int, depth: float
) -> trimesh.Trimesh | None:
    """A camera frustum (apex + image-plane quad) as a thin mesh, scipy-free.

    A mesh is used instead of a line ``Path3D`` so the GLB export has no SciPy
    dependency (``trimesh.load_path`` requires SciPy).
    """
    try:
        k_inv = np.linalg.inv(np.asarray(k, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    corners = [
        depth * (k_inv @ np.array([u, v, 1.0]))
        for u, v in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    verts = np.array([np.zeros(3), *corners])  # 0 = apex, 1..4 = image corners
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Set vertex colors directly: face colors would trigger a SciPy-backed
    # face->vertex conversion during GLB export.
    mesh.visual.vertex_colors = np.tile(  # ty: ignore[invalid-assignment]
        np.array([0, 200, 255, 90], dtype=np.uint8), (len(verts), 1)
    )
    return mesh


def build_glb_from_demo_data(
    demo_data: dict[str, Any],
    glb_path: str | Path,
    *,
    frame: int = 0,
    show_dynamic: bool = True,
    max_points: int = 200_000,
) -> Path:
    """Export a frame's point cloud (+ ref0 camera frustum) to a GLB for gr.Model3D."""
    glb_path = Path(glb_path)
    meta = demo_data["meta"]
    points = demo_data["points"]

    xyz_all = np.asarray(points["xyzRef0"], dtype=np.float64)  # [F, P, 3]
    if not 0 <= frame < len(xyz_all):
        raise ValueError(f"frame {frame} out of range for {len(xyz_all)} frames")
    xyz = xyz_all[frame]  # [P, 3]
    rgb = np.asarray(points["rgb"], dtype=np.uint8)[frame].copy()  # [P, 3]
    visible = np.asarray(points["visibility"], dtype=bool)[frame]  # [P]
    is_dynamic = np.asarray(points["isDynamic"], dtype=bool)  # [P]

    keep = np.isfinite(xyz).all(axis=1) & visible
    xyz, rgb, is_dynamic = xyz[keep], rgb[keep], is_dynamic[keep]
    if show_dynamic and is_dynamic.any():
        rgb[is_dynamic] = DYNAMIC_RGB
    if max_points > 0 and len(xyz) > max_points:
        print(
            f"build_glb_from_demo_data: subsampling {len(xyz)} -> {max_points} points",
            file=sys.stderr,
        )
        idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz, rgb, is_dynamic = xyz[idx], rgb[idx], is_dynamic[idx]

    rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)], axis=1)
    scene = trimesh.Scene([trimesh.PointCloud(xyz, colors=rgba)])

    radius = float(meta.get("bounds", {}).get("radius", 0.0)) or 0.1
    frustum = _frustum_mesh(
        meta["ref0K"],
        int(meta["videoWidth"]),
        int(meta["videoHeight"]),
        depth=0.5 * radius,
    )
    if frustum is not None:
        scene.add_geometry(frustum)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(glb_path))
    return glb_path
