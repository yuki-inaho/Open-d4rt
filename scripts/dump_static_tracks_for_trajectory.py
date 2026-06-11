#!/usr/bin/env python3
"""Run D4RT on a COLMAP image subset and dump static-point tracks as an .npz.

The companion ``check_colmap_trajectory_consistency.py`` derives a camera
trajectory from static-point 2D-3D correspondences via PnP. That needs, per
frame, each point's *moving* 2D projection -- which lives in D4RT's ``track_*``
outputs (``point_uv_px`` is just the static query grid). So we run the model in
grid-track mode over a dense query grid, keep the points the model flags as
static, and write their per-frame tracks (uv), reference 3D, visibility, and the
ref0 intrinsics, tagged with the matching COLMAP image file names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import _resize_video  # noqa: E402
from scripts.build_demo_from_video import _build_inference_model  # noqa: E402
from src.core import build_logger, load_yaml_config, seed_everything  # noqa: E402
from vis.build_like_demo import _build_uv_grid, _export_demo_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Model config YAML.")
    parser.add_argument("--ckpt-path", required=True, help="OpenD4RT checkpoint path.")
    parser.add_argument(
        "--image-dir", required=True, help="Directory holding the source frames."
    )
    parser.add_argument(
        "--frame-list",
        required=True,
        help="Text file of image file names (one per line), in temporal order.",
    )
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--frame-stride", type=int, default=1, help="Sub-sample the frame list."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-cols", type=int, default=20)
    parser.add_argument("--grid-rows", type=int, default=20)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    return parser.parse_args()


def _select_frame_names(frame_list: Path, num_frames: int, stride: int) -> list[str]:
    names = [ln.strip() for ln in frame_list.read_text().splitlines() if ln.strip()]
    selected = names[:: max(stride, 1)][:num_frames]
    if len(selected) < 3:
        raise ValueError(
            f"Need >=3 frames; selected {len(selected)} from {frame_list}."
        )
    return selected


def _load_named_frames(image_dir: Path, names: list[str]) -> np.ndarray:
    frames = []
    for name in names:
        path = image_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"Frames have differing shapes: {shapes}")
    return np.stack(frames, axis=0)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = build_logger("dump_static_tracks", output_path.parent)

    names = _select_frame_names(
        Path(args.frame_list).resolve(), args.num_frames, args.frame_stride
    )
    video_rgb = _load_named_frames(image_dir, names)
    logger.info("Loaded %d frames %s", video_rgb.shape[0], video_rgb.shape[1:])

    cfg = load_yaml_config(config_path)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    video_model_rgb = _resize_video(
        video_rgb, image_hw=(int(image_size[0]), int(image_size[1]))
    )

    grid = _build_uv_grid(
        width=int(video_rgb.shape[2]),
        height=int(video_rgb.shape[1]),
        cols=int(args.grid_cols),
        rows=int(args.grid_rows),
        max_points=int(args.grid_cols) * int(args.grid_rows),
    )
    logger.info("Built %d grid query points", grid.shape[0])

    model, _ = _build_inference_model(cfg, ckpt_path, args.device)
    logger.info("Running grid-track inference on %d frames", video_rgb.shape[0])
    package = _export_demo_data(
        model=model,
        video_rgb=video_rgb,
        video_model_rgb=video_model_rgb,
        point_query_uv_px=grid,
        point_query_chunk_size=int(args.query_chunk_size),
        track_query_chunk_size=int(args.query_chunk_size),
        track_selection="grid",
        track_max_points=int(grid.shape[0]),
        track_min_visible_frames=2,
        track_query_uv_px=grid,
        suppress_depth_boundary_tracks=True,
    )

    # track_* arrays are [tracks, frames, ...]; the checker wants [frames, points, ...].
    track_xyz = np.asarray(package["track_xyz_ref0"], dtype=np.float32)  # [T, F, 3]
    track_uv = np.asarray(package["track_uv_px"], dtype=np.float32)  # [T, F, 2]
    track_vis = np.asarray(package["track_visibility"], dtype=bool)  # [T, F]
    is_dynamic = np.asarray(package["point_is_dynamic"], dtype=bool)  # [P == T]
    ref0_k = np.asarray(package["ref0_K"], dtype=np.float32)

    np.savez(
        output_path,
        point_xyz_ref0=track_xyz.transpose(1, 0, 2),
        point_uv_px=track_uv.transpose(1, 0, 2),
        point_visibility=track_vis.transpose(1, 0),
        point_is_dynamic=is_dynamic,
        ref0_K=ref0_k,
        frame_names=np.array(names),
    )
    logger.info(
        "Saved %d tracks over %d frames (%d static) to %s",
        track_xyz.shape[0],
        track_xyz.shape[1],
        int((~is_dynamic).sum()),
        output_path,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
