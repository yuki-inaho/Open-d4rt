#!/usr/bin/env python3
"""Run D4RT on a clip and dump a dense-scene ``.npz`` for the Rerun viewer.

The dump feeds ``vis.rerun_visualize.save_dense_scene_to_rrd``, which renders, per
frame, the RGB image, an estimated depth image, the colored 3D point cloud, and
the camera frustum. D4RT is query-point based (no dense per-pixel depth), so we
query a dense regular grid; its per-frame z (in the PnP-recovered camera frame)
forms the low-resolution depth image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import _resize_video  # noqa: E402
from scripts.build_demo_from_video import _build_inference_model  # noqa: E402
from scripts.dump_static_tracks_for_trajectory import (  # noqa: E402
    _load_named_frames,
    _select_frame_names,
)
from src.core import build_logger, load_yaml_config, seed_everything  # noqa: E402
from vis.build_like_demo import _build_uv_grid, _export_demo_data  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument(
        "--image-dir", required=True, help="Directory of source frames."
    )
    parser.add_argument(
        "--frame-list", help="Optional file of frame names; else sorted dir order."
    )
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-cols", type=int, default=96)
    parser.add_argument("--grid-rows", type=int, default=72)
    parser.add_argument("--query-chunk-size", type=int, default=96)
    return parser.parse_args()


def _resolve_frame_names(
    image_dir: Path, frame_list: str | None, num_frames: int, stride: int
) -> list[str]:
    if frame_list is not None:
        return _select_frame_names(Path(frame_list), num_frames, stride)
    files = sorted(
        p.name for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    selected = files[:: max(stride, 1)][:num_frames]
    if len(selected) < 3:
        raise ValueError(f"Need >=3 frames; selected {len(selected)} from {image_dir}.")
    return selected


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = build_logger("dump_dense_scene", output_path.parent)

    names = _resolve_frame_names(
        image_dir, args.frame_list, args.num_frames, args.frame_stride
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
    model, _ = _build_inference_model(cfg, ckpt_path, args.device)
    logger.info(
        "Dense grid inference: %d frames, %d points", video_rgb.shape[0], grid.shape[0]
    )
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

    # track_* are [points, frames, ...]; the viewer wants [frames, points, ...].
    track_xyz = np.asarray(package["track_xyz_ref0"], dtype=np.float32)  # [P, F, 3]
    track_uv = np.asarray(package["track_uv_px"], dtype=np.float32)  # [P, F, 2]
    track_vis = np.asarray(package["track_visibility"], dtype=bool)  # [P, F]
    point_rgb = np.asarray(package["point_rgb"], dtype=np.uint8)  # [F, P, 3]

    np.savez(
        output_path,
        point_xyz_ref0=track_xyz.transpose(1, 0, 2),
        point_uv_px=track_uv.transpose(1, 0, 2),
        point_visibility=track_vis.transpose(1, 0),
        point_is_dynamic=np.asarray(package["point_is_dynamic"], dtype=bool),
        rgb=point_rgb[0],
        grid_uv=grid.astype(np.float32),
        ref0_K=np.asarray(package["ref0_K"], dtype=np.float32),
        frames_rgb=video_rgb.astype(np.uint8),
        frame_names=np.array(names),
    )
    logger.info(
        "Saved dense scene (%d frames, %d points) to %s",
        track_xyz.shape[1],
        track_xyz.shape[0],
        output_path,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
