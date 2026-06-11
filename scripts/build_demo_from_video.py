#!/usr/bin/env python3
"""Build a Viser demo package from a local video or GIF.

Pipeline overview:
  1. Parse CLI args and resolve/validate the config, input, and checkpoint paths.
  2. Decode the input clip into RGB frames (with a PIL GIF fallback decoder).
  3. Build the D4RT model and load the checkpoint weights onto the target device.
  4. Run inference to produce per-frame point and track predictions.
  5. Write the demo package: ``demo_data.json`` + ``manifest.json`` + video assets.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped
from PIL import Image, ImageSequence

# Local script execution needs repo root on sys.path before project imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import (  # noqa: E402
    _load_video_rgb,
    _resolve_device,
    _resize_video,
    _unwrap_state_dict,
)
from scripts._demo_types import (  # noqa: E402
    BoundsRadius,
    BoundsVector,
    DemoPackage,
    IntrinsicsMatrix,
    PointConfidenceArray,
    PointDynamicMaskArray,
    PointFrameUvArray,
    PointMotionScoreArray,
    PointRgbArray,
    PointVisibilityArray,
    PointXyzArray,
    TrackConfidenceArray,
    TrackFrameUvArray,
    TrackQueryTimeArray,
    TrackVisibilityArray,
    TrackXyzArray,
    UvPointArray,
    UvTrackArray,
    VideoRGB,
)
from src.core import build_logger, load_yaml_config, seed_everything  # noqa: E402
from src.model import build_model  # noqa: E402
from vis.build_like_demo import (  # noqa: E402
    _build_uv_grid,
    _export_demo_data,
    _export_video_from_frames,
    _jsonable_float_array,
)

# Fallback output FPS used when neither the CLI override nor the input clip
# supplies a positive frame rate.
DEFAULT_OUTPUT_FPS = 8.0

# Package fields covered by the runtime shape/dtype contract in
# ``_validate_demo_arrays``. Mirrors that function's keyword parameters so the
# call site can be data-driven instead of re-spelling every field.
_VALIDATED_ARRAY_FIELDS: tuple[str, ...] = (
    "track_query_uv_px",
    "track_query_t_src",
    "track_xyz_ref0",
    "track_uv_px",
    "track_visibility",
    "track_confidence",
    "point_query_uv_px",
    "point_xyz_ref0",
    "point_visibility",
    "point_uv_px",
    "point_confidence",
    "point_motion_score",
    "point_is_dynamic",
    "point_rgb",
    "bounds_min",
    "bounds_max",
    "bounds_center",
    "bounds_radius",
    "ref0_K",
)


# === Argument type validators ===


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _non_negative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a float, got {raw!r}") from exc
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"expected a non-negative float, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight Viser demo package from a local video or GIF."
    )
    parser.add_argument("--config", required=True, help="Model config YAML.")
    parser.add_argument("--ckpt-path", required=True, help="OpenD4RT checkpoint path.")
    parser.add_argument("--input", required=True, help="Input video/GIF path.")
    parser.add_argument(
        "--output-dir", required=True, help="Output demo package directory."
    )
    parser.add_argument("--num-frames", type=_positive_int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-grid-cols", type=_positive_int, default=8)
    parser.add_argument("--point-grid-rows", type=_positive_int, default=8)
    parser.add_argument("--point-max-points", type=_positive_int, default=64)
    parser.add_argument("--track-max-points", type=_positive_int, default=16)
    parser.add_argument("--track-min-visible-frames", type=_positive_int, default=2)
    parser.add_argument("--point-query-chunk-size", type=_positive_int, default=16)
    parser.add_argument("--track-query-chunk-size", type=_positive_int, default=16)
    parser.add_argument(
        "--fps",
        type=_non_negative_float,
        default=0.0,
        help="Override output FPS. Use 0 to preserve input FPS.",
    )
    parser.add_argument("--no-suppress-depth-boundary-tracks", action="store_true")
    return parser.parse_args()


# === Small conversion / guard helpers ===


def _resolve_fps(provided_fps: float, fallback_fps: float) -> float:
    """Return ``provided_fps`` when positive, else ``fallback_fps``."""
    return float(provided_fps if provided_fps > 0.0 else fallback_fps)


def _ensure_file_exists(path: Path) -> None:
    """Raise ``FileNotFoundError`` when a required input path is missing."""
    if not path.exists():
        raise FileNotFoundError(path)


def _to_int_list(value: Any) -> list[Any]:
    """Convert an array-like to a JSON-serializable int32 nested list."""
    return np.asarray(value, dtype=np.int32).tolist()


# === Frame decoding ===


@jaxtyped(typechecker=beartype)
def _decode_gif_frames(
    path: Path, max_frames: int, video_decode_error: Exception
) -> tuple[VideoRGB, float]:
    """Decode a GIF into RGB frames with PIL when the video decoder fails."""
    frames: list[np.ndarray] = []
    try:
        with Image.open(path) as img:
            duration_ms = float(img.info.get("duration", 125) or 125)
            fps = 1000.0 / max(duration_ms, 1.0)
            for frame in ImageSequence.Iterator(img):
                frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
                if len(frames) >= int(max_frames):
                    break
    except Exception as gif_error:
        raise RuntimeError(f"Failed to decode GIF fallback for {path}") from gif_error
    if not frames:
        raise RuntimeError(
            f"No frames decoded from GIF: {path}"
        ) from video_decode_error
    return np.stack(frames, axis=0), float(fps)


@jaxtyped(typechecker=beartype)
def _load_frames(path: Path, max_frames: int) -> tuple[VideoRGB, float]:
    """Decode the input clip, falling back to PIL GIF decoding on failure."""
    try:
        return _load_video_rgb(path, max_frames=max_frames)
    except Exception as video_decode_error:
        if path.suffix.lower() != ".gif":
            raise
        return _decode_gif_frames(path, max_frames, video_decode_error)


# === Demo-package validation and writing ===


@jaxtyped(typechecker=beartype)
def _validate_demo_arrays(
    *,
    video_rgb: VideoRGB,
    track_query_uv_px: UvTrackArray,
    track_query_t_src: TrackQueryTimeArray,
    track_xyz_ref0: TrackXyzArray,
    track_uv_px: TrackFrameUvArray,
    track_visibility: TrackVisibilityArray,
    track_confidence: TrackConfidenceArray,
    point_query_uv_px: UvPointArray,
    point_xyz_ref0: PointXyzArray,
    point_visibility: PointVisibilityArray,
    point_uv_px: PointFrameUvArray,
    point_confidence: PointConfidenceArray,
    point_motion_score: PointMotionScoreArray,
    point_is_dynamic: PointDynamicMaskArray,
    point_rgb: PointRgbArray,
    bounds_min: BoundsVector,
    bounds_max: BoundsVector,
    bounds_center: BoundsVector,
    bounds_radius: BoundsRadius,
    ref0_K: IntrinsicsMatrix,
) -> None:
    """Validate demo-package array shapes, dtypes, and cross-dimension consistency.

    The body is intentionally empty: the ``@jaxtyped(typechecker=beartype)``
    decorator enforces every parameter's shape/dtype and the shared axis names
    (``frames``/``tracks``/``points``) across all arrays at call time, acting as
    a precondition guard before viewer assets are written.
    """


def _validate_package_contract(package: DemoPackage, video_rgb: VideoRGB) -> None:
    """Run the runtime array contract over a demo package in one call.

    Data-drives the validation from :data:`_VALIDATED_ARRAY_FIELDS` so the field
    list lives in exactly one place. Splatting the fields as keyword arguments
    keeps jaxtyping's per-field and cross-dimension checks fully active.
    """
    fields = cast("dict[str, Any]", package)
    _validate_demo_arrays(
        video_rgb=video_rgb,
        **{name: fields[name] for name in _VALIDATED_ARRAY_FIELDS},
    )


def _build_demo_meta(
    package: DemoPackage, input_path: Path, output_fps: float
) -> dict[str, Any]:
    """Assemble the ``meta`` block describing the demo package."""
    return {
        "fps": output_fps,
        "numFrames": int(package["num_frames"]),
        "videoWidth": int(package["video_width"]),
        "videoHeight": int(package["video_height"]),
        "crop": {"top": 0, "bottom": int(package["video_height"])},
        "clipFrames": int(package["clip_frames"]),
        "trackStitchDiagnostics": package.get("track_stitch_diagnostics", {}),
        "trackCount": int(package["track_query_uv_px"].shape[0]),
        "trackCountPred": int(package["track_query_uv_px"].shape[0]),
        "trackCountGt": 0,
        "pointCountPerFrame": int(package["point_query_uv_px"].shape[0]),
        "bounds": {
            "min": _jsonable_float_array(package["bounds_min"]),
            "max": _jsonable_float_array(package["bounds_max"]),
            "center": _jsonable_float_array(package["bounds_center"]),
            "radius": float(package["bounds_radius"][0]),
        },
        "ref0K": _jsonable_float_array(package["ref0_K"], ndigits=5),
        "camera": None,
        "cameraPred": None,
        "depthPred": None,
        "source": {
            "type": "local_video",
            "path": str(input_path),
        },
    }


def _build_demo_data_json(package: DemoPackage, meta: dict[str, Any]) -> dict[str, Any]:
    """Assemble the viewer ``demo_data.json`` payload from a demo package."""
    return {
        "meta": meta,
        "tracks": {
            "queryUvPx": _jsonable_float_array(package["track_query_uv_px"], ndigits=3),
            "queryTSrc": _to_int_list(package["track_query_t_src"]),
            "xyzRef0": _jsonable_float_array(package["track_xyz_ref0"], ndigits=5),
            "uvPx": _jsonable_float_array(package["track_uv_px"], ndigits=3),
            "visibility": _to_int_list(package["track_visibility"]),
            "confidence": _jsonable_float_array(package["track_confidence"], ndigits=4),
        },
        "points": {
            "queryUvPx": _jsonable_float_array(package["point_query_uv_px"], ndigits=3),
            "xyzRef0": _jsonable_float_array(package["point_xyz_ref0"], ndigits=5),
            "visibility": _to_int_list(package["point_visibility"]),
            "rgb": _to_int_list(package["point_rgb"]),
            "uvPx": _jsonable_float_array(package["point_uv_px"], ndigits=3),
            "confidence": _jsonable_float_array(package["point_confidence"], ndigits=4),
            "motionScore": _jsonable_float_array(
                package["point_motion_score"], ndigits=5
            ),
            "isDynamic": _to_int_list(package["point_is_dynamic"]),
        },
        "pointsRaw": {
            "xyzRef0": _jsonable_float_array(package["point_xyz_ref0"], ndigits=5),
        },
    }


def _write_viewer_files(
    *,
    output_dir: Path,
    assets_dir: Path,
    input_path: Path,
    data_json: dict[str, Any],
    video_rgb: VideoRGB,
    output_fps: float,
) -> None:
    """Write ``demo_data.json``, the video/poster assets, and ``manifest.json``."""
    (assets_dir / "demo_data.json").write_text(
        json.dumps(data_json, ensure_ascii=False), encoding="utf-8"
    )
    video_name, poster_name = _export_video_from_frames(
        video_rgb=video_rgb,
        fps=output_fps,
        dst_video=assets_dir / "input_video.mp4",
    )
    manifest = {
        "input": str(input_path),
        "video_copy": f"assets/{video_name}",
        "video_poster": f"assets/{poster_name}",
        "data_json": "assets/demo_data.json",
        "viewer": "viser",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_demo_package(
    *,
    output_dir: Path,
    input_path: Path,
    package: DemoPackage,
    video_rgb: VideoRGB,
    fps: float,
) -> None:
    """Validate the package and write the full Viser demo bundle to disk."""
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_fps = _resolve_fps(fps, DEFAULT_OUTPUT_FPS)

    _validate_package_contract(package, video_rgb)
    meta = _build_demo_meta(package, input_path, output_fps)
    data_json = _build_demo_data_json(package, meta)
    _write_viewer_files(
        output_dir=output_dir,
        assets_dir=assets_dir,
        input_path=input_path,
        data_json=data_json,
        video_rgb=video_rgb,
        output_fps=output_fps,
    )


# === Inference orchestration ===


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve and validate the config, input, checkpoint, and output paths."""
    config_path = Path(args.config).resolve()
    input_path = Path(args.input).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    for path in (config_path, input_path, ckpt_path):
        _ensure_file_exists(path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return config_path, input_path, ckpt_path, output_dir


def _load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """Load and unwrap a checkpoint into a model state dict."""
    try:
        payload = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = _unwrap_state_dict(payload)
    if not state:
        raise RuntimeError(f"No model weights found in checkpoint: {path}")
    return state


def _build_inference_model(
    cfg: Any, ckpt_path: Path, device_arg: str
) -> tuple[torch.nn.Module, torch.device]:
    """Build the model, load checkpoint weights, and move it to the target device.

    The device is resolved only after the checkpoint loads successfully, so a
    checkpoint mismatch surfaces before any device-availability error (matching
    the original call ordering). Returns the ready model and its resolved device.
    """
    model = build_model(cfg["model"]).eval()
    state = _load_checkpoint_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    del state
    gc.collect()
    device = _resolve_device(device_arg)
    return model.to(device).eval(), device


def main() -> int:
    args = parse_args()
    config_path, input_path, ckpt_path, output_dir = _resolve_paths(args)
    logger = build_logger("build_demo_from_video", output_dir)

    cfg = load_yaml_config(config_path)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)

    video_rgb, input_fps = _load_frames(input_path, max_frames=args.num_frames)
    fps = _resolve_fps(args.fps, input_fps)
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    video_model_rgb = _resize_video(
        video_rgb, image_hw=(int(image_size[0]), int(image_size[1]))
    )

    point_query_uv_px = _build_uv_grid(
        width=int(video_rgb.shape[2]),
        height=int(video_rgb.shape[1]),
        cols=int(args.point_grid_cols),
        rows=int(args.point_grid_rows),
        max_points=args.point_max_points,
    )

    logger.info("Building model from %s", config_path)
    model, device = _build_inference_model(cfg, ckpt_path, args.device)
    logger.info(
        "Running inference on %s with %d frames and %d point queries",
        device,
        video_rgb.shape[0],
        point_query_uv_px.shape[0],
    )
    package = _export_demo_data(
        model=model,
        video_rgb=video_rgb,
        video_model_rgb=video_model_rgb,
        point_query_uv_px=point_query_uv_px,
        point_query_chunk_size=args.point_query_chunk_size,
        track_query_chunk_size=args.track_query_chunk_size,
        track_selection="motion",
        track_max_points=args.track_max_points,
        track_min_visible_frames=args.track_min_visible_frames,
        suppress_depth_boundary_tracks=not bool(args.no_suppress_depth_boundary_tracks),
    )
    _write_demo_package(
        output_dir=output_dir,
        input_path=input_path,
        package=cast(DemoPackage, package),
        video_rgb=video_rgb,
        fps=fps,
    )
    logger.info("Saved demo package to %s", output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
