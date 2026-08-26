#!/usr/bin/env python3
"""Local Gradio app for sparse matching between two image frames with OpenD4RT.

The model is a video model, so the two uploaded images are treated as frame 0
and frame 1.  A regular grid of source-frame pixels is queried and the
predicted frame-1 locations are drawn with matching colors.  The checkpoint is
loaded lazily on the first Match click, rather than while the web server starts.

This app is intended primarily for two frames from the same video.  Images with
different sizes are supported: each is resized independently for model input,
while all displayed coordinates remain in the original image coordinate system.
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch

# Local script execution needs repo root on sys.path before project imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import _build_query_for_targets  # noqa: E402
from scripts.build_demo_from_video import _build_inference_model  # noqa: E402
from src.core import load_yaml_config, seed_everything  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
)
_CANVAS_GUTTER = 28


@dataclass(frozen=True)
class AppSettings:
    """Configuration that is fixed for one running Gradio server."""

    config_path: Path
    checkpoint_path: Path
    device: str


@dataclass(frozen=True)
class MatchSet:
    """Accepted two-frame correspondences in source and target pixel space."""

    source_xy: np.ndarray
    target_xy: np.ndarray
    visibility_probability: np.ndarray
    confidence: np.ndarray


class _ModelLoader:
    """Keep one ready-to-run OpenD4RT instance for the process lifetime."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._model: torch.nn.Module | None = None
        self._config: Any | None = None

    def get(self) -> tuple[torch.nn.Module, Any]:
        """Load weights once and return the model plus its parsed config."""
        if self._model is not None and self._config is not None:
            return self._model, self._config

        config_path = self._settings.config_path.expanduser().resolve()
        checkpoint_path = self._settings.checkpoint_path.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{checkpoint_path}. Download it with the README command first."
            )
        config = load_yaml_config(config_path)
        seed_everything(int(config.get_path("experiment.seed", 42)), deterministic=True)
        model, _ = _build_inference_model(
            config, checkpoint_path, self._settings.device
        )
        self._model = model
        self._config = config
        return model, config


def _as_rgb(image: np.ndarray | None, label: str) -> np.ndarray:
    """Validate a Gradio image value and return a contiguous uint8 RGB image."""
    if image is None:
        raise ValueError(f"Upload {label}.")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"{label} must be a grayscale, RGB, or RGBA image.")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"{label} must be at least 2 by 2 pixels.")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = array * scale
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _model_image_hw(config: Any) -> tuple[int, int]:
    """Read the model's fixed input height and width from its YAML config."""
    image_size = config.get_path("model.input.image_size", [256, 256])
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError(
            f"model.input.image_size must contain [height, width], got {image_size}"
        )
    height, width = (int(image_size[0]), int(image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid model image size: {(height, width)}")
    return height, width


def _resize_frame(frame_rgb: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    interpolation = (
        cv2.INTER_AREA
        if frame_rgb.shape[0] >= height and frame_rgb.shape[1] >= width
        else cv2.INTER_LINEAR
    )
    return cv2.resize(frame_rgb, (width, height), interpolation=interpolation)


def prepare_two_frame_video(
    source_rgb: np.ndarray, target_rgb: np.ndarray, image_hw: tuple[int, int]
) -> np.ndarray:
    """Resize each original frame to the common model resolution and stack it."""
    return np.stack(
        [_resize_frame(source_rgb, image_hw), _resize_frame(target_rgb, image_hw)],
        axis=0,
    )


def make_source_grid(
    width: int,
    height: int,
    cols: int,
    rows: int,
    max_points: int,
    margin_ratio: float = 0.06,
) -> np.ndarray:
    """Build a deterministic source-pixel grid, keeping it safely inside edges."""
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    max_points = max(1, int(max_points))
    margin = float(np.clip(margin_ratio, 0.0, 0.45))
    max_x, max_y = float(max(width - 1, 0)), float(max(height - 1, 0))
    xs = np.linspace(max_x * margin, max_x * (1.0 - margin), num=cols)
    ys = np.linspace(max_y * margin, max_y * (1.0 - margin), num=rows)
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if points.shape[0] > max_points:
        selected = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[selected]
    return points.astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def prediction_to_matches(
    *,
    source_xy: np.ndarray,
    target_uv_norm: np.ndarray,
    visibility_logits: np.ndarray,
    confidence: np.ndarray,
    target_hw: tuple[int, int],
    visibility_threshold: float,
) -> MatchSet:
    """Map normalized predictions to target pixels and remove invalid queries."""
    source_xy = np.asarray(source_xy, dtype=np.float32).reshape(-1, 2)
    target_uv_norm = np.asarray(target_uv_norm, dtype=np.float32).reshape(-1, 2)
    visibility_probability = _sigmoid(visibility_logits).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    count = source_xy.shape[0]
    if not (
        target_uv_norm.shape[0] == count
        and visibility_probability.shape[0] == count
        and confidence.shape[0] == count
    ):
        raise ValueError(
            "Model prediction dimensions do not match the source query count."
        )

    target_height, target_width = target_hw
    target_xy = target_uv_norm.copy()
    target_xy[:, 0] *= float(max(target_width - 1, 1))
    target_xy[:, 1] *= float(max(target_height - 1, 1))
    accepted = (
        np.isfinite(source_xy).all(axis=1)
        & np.isfinite(target_xy).all(axis=1)
        & np.isfinite(visibility_probability)
        & np.isfinite(confidence)
        & (target_uv_norm[:, 0] >= 0.0)
        & (target_uv_norm[:, 0] <= 1.0)
        & (target_uv_norm[:, 1] >= 0.0)
        & (target_uv_norm[:, 1] <= 1.0)
        & (visibility_probability >= float(visibility_threshold))
    )
    return MatchSet(
        source_xy=source_xy[accepted],
        target_xy=target_xy[accepted],
        visibility_probability=visibility_probability[accepted],
        confidence=confidence[accepted],
    )


def infer_two_frame_matches(
    *,
    model: torch.nn.Module,
    config: Any,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    grid_cols: int,
    grid_rows: int,
    max_points: int,
    query_chunk_size: int,
    visibility_threshold: float,
) -> tuple[MatchSet, int]:
    """Run OpenD4RT's frame-0 to frame-1 query interface for a source grid."""
    source_height, source_width = source_rgb.shape[:2]
    target_height, target_width = target_rgb.shape[:2]
    source_xy = make_source_grid(
        source_width, source_height, grid_cols, grid_rows, max_points
    )
    query_uv = source_xy / np.asarray(
        [max(source_width - 1, 1), max(source_height - 1, 1)], dtype=np.float32
    )
    device = next(model.parameters()).device
    video_rgb = prepare_two_frame_video(source_rgb, target_rgb, _model_image_hw(config))
    video_b = (
        torch.from_numpy(video_rgb)
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        / 255.0
    )
    aspect_b = torch.tensor(
        [[float(source_width) / float(source_height)]],
        device=device,
        dtype=torch.float32,
    )
    query_count = source_xy.shape[0]
    query = _build_query_for_targets(
        query_uv_norm=query_uv,
        t_src=np.zeros((query_count,), dtype=np.int64),
        t_tgt=np.ones((query_count,), dtype=np.int64),
        t_cam=np.ones((query_count,), dtype=np.int64),
        device=device,
    )
    with torch.inference_mode():
        memory = _encode_model_memory(model, video_b, aspect_b)
        prediction = _run_model_for_queries(
            model=model,
            video_b=video_b,
            aspect_b=aspect_b,
            query=query,
            chunk_size=max(1, int(query_chunk_size)),
            memory_b=memory,
        )
    matches = prediction_to_matches(
        source_xy=source_xy,
        target_uv_norm=prediction["uv_2d"].numpy(),
        visibility_logits=prediction["visibility"].numpy(),
        confidence=prediction["confidence"].numpy(),
        target_hw=(target_height, target_width),
        visibility_threshold=visibility_threshold,
    )
    return matches, int(query_count)


def _match_color(index: int) -> tuple[int, int, int]:
    """Return perceptually separated, deterministic RGB colors for overlays."""
    hue = (index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.98)
    return round(255 * red), round(255 * green), round(255 * blue)


def render_match_overlay(
    source_rgb: np.ndarray, target_rgb: np.ndarray, matches: MatchSet
) -> np.ndarray:
    """Place original frames side-by-side and draw color-linked correspondences."""
    source_height, source_width = source_rgb.shape[:2]
    target_height, target_width = target_rgb.shape[:2]
    canvas_height = max(source_height, target_height)
    canvas_width = source_width + _CANVAS_GUTTER + target_width
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    canvas[:source_height, :source_width] = source_rgb
    target_x_offset = source_width + _CANVAS_GUTTER
    canvas[:target_height, target_x_offset:] = target_rgb

    for index, (source_xy, target_xy) in enumerate(
        zip(matches.source_xy, matches.target_xy, strict=True)
    ):
        color = _match_color(index)
        source_point = tuple(np.rint(source_xy).astype(np.int32))
        target_point = (
            int(round(float(target_xy[0]))) + target_x_offset,
            int(round(float(target_xy[1]))),
        )
        cv2.line(
            canvas, source_point, target_point, color, thickness=1, lineType=cv2.LINE_AA
        )
        cv2.circle(
            canvas,
            source_point,
            radius=4,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            source_point,
            radius=5,
            color=(20, 20, 20),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            target_point,
            radius=4,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            target_point,
            radius=5,
            color=(20, 20, 20),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def match_rows(matches: MatchSet) -> list[list[float]]:
    """Create Gradio table rows, retaining original source and target pixels."""
    return [
        [
            float(source_xy[0]),
            float(source_xy[1]),
            float(target_xy[0]),
            float(target_xy[1]),
            float(visibility),
            float(confidence),
        ]
        for source_xy, target_xy, visibility, confidence in zip(
            matches.source_xy,
            matches.target_xy,
            matches.visibility_probability,
            matches.confidence,
            strict=True,
        )
    ]


def _build_match_handler(settings: AppSettings):
    """Create the event callback and its process-local lazy model cache."""
    loader = _ModelLoader(settings)

    def run_match(
        source: np.ndarray | None,
        target: np.ndarray | None,
        grid_cols: float,
        grid_rows: float,
        max_points: float,
        query_chunk_size: float,
        visibility_threshold: float,
    ) -> tuple[np.ndarray | None, list[list[float]], str]:
        try:
            source_rgb = _as_rgb(source, "the source frame")
            target_rgb = _as_rgb(target, "the target frame")
            model, config = loader.get()
            matches, query_count = infer_two_frame_matches(
                model=model,
                config=config,
                source_rgb=source_rgb,
                target_rgb=target_rgb,
                grid_cols=int(grid_cols),
                grid_rows=int(grid_rows),
                max_points=int(max_points),
                query_chunk_size=int(query_chunk_size),
                visibility_threshold=float(visibility_threshold),
            )
            overlay = render_match_overlay(source_rgb, target_rgb, matches)
            aspect_warning = ""
            source_aspect = source_rgb.shape[1] / source_rgb.shape[0]
            target_aspect = target_rgb.shape[1] / target_rgb.shape[0]
            if not np.isclose(source_aspect, target_aspect, rtol=0.01, atol=0.01):
                aspect_warning = (
                    "\n\n> Note: the input aspect ratios differ. Each image was resized "
                    "independently for the model; frames from one video are recommended."
                )
            summary = (
                f"### {len(matches.source_xy)} / {query_count} correspondences shown\n\n"
                f"Visibility threshold: `{float(visibility_threshold):.2f}`. "
                "Coordinates in the table are pixels in the original uploads."
                f"{aspect_warning}"
            )
            return overlay, match_rows(matches), summary
        except Exception as exc:  # Surface setup/inference failures in the local UI.
            return None, [], f"### Matching failed\n\n`{type(exc).__name__}: {exc}`"

    return run_match


def build_ui(settings: AppSettings) -> gr.Blocks:
    """Construct the Gradio UI without loading the checkpoint."""
    run_match = _build_match_handler(settings)
    with gr.Blocks(title="OpenD4RT two-frame matching") as demo:
        gr.Markdown(
            "# OpenD4RT two-frame matching\n"
            "Upload two frames, then inspect OpenD4RT's sparse frame-0 → frame-1 "
            "correspondences. Best results come from nearby frames of the same video."
        )
        with gr.Row():
            source = gr.Image(
                label="Source frame (frame 0)", type="numpy", image_mode="RGB"
            )
            target = gr.Image(
                label="Target frame (frame 1)", type="numpy", image_mode="RGB"
            )
        with gr.Accordion("Matching settings", open=False):
            with gr.Row():
                grid_cols = gr.Slider(
                    2, 24, value=8, step=1, label="Source grid columns"
                )
                grid_rows = gr.Slider(2, 24, value=8, step=1, label="Source grid rows")
                max_points = gr.Slider(
                    4, 256, value=64, step=4, label="Maximum queries"
                )
            with gr.Row():
                query_chunk_size = gr.Slider(
                    1, 128, value=8, step=1, label="Queries per GPU decode chunk"
                )
                visibility_threshold = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05, label="Minimum visibility"
                )
        run_button = gr.Button("Match frames", variant="primary")
        summary = gr.Markdown(
            "Upload a source and target frame, then click **Match frames**."
        )
        overlay = gr.Image(label="OpenD4RT correspondences", interactive=False)
        table = gr.Dataframe(
            headers=[
                "source x",
                "source y",
                "target x",
                "target y",
                "visibility",
                "confidence",
            ],
            datatype=["number"] * 6,
            label="Matches in original image coordinates",
            interactive=False,
        )
        run_button.click(
            run_match,
            inputs=[
                source,
                target,
                grid_cols,
                grid_rows,
                max_points,
                query_chunk_size,
                visibility_threshold,
            ],
            outputs=[overlay, table, summary],
            concurrency_limit=1,
        )
    return demo


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve OpenD4RT two-frame matching in Gradio."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Model YAML config path."
    )
    parser.add_argument(
        "--ckpt-path", default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path."
    )
    parser.add_argument(
        "--device", default="cuda", help="cuda, cpu, or auto (default: cuda)."
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Bind address; localhost by default. Expose only on a trusted network.",
    )
    parser.add_argument("--server-port", type=_positive_int, default=7861)
    parser.add_argument(
        "--share", action="store_true", help="Create a temporary public Gradio link."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = AppSettings(
        config_path=Path(args.config),
        checkpoint_path=Path(args.ckpt_path),
        device=args.device,
    )
    build_ui(settings).queue(default_concurrency_limit=1).launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
