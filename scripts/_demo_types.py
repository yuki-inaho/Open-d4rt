"""Shared array shape contracts for the Viser demo package.

These ``jaxtyping`` aliases and the :class:`DemoPackage` ``TypedDict`` are the
single source of truth for the demo-package structure produced by
``vis.build_like_demo._export_demo_data`` and consumed by
``scripts.build_demo_from_video``. Keeping them in a dedicated module lets the
CLI script, its tests, and any future tooling share one definition instead of
re-declaring the shapes.
"""

from __future__ import annotations

from typing import Any, TypeAlias, TypedDict

import numpy as np
from jaxtyping import Bool, Float, Int, UInt8
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
UInt8Array: TypeAlias = NDArray[np.uint8]

VideoRGB: TypeAlias = UInt8[UInt8Array, "frames height width 3"]
UvPointArray: TypeAlias = Float[FloatArray, "points 2"]
UvTrackArray: TypeAlias = Float[FloatArray, "tracks 2"]
TrackFrameUvArray: TypeAlias = Float[FloatArray, "tracks frames 2"]
PointFrameUvArray: TypeAlias = Float[FloatArray, "frames points 2"]
TrackXyzArray: TypeAlias = Float[FloatArray, "tracks frames 3"]
PointXyzArray: TypeAlias = Float[FloatArray, "frames points 3"]
TrackVisibilityArray: TypeAlias = Bool[BoolArray, "tracks frames"]
PointVisibilityArray: TypeAlias = Bool[BoolArray, "frames points"]
TrackConfidenceArray: TypeAlias = Float[FloatArray, "tracks frames"]
PointConfidenceArray: TypeAlias = Float[FloatArray, "frames points"]
PointMotionScoreArray: TypeAlias = Float[FloatArray, "points"]  # noqa: F821
PointDynamicMaskArray: TypeAlias = Bool[BoolArray, "points"]  # noqa: F821
PointRgbArray: TypeAlias = UInt8[UInt8Array, "frames points 3"]
TrackQueryTimeArray: TypeAlias = Int[IntArray, "tracks"]  # noqa: F821
BoundsVector: TypeAlias = Float[FloatArray, "3"]
BoundsRadius: TypeAlias = Float[FloatArray, "1"]
IntrinsicsMatrix: TypeAlias = Float[FloatArray, "3 3"]


class DemoPackage(TypedDict):
    num_frames: int
    video_width: int
    video_height: int
    clip_frames: int
    track_stitch_diagnostics: dict[str, Any]
    track_query_uv_px: UvTrackArray
    track_query_t_src: TrackQueryTimeArray
    track_xyz_ref0: TrackXyzArray
    track_uv_px: TrackFrameUvArray
    track_visibility: TrackVisibilityArray
    track_confidence: TrackConfidenceArray
    point_query_uv_px: UvPointArray
    point_xyz_ref0: PointXyzArray
    point_visibility: PointVisibilityArray
    point_uv_px: PointFrameUvArray
    point_confidence: PointConfidenceArray
    point_motion_score: PointMotionScoreArray
    point_is_dynamic: PointDynamicMaskArray
    point_rgb: PointRgbArray
    bounds_min: BoundsVector
    bounds_max: BoundsVector
    bounds_center: BoundsVector
    bounds_radius: BoundsRadius
    ref0_K: IntrinsicsMatrix


__all__ = [
    "FloatArray",
    "IntArray",
    "BoolArray",
    "UInt8Array",
    "VideoRGB",
    "UvPointArray",
    "UvTrackArray",
    "TrackFrameUvArray",
    "PointFrameUvArray",
    "TrackXyzArray",
    "PointXyzArray",
    "TrackVisibilityArray",
    "PointVisibilityArray",
    "TrackConfidenceArray",
    "PointConfidenceArray",
    "PointMotionScoreArray",
    "PointDynamicMaskArray",
    "PointRgbArray",
    "TrackQueryTimeArray",
    "BoundsVector",
    "BoundsRadius",
    "IntrinsicsMatrix",
    "DemoPackage",
]
