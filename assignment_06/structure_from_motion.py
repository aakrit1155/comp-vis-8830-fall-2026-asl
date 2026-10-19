from __future__ import annotations

from typing import Any, Sequence, cast

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from .media_utils import load_image_bytes

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MIN_POINT_COUNT = 8
DEFAULT_POINT_COUNT = 8
POINT_DISPLAY_WIDTH = 700
CORNER_LABELS = ["TL", "TR", "BR", "BL"]


# ============================================================
# Small helpers
# ============================================================


def _point_label(index: int) -> str:
    if index < 4:
        return CORNER_LABELS[index]
    return f"F{index - 3}"


def _matrix_text(matrix: np.ndarray) -> str:
    return np.array2string(
        np.asarray(matrix),
        precision=5,
        suppress_small=True,
    )


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(
        src,
        np.asarray(homography, dtype=np.float64),
    )
    return transformed.reshape(-1, 2)


def _point_errors(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed_f = np.asarray(observed, dtype=np.float64)
    predicted_f = np.asarray(predicted, dtype=np.float64)
    return np.linalg.norm(observed_f - predicted_f, axis=1)


def _reprojection_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    errors = _point_errors(observed, predicted)
    if len(errors) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors**2)))


def _safe_inverse(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix_f = np.asarray(matrix, dtype=np.float64)
    if abs(float(np.linalg.det(matrix_f))) < 1e-12:
        raise ValueError(f"{name} is singular and cannot be inverted.")
    return np.linalg.inv(matrix_f)


def _polygon_width_height(corners: np.ndarray) -> tuple[float, float]:
    quad = np.asarray(corners, dtype=np.float64)
    top_width = np.linalg.norm(quad[1] - quad[0])
    bottom_width = np.linalg.norm(quad[2] - quad[3])
    left_height = np.linalg.norm(quad[3] - quad[0])
    right_height = np.linalg.norm(quad[2] - quad[1])
    return (
        float((top_width + bottom_width) / 2.0),
        float((left_height + right_height) / 2.0),
    )


# ============================================================
# Camera geometry
# ============================================================


def _camera_matrix_from_hfov(
    width: int,
    height: int,
    hfov_degrees: float,
) -> np.ndarray:
    hfov = np.deg2rad(hfov_degrees)
    fx = width / (2.0 * np.tan(hfov / 2.0))
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _scale_intrinsics(
    calibration_k: np.ndarray,
    calibration_width: int,
    calibration_height: int,
    processed_width: int,
    processed_height: int,
) -> np.ndarray:
    sx = processed_width / float(calibration_width)
    sy = processed_height / float(calibration_height)

    scale_matrix = np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scale_matrix @ np.asarray(calibration_k, dtype=np.float64)


def _pose_from_plane_homography(
    homography: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For the planar world Z = 0:

        H ~ K [r1 r2 t]

    The physical dimensions of the planar object provide metric scale.
    Camera center is C = -R^T t.
    """
    normalized = _safe_inverse(k, "Camera intrinsic matrix K") @ homography

    b1 = normalized[:, 0]
    b2 = normalized[:, 1]
    b3 = normalized[:, 2]

    denominator = np.linalg.norm(b1) + np.linalg.norm(b2)
    if denominator < 1e-12:
        raise ValueError("Degenerate homography: camera pose cannot be recovered.")

    scale = 2.0 / denominator
    if scale * b3[2] < 0:
        scale = -scale

    r1 = scale * b1
    r2 = scale * b2
    t = scale * b3
    r3 = np.cross(r1, r2)

    r_approx = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(r_approx)
    r = u @ vt

    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    camera_center = -r.T @ t
    return r, t, camera_center


def _project_plane_points(
    world_xy: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    plane_projection = k @ np.column_stack([r[:, 0], r[:, 1], t])
    homogeneous = np.column_stack(
        [
            np.asarray(world_xy, dtype=np.float64),
            np.ones(len(world_xy), dtype=np.float64),
        ]
    )
    image_h = (plane_projection @ homogeneous.T).T

    if np.any(np.abs(image_h[:, 2]) < 1e-12):
        raise ValueError("A projected point lies at infinity.")

    return image_h[:, :2] / image_h[:, 2:3]


# ============================================================
# Homography / plane geometry
# ============================================================


def _fit_homography(
    world_points: np.ndarray,
    image_points: np.ndarray,
    *,
    ransac_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit plane -> image homography.

    Four points use the exact four-point projective transform.
    More than four use RANSAC for robustness.
    """
    world = np.asarray(world_points, dtype=np.float32)
    image = np.asarray(image_points, dtype=np.float32)

    if len(world) != len(image):
        raise ValueError("World and image correspondence counts do not match.")
    if len(world) < 4:
        raise ValueError("At least four planar correspondences are required.")

    if len(world) == 4:
        homography = cv2.getPerspectiveTransform(world, image)
        return homography, np.ones(4, dtype=bool)

    homography, mask = cv2.findHomography(
        world,
        image,
        cv2.RANSAC,
        ransac_threshold,
    )

    if homography is None or mask is None:
        raise ValueError(
            "Homography estimation failed. Check that point numbers refer to "
            "the same physical locations in every image."
        )

    return homography, mask.reshape(-1).astype(bool)


def _make_object_mask(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    polygon = np.round(np.asarray(corners, dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


# ============================================================
# Drawing helpers
# ============================================================


def _draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
    gap_length: int = 7,
) -> None:
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return

    direction = delta / length
    step = dash_length + gap_length
    distance = 0.0

    while distance < length:
        dash_start = p0 + direction * distance
        dash_end = p0 + direction * min(distance + dash_length, length)
        cv2.line(
            image,
            tuple(np.round(dash_start).astype(int)),
            tuple(np.round(dash_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        distance += step


def _draw_dashed_polygon(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    polygon = np.round(np.asarray(points, dtype=np.float64)).astype(int)
    for index in range(len(polygon)):
        p0 = tuple(polygon[index])
        p1 = tuple(polygon[(index + 1) % len(polygon)])
        _draw_dashed_line(image, p0, p1, color, thickness=thickness)


def _annotate_selected_points(
    rgb: np.ndarray,
    points: Sequence[Sequence[float]],
) -> np.ndarray:
    vis = rgb.copy()

    if len(points) >= 4:
        corners = np.asarray(points[:4], dtype=np.float32)
        cv2.polylines(
            vis,
            [np.round(corners).astype(np.int32)],
            True,
            (50, 220, 90),
            3,
            cv2.LINE_AA,
        )

    for index, point in enumerate(points):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))

        cv2.circle(vis, (x, y), 6, (245, 75, 75), -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 7, (255, 255, 255), 1, cv2.LINE_AA)

        label = f"{index + 1}:{_point_label(index)}"
        cv2.putText(
            vis,
            label,
            (x + 8, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (x + 8, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


def _feature_validation_overlay(
    rgb: np.ndarray,
    corners: np.ndarray,
    observed_features: np.ndarray,
    predicted_features: np.ndarray,
) -> np.ndarray:
    """
    Corner-only homography predicts held-out interior features.

    Green circle: observed feature.
    Magenta cross: predicted feature.
    Yellow segment: residual.
    """
    vis = rgb.copy()
    cv2.polylines(
        vis,
        [np.round(corners).astype(np.int32)],
        True,
        (90, 200, 120),
        2,
        cv2.LINE_AA,
    )

    for offset, (actual, predicted) in enumerate(
        zip(observed_features, predicted_features),
        start=5,
    ):
        ax, ay = np.round(actual).astype(int)
        px, py = np.round(predicted).astype(int)

        cv2.line(
            vis, (int(ax), int(ay)), (int(px), int(py)), (245, 190, 45), 2, cv2.LINE_AA
        )
        cv2.circle(vis, (int(ax), int(ay)), 5, (45, 200, 95), -1, cv2.LINE_AA)
        cv2.drawMarker(
            vis,
            (int(px), int(py)),
            (225, 65, 220),
            markerType=cv2.MARKER_CROSS,
            markerSize=13,
            thickness=2,
        )
        cv2.putText(
            vis,
            f"F{offset - 4}",
            (int(ax) + 7, int(ay) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


def _boundary_validation_overlay(
    rgb: np.ndarray,
    observed_corners: np.ndarray,
    predicted_corners: np.ndarray,
    interior_features: np.ndarray,
) -> np.ndarray:
    """Display observed boundary and held-out corner predictions."""
    vis = rgb.copy()
    observed = np.round(observed_corners).astype(np.int32)
    predicted = np.round(predicted_corners).astype(np.int32)
    cv2.polylines(vis, [observed], True, (45, 205, 100), 3, cv2.LINE_AA)
    _draw_dashed_polygon(vis, predicted, (240, 125, 45), thickness=3)
    for index, (actual, estimate) in enumerate(
        zip(observed_corners, predicted_corners)
    ):
        ax, ay = np.round(actual).astype(int)
        ex, ey = np.round(estimate).astype(int)
        cv2.line(
            vis, (int(ax), int(ay)), (int(ex), int(ey)), (245, 200, 60), 2, cv2.LINE_AA
        )
        cv2.circle(vis, (int(ax), int(ay)), 5, (45, 205, 100), -1, cv2.LINE_AA)
        cv2.drawMarker(
            vis,
            (int(ex), int(ey)),
            (240, 125, 45),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=14,
            thickness=2,
        )
        cv2.putText(
            vis,
            CORNER_LABELS[index],
            (int(ax) + 7, int(ay) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for point in interior_features:
        x, y = np.round(point).astype(int)
        cv2.circle(vis, (int(x), int(y)), 3, (80, 150, 245), -1, cv2.LINE_AA)
    return vis


# ============================================================
# ORB / RANSAC independent evidence
# ============================================================


def _draw_match_pairs(
    ref_rgb: np.ndarray,
    target_rgb: np.ndarray,
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> np.ndarray:
    h1, w1 = ref_rgb.shape[:2]
    h2, w2 = target_rgb.shape[:2]

    canvas = np.full((max(h1, h2), w1 + w2, 3), 245, dtype=np.uint8)
    canvas[:h1, :w1] = ref_rgb
    canvas[:h2, w1 : w1 + w2] = target_rgb

    for match in matches:
        p1 = kp1[int(match.queryIdx)].pt
        p2 = kp2[int(match.trainIdx)].pt

        x1 = int(round(float(p1[0])))
        y1 = int(round(float(p1[1])))
        x2 = int(round(float(p2[0]))) + w1
        y2 = int(round(float(p2[1])))

        cv2.circle(canvas, (x1, y1), 3, (35, 210, 80), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2, y2), 3, (35, 210, 80), -1, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x2, y2), (40, 170, 230), 1, cv2.LINE_AA)

    return canvas


def _orb_view_consistency(
    ref_rgb: np.ndarray,
    target_rgb: np.ndarray,
    ref_mask: np.ndarray,
    target_mask: np.ndarray,
    ref_quad: np.ndarray,
    target_quad: np.ndarray,
) -> dict[str, Any]:
    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)

    orb: cv2.ORB = cv2.ORB.create(1500)
    kp1_raw, des1 = orb.detectAndCompute(ref_gray, ref_mask)
    kp2_raw, des2 = orb.detectAndCompute(target_gray, target_mask)

    if kp1_raw is None or kp2_raw is None or des1 is None or des2 is None:
        return {
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "corner_rmse": np.nan,
            "visualization": None,
        }

    kp1 = cast(Sequence[cv2.KeyPoint], kp1_raw)
    kp2 = cast(Sequence[cv2.KeyPoint], kp2_raw)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    good_matches: list[cv2.DMatch] = []
    for raw_pair in knn_matches:
        pair = cast(Sequence[cv2.DMatch], raw_pair)
        if len(pair) < 2:
            continue
        m = pair[0]
        n = pair[1]
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return {
            "matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "corner_rmse": np.nan,
            "visualization": None,
        }

    src = np.asarray(
        [kp1[int(match.queryIdx)].pt for match in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    dst = np.asarray(
        [kp2[int(match.trainIdx)].pt for match in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None:
        return {
            "matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "corner_rmse": np.nan,
            "visualization": None,
        }

    flags = inlier_mask.reshape(-1).astype(bool)
    inlier_matches: list[cv2.DMatch] = [
        match for match, keep in zip(good_matches, flags) if bool(keep)
    ]

    predicted_quad = cv2.perspectiveTransform(
        np.asarray(ref_quad, dtype=np.float32).reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)

    corner_rmse = _reprojection_rmse(
        np.asarray(target_quad, dtype=np.float32),
        predicted_quad,
    )

    visualization = _draw_match_pairs(
        ref_rgb,
        target_rgb,
        kp1,
        kp2,
        inlier_matches[:40],
    )

    return {
        "matches": len(good_matches),
        "inliers": len(inlier_matches),
        "inlier_ratio": (
            len(inlier_matches) / len(good_matches) if good_matches else 0.0
        ),
        "corner_rmse": corner_rmse,
        "visualization": visualization,
    }


# ============================================================
# Reconstruction visualization
# ============================================================


def _render_world_reconstruction(
    reconstructed_sets: list[np.ndarray],
    reconstructed_mean: np.ndarray,
    object_width: float,
    object_height: float,
) -> np.ndarray:
    canvas_width = 980
    canvas_height = 680
    margin = 85

    canvas = np.full((canvas_height, canvas_width, 3), 248, dtype=np.uint8)

    scale = min(
        (canvas_width - 2 * margin) / max(object_width, 1e-9),
        (canvas_height - 2 * margin) / max(object_height, 1e-9),
    )

    def map_point(point: np.ndarray) -> tuple[int, int]:
        x = margin + float(point[0]) * scale
        y = margin + float(point[1]) * scale
        return int(round(x)), int(round(y))

    true_boundary = np.array(
        [
            [0.0, 0.0],
            [object_width, 0.0],
            [object_width, object_height],
            [0.0, object_height],
        ],
        dtype=np.float32,
    )
    true_pixels = np.asarray(
        [map_point(point) for point in true_boundary], dtype=np.int32
    )
    cv2.fillPoly(canvas, [true_pixels], (238, 238, 238))
    cv2.polylines(canvas, [true_pixels], True, (75, 75, 75), 3, cv2.LINE_AA)

    palette = [
        (37, 99, 235),
        (124, 58, 237),
        (5, 150, 105),
        (234, 88, 12),
    ]

    for view_index, points in enumerate(reconstructed_sets):
        color = palette[view_index % len(palette)]
        point_pixels = np.asarray(
            [map_point(point) for point in points[:4]], dtype=np.int32
        )
        _draw_dashed_polygon(canvas, point_pixels, color, thickness=2)
        for point in points:
            x, y = map_point(point)
            cv2.circle(canvas, (x, y), 4, color, -1, cv2.LINE_AA)

    mean_corners = reconstructed_mean[:4]
    mean_corner_pixels = np.asarray(
        [map_point(point) for point in mean_corners], dtype=np.int32
    )
    cv2.polylines(canvas, [mean_corner_pixels], True, (215, 45, 45), 4, cv2.LINE_AA)

    for index, point in enumerate(reconstructed_mean):
        x, y = map_point(point)
        cv2.circle(canvas, (x, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 8, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{index + 1}:{_point_label(index)}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Top-down reconstruction in the object plane",
        (28, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Gray = known boundary | dashed colored polygons / dots = back-projected points from Views 1-4",
        (28, canvas_height - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Red = mean reconstructed boundary | black outlined circles = mean reconstructed points",
        (28, canvas_height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )

    legend_x = canvas_width - 200
    legend_y = 70
    for view_index, color in enumerate(palette, start=1):
        cy = legend_y + 24 * (view_index - 1)
        cv2.circle(canvas, (legend_x, cy), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"View {view_index}",
            (legend_x + 14, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )

    return canvas


def _rectify_planar_view(
    rgb: np.ndarray,
    plane_to_image_h: np.ndarray,
    object_width_mm: float,
    object_height_mm: float,
    world_points: np.ndarray,
) -> np.ndarray:
    """Perspective-rectify the observed object surface into its metric XY plane.

    This removes *projective perspective*, not lens radial distortion.
    The homography is the existing all-correspondence fit (H_all).
    """
    max_dimension = 760
    margin = 24
    scale = max_dimension / max(object_width_mm, object_height_mm)
    output_width = max(1, int(round(object_width_mm * scale)))
    output_height = max(1, int(round(object_height_mm * scale)))

    world_to_canvas = np.array(
        [[scale, 0.0, margin], [0.0, scale, margin], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    image_to_canvas = world_to_canvas @ _safe_inverse(
        plane_to_image_h, "All-point homography"
    )
    canvas = cv2.warpPerspective(
        rgb,
        image_to_canvas,
        (output_width + 2 * margin, output_height + 2 * margin),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(247, 247, 247),
    )

    # Use the canonical metric rectangle, not a warped background polygon.
    corners = np.array(
        [
            [0, 0],
            [object_width_mm, 0],
            [object_width_mm, object_height_mm],
            [0, object_height_mm],
        ],
        dtype=np.float32,
    )
    canvas_corners = _transform_points(corners, world_to_canvas)
    cv2.polylines(
        canvas,
        [np.round(canvas_corners).astype(np.int32)],
        True,
        (38, 148, 238),
        2,
        cv2.LINE_AA,
    )
    # Show the recovered interior feature locations on the actual book texture.
    recovered = _transform_points(world_points, world_to_canvas)
    for index, point in enumerate(recovered):
        x, y = np.round(point).astype(int)
        cv2.circle(canvas, (int(x), int(y)), 5, (14, 172, 108), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), 6, (255, 255, 255), 1, cv2.LINE_AA)
        label = _point_label(index)
        cv2.putText(
            canvas,
            label,
            (int(x) + 7, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (int(x) + 7, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _render_camera_positions(
    camera_centers: list[np.ndarray],
    object_width: float,
) -> np.ndarray:
    """Render X-Z side view; the object plane lies at Z = 0."""
    canvas_width = 760
    canvas_height = 480
    margin = 70

    centers = np.vstack(camera_centers)
    x_values = np.concatenate([centers[:, 0], np.array([0.0, object_width])])
    z_values = np.concatenate([centers[:, 2], np.array([0.0])])

    xmin = float(np.min(x_values))
    xmax = float(np.max(x_values))
    zmin = float(np.min(z_values))
    zmax = float(np.max(z_values))

    x_pad = max(20.0, 0.15 * max(xmax - xmin, object_width, 1.0))
    z_pad = max(20.0, 0.15 * max(zmax - zmin, 1.0))

    xmin -= x_pad
    xmax += x_pad
    zmin -= z_pad
    zmax += z_pad

    canvas = np.full((canvas_height, canvas_width, 3), 248, dtype=np.uint8)

    def map_xz(x: float, z: float) -> tuple[int, int]:
        px = margin + (x - xmin) / max(xmax - xmin, 1e-9) * (canvas_width - 2 * margin)
        py = (
            canvas_height
            - margin
            - (z - zmin) / max(zmax - zmin, 1e-9) * (canvas_height - 2 * margin)
        )
        return int(round(px)), int(round(py))

    object_start = map_xz(0.0, 0.0)
    object_end = map_xz(object_width, 0.0)
    cv2.line(canvas, object_start, object_end, (60, 60, 60), 5, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Object plane",
        (min(object_start[0], object_end[0]), object_start[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    for index, center in enumerate(camera_centers, start=1):
        point = map_xz(float(center[0]), float(center[2]))
        cv2.circle(canvas, point, 7, (42, 82, 152), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"V{index}",
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (42, 82, 152),
            2,
            cv2.LINE_AA,
        )

    return canvas


# ============================================================
# Point-selection session state
# ============================================================


def _bump_point_component_nonce() -> None:
    st.session_state["sfm_point_component_nonce"] = (
        st.session_state.get("sfm_point_component_nonce", 0) + 1
    )


def _initialize_manual_point_state(signature: tuple[Any, ...]) -> None:
    if st.session_state.get("sfm_point_signature") == signature:
        return

    st.session_state["sfm_point_signature"] = signature
    st.session_state["sfm_manual_points"] = [[], [], [], []]
    st.session_state["sfm_last_click_event"] = [None, None, None, None]
    _bump_point_component_nonce()
    st.session_state.pop("sfm_results", None)


def _clear_all_manual_points() -> None:
    st.session_state["sfm_manual_points"] = [[], [], [], []]
    st.session_state["sfm_last_click_event"] = [None, None, None, None]
    _bump_point_component_nonce()
    st.session_state.pop("sfm_results", None)


def _clear_view_points(view_index: int) -> None:
    raw_points = st.session_state.get("sfm_manual_points", [[], [], [], []])
    points_by_view = [list(view) for view in raw_points]
    points_by_view[view_index] = []
    st.session_state["sfm_manual_points"] = points_by_view

    raw_events = st.session_state.get(
        "sfm_last_click_event",
        [None, None, None, None],
    )
    events = list(raw_events)
    events[view_index] = None
    st.session_state["sfm_last_click_event"] = events

    _bump_point_component_nonce()
    st.session_state.pop("sfm_results", None)


def _undo_view_point(view_index: int) -> None:
    raw_points = st.session_state.get("sfm_manual_points", [[], [], [], []])
    points_by_view = [list(view) for view in raw_points]

    if points_by_view[view_index]:
        points_by_view[view_index].pop()

    st.session_state["sfm_manual_points"] = points_by_view

    raw_events = st.session_state.get(
        "sfm_last_click_event",
        [None, None, None, None],
    )
    events = list(raw_events)
    events[view_index] = None
    st.session_state["sfm_last_click_event"] = events

    _bump_point_component_nonce()
    st.session_state.pop("sfm_results", None)


def _consume_image_click(
    view_index: int,
    click_value: dict[str, Any] | None,
    image: np.ndarray,
    requested_display_width: int,
    maximum_points: int,
) -> bool:
    if click_value is None or not isinstance(click_value, dict):
        return False

    raw_x = click_value.get("x")
    raw_y = click_value.get("y")

    if not isinstance(raw_x, (int, float)) or not isinstance(raw_y, (int, float)):
        return False

    raw_time = click_value.get("unix_time", 0)
    unix_time = int(raw_time) if isinstance(raw_time, (int, float)) else 0

    event_signature = (
        unix_time,
        round(float(raw_x), 4),
        round(float(raw_y), 4),
    )

    raw_events = st.session_state.get(
        "sfm_last_click_event",
        [None, None, None, None],
    )
    events = list(raw_events)

    if events[view_index] == event_signature:
        return False

    events[view_index] = event_signature
    st.session_state["sfm_last_click_event"] = events

    raw_points = st.session_state.get("sfm_manual_points", [[], [], [], []])
    points_by_view = [list(view) for view in raw_points]
    points = points_by_view[view_index]

    if len(points) >= maximum_points:
        return False

    image_height, image_width = image.shape[:2]
    expected_height = max(
        1,
        int(round(requested_display_width * image_height / image_width)),
    )

    returned_width = click_value.get("width", requested_display_width)
    returned_height = click_value.get("height", expected_height)

    rendered_width = (
        float(returned_width)
        if isinstance(returned_width, (int, float)) and float(returned_width) > 0
        else float(requested_display_width)
    )
    rendered_height = (
        float(returned_height)
        if isinstance(returned_height, (int, float)) and float(returned_height) > 0
        else float(expected_height)
    )

    x = float(raw_x) / rendered_width * image_width
    y = float(raw_y) / rendered_height * image_height

    x = float(np.clip(x, 0.0, image_width - 1.0))
    y = float(np.clip(y, 0.0, image_height - 1.0))

    points.append([x, y])
    points_by_view[view_index] = points
    st.session_state["sfm_manual_points"] = points_by_view
    st.session_state.pop("sfm_results", None)
    return True


# ============================================================
# Main planar SfM processing
# ============================================================


def process_planar_sfm(
    images: list[np.ndarray],
    selected_points: list[np.ndarray],
    object_width_mm: float,
    object_height_mm: float,
    intrinsic_mode: str,
    hfov_degrees: float | None = None,
    calibration_k: np.ndarray | None = None,
    calibration_resolution: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if len(images) != 4:
        raise ValueError("Exactly four images are required.")
    if len(selected_points) != 4:
        raise ValueError("Exactly four correspondence sets are required.")

    point_counts = [len(points) for points in selected_points]
    if len(set(point_counts)) != 1:
        raise ValueError("Each image must contain the same number of correspondences.")
    if point_counts[0] < MIN_POINT_COUNT:
        raise ValueError(
            f"At least {MIN_POINT_COUNT} correspondences are required: "
            "4 boundary corners plus at least 4 interior features."
        )

    world_corners = np.array(
        [
            [0.0, 0.0],
            [object_width_mm, 0.0],
            [object_width_mm, object_height_mm],
            [0.0, object_height_mm],
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Reference-plane coordinate system
    # --------------------------------------------------------
    # View 1 defines the metric plane coordinates of all selected points.
    reference_points = np.asarray(selected_points[0], dtype=np.float32)
    reference_corner_h, _ = _fit_homography(world_corners, reference_points[:4])
    reference_inverse = _safe_inverse(reference_corner_h, "Reference corner homography")
    world_points = _transform_points(reference_points, reference_inverse)
    world_points[:4] = world_corners
    feature_world_points = np.asarray(world_points[4:], dtype=np.float32)

    view_results: list[dict[str, Any]] = []
    reconstructed_sets: list[np.ndarray] = []

    for view_index, (rgb, raw_image_points) in enumerate(zip(images, selected_points)):
        image_points = np.asarray(raw_image_points, dtype=np.float32)
        image_height, image_width = rgb.shape[:2]

        if intrinsic_mode == "hfov":
            if hfov_degrees is None:
                raise ValueError("Horizontal field of view is required.")
            k = _camera_matrix_from_hfov(
                width=image_width,
                height=image_height,
                hfov_degrees=hfov_degrees,
            )
        else:
            if calibration_k is None or calibration_resolution is None:
                raise ValueError("Calibrated camera parameters are required.")
            calibration_width, calibration_height = calibration_resolution
            k = _scale_intrinsics(
                calibration_k=calibration_k,
                calibration_width=calibration_width,
                calibration_height=calibration_height,
                processed_width=image_width,
                processed_height=image_height,
            )

        # Homography using all selected points: this is the main planar reconstruction.
        h_all, all_inlier_mask = _fit_homography(world_points, image_points)
        all_projection = _transform_points(world_points, h_all)
        homography_rmse = _reprojection_rmse(image_points, all_projection)

        # Corner-only homography predicting held-out interior features.
        h_corners, _ = _fit_homography(world_corners, image_points[:4])
        predicted_features = _transform_points(feature_world_points, h_corners)
        feature_errors_px = _point_errors(image_points[4:], predicted_features)
        feature_rmse_px = _reprojection_rmse(image_points[4:], predicted_features)

        # Interior-only homography predicts corners that were excluded from its fit.
        # Extrapolation can be unstable if features are clustered or collinear.
        h_features, feature_inlier_mask = _fit_homography(
            feature_world_points, image_points[4:]
        )
        predicted_corners = _transform_points(world_corners, h_features)
        boundary_errors_px = _point_errors(image_points[:4], predicted_corners)
        boundary_rmse_px = _reprojection_rmse(image_points[:4], predicted_corners)

        # Back-project the image points into the common metric plane.
        reconstructed_world = _transform_points(
            image_points,
            _safe_inverse(h_all, "All-point homography"),
        )
        reconstructed_sets.append(reconstructed_world)

        r, t, camera_center = _pose_from_plane_homography(h_all, k)
        pose_projection = _project_plane_points(world_points, k, r, t)
        pose_rmse = _reprojection_rmse(image_points, pose_projection)

        object_mask = _make_object_mask(rgb, image_points[:4])
        feature_overlay = _feature_validation_overlay(
            rgb,
            image_points[:4],
            image_points[4:],
            predicted_features,
        )

        boundary_overlay = _boundary_validation_overlay(
            rgb, image_points[:4], predicted_corners, image_points[4:]
        )

        view_results.append(
            {
                "image": rgb,
                "points": image_points,
                "mask": object_mask,
                "K": k,
                "H_all": h_all,
                "H_corners": h_corners,
                "H_features": h_features,
                "feature_inlier_mask": feature_inlier_mask,
                "feature_inliers": int(np.sum(feature_inlier_mask)),
                "predicted_corners": predicted_corners,
                "boundary_errors_px": boundary_errors_px,
                "boundary_rmse_px": boundary_rmse_px,
                "boundary_overlay": boundary_overlay,
                "R": r,
                "t": t,
                "C": camera_center,
                "all_inlier_mask": all_inlier_mask,
                "all_inliers": int(np.sum(all_inlier_mask)),
                "homography_rmse": homography_rmse,
                "pose_rmse": pose_rmse,
                "predicted_features": predicted_features,
                "feature_errors_px": feature_errors_px,
                "feature_rmse_px": feature_rmse_px,
                "reconstructed_points": reconstructed_world,
                "feature_overlay": feature_overlay,
                "is_reference_view": view_index == 0,
            }
        )

    # Views 2-4 are the actual held-out validation views because View 1 defined
    # the reference plane coordinates.
    target_views = view_results[1:]
    feature_validation_rmse_target = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        np.asarray(view["feature_errors_px"], dtype=np.float64) ** 2
                        for view in target_views
                    ]
                )
            )
        )
    )

    boundary_validation_rmse_target = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        np.asarray(view["boundary_errors_px"], dtype=np.float64) ** 2
                        for view in target_views
                    ]
                )
            )
        )
    )

    reconstructed_stack = np.stack(reconstructed_sets, axis=0)
    reconstructed_mean = np.mean(reconstructed_stack, axis=0)
    deviations = reconstructed_stack - reconstructed_mean[None, :, :]
    point_consistency_rmse = float(np.sqrt(np.mean(np.sum(deviations**2, axis=2))))

    reconstructed_corners = reconstructed_mean[:4]
    reconstructed_width, reconstructed_height = _polygon_width_height(
        reconstructed_corners
    )
    width_error_percent = float(
        abs(reconstructed_width - object_width_mm) / object_width_mm * 100.0
    )
    height_error_percent = float(
        abs(reconstructed_height - object_height_mm) / object_height_mm * 100.0
    )

    reconstruction_image = _render_world_reconstruction(
        reconstructed_sets,
        reconstructed_mean,
        object_width_mm,
        object_height_mm,
    )

    # Each camera view is rectified independently into the same metric XY frame.
    rectified_views = [
        _rectify_planar_view(
            np.asarray(view["image"], dtype=np.uint8),
            np.asarray(view["H_all"], dtype=np.float64),
            object_width_mm,
            object_height_mm,
            world_points,
        )
        for view in view_results
    ]

    camera_centers = [np.asarray(view["C"], dtype=np.float64) for view in view_results]
    camera_map = _render_camera_positions(camera_centers, object_width_mm)

    orb_results: list[dict[str, Any]] = []
    for target_index in range(1, 4):
        orb_results.append(
            _orb_view_consistency(
                images[0],
                images[target_index],
                cast(np.ndarray, view_results[0]["mask"]),
                cast(np.ndarray, view_results[target_index]["mask"]),
                np.asarray(selected_points[0][:4], dtype=np.float32),
                np.asarray(selected_points[target_index][:4], dtype=np.float32),
            )
        )

    return {
        "views": view_results,
        "world_points": world_points,
        "reconstructed_sets": reconstructed_sets,
        "reconstructed_mean": reconstructed_mean,
        "reconstruction_image": reconstruction_image,
        "rectified_views": rectified_views,
        "point_consistency_rmse": point_consistency_rmse,
        "reconstructed_width": reconstructed_width,
        "reconstructed_height": reconstructed_height,
        "width_error_percent": width_error_percent,
        "height_error_percent": height_error_percent,
        "feature_validation_rmse_target": feature_validation_rmse_target,
        "boundary_validation_rmse_target": boundary_validation_rmse_target,
        "camera_map": camera_map,
        "orb_results": orb_results,
    }


# ============================================================
# Streamlit UI
# ============================================================


def _reset_sfm() -> None:
    st.session_state["sfm_nonce"] = st.session_state.get("sfm_nonce", 0) + 1

    for key in (
        "sfm_results",
        "sfm_manual_points",
        "sfm_last_click_event",
        "sfm_point_signature",
        "sfm_active_view",
    ):
        st.session_state.pop(key, None)

    _bump_point_component_nonce()


def render_structure_from_motion() -> None:
    st.markdown(
        """
<div class="cv-task-hero">
    <h2>Question 2 — Planar Structure from Motion</h2>
    <p>
        Reconstruct a planar object from four viewpoints using manually selected
        multi-view correspondences, recover an estimated camera pose for every
        image, and show the mathematical homography-based reconstruction.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="cv-input-guide">
<b>Recommended experiment.</b>
Use one textured planar object with known physical dimensions. Keep the object
stationary and photograph it from four viewpoints using the same camera and lens.
The first four clicks are the physical boundary corners. The remaining clicks are
repeatable interior features used to strengthen the homography and to provide a
simple held-out validation check.
</div>
        """,
        unsafe_allow_html=True,
    )

    _, reset_col = st.columns([5, 1])
    with reset_col:
        st.button(
            "↻ Reset",
            key="sfm_reset",
            on_click=_reset_sfm,
            width="stretch",
        )

    nonce = st.session_state.get("sfm_nonce", 0)

    # ========================================================
    # Upload
    # ========================================================
    uploads = st.file_uploader(
        "Upload exactly four viewpoints",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key=f"sfm_images_{nonce}",
        help="RGB and grayscale images are both supported.",
    )

    if not uploads:
        return

    if len(uploads) != 4:
        st.warning(
            f"Exactly four viewpoints are required. You selected {len(uploads)}."
        )
        return

    image_bytes = [upload.getvalue() for upload in uploads]
    if any(len(data) > MAX_IMAGE_BYTES for data in image_bytes):
        st.error("At least one image exceeds the internal 12 MB safety limit.")
        return

    preview_columns = st.columns(4)
    for index, (column, data) in enumerate(zip(preview_columns, image_bytes), start=1):
        with column:
            st.image(data, caption=f"View {index}", width="stretch")

    st.divider()

    # ========================================================
    # 1. Object geometry
    # ========================================================
    st.markdown("### 1. Object geometry")
    g1, g2, g3 = st.columns(3)

    with g1:
        object_width = st.number_input(
            "Object width (mm)",
            min_value=10.0,
            value=210.0,
            step=1.0,
            help="Enter the actual physical width of the planar surface.",
        )

    with g2:
        object_height = st.number_input(
            "Object height (mm)",
            min_value=10.0,
            value=297.0,
            step=1.0,
            help="Enter the actual physical height of the planar surface.",
        )

    with g3:
        max_side = st.select_slider(
            "Internal image max dimension",
            options=[800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200],
            value=1000,
            help=(
                "Maximum width or height used internally for OpenCV processing "
                "while preserving aspect ratio."
            ),
        )

    st.caption(
        "Measure exactly the planar surface whose four corners will be clicked. "
        "For a book, use the front-cover dimensions only."
    )

    try:
        images: list[np.ndarray] = []
        for data in image_bytes:
            rgb, _, _ = load_image_bytes(data, max_side=max_side)
            images.append(rgb)
    except Exception as exc:
        st.error(f"Unable to load the images: {exc}")
        return

    # ========================================================
    # 2. Manual correspondence selection
    # ========================================================
    st.markdown("### 2. Select matching object points")

    number_of_points = st.slider(
        "Correspondences per image",
        min_value=MIN_POINT_COUNT,
        max_value=12,
        value=DEFAULT_POINT_COUNT,
        step=1,
        help=(
            "Points 1-4 must be TL, TR, BR, BL. Points 5+ should be distinctive "
            "interior features such as sharp letter corners or logo intersections."
        ),
    )

    st.markdown(
        """
<div class="cv-result-note">
<b>Click order is critical.</b><br>
In every image select exactly the same physical points in exactly the same order:
<b>1 = TL, 2 = TR, 3 = BR, 4 = BL</b>. Then select the same distinctive interior
features as points 5, 6, 7, ... in every view.
</div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Spread the interior features across the book plane. Good choices are sharp "
        "letter corners, logo intersections, printed dots, and high-contrast corners. "
        "Avoid collinear or tightly clustered features."
    )

    selection_signature: tuple[Any, ...] = (
        tuple((upload.name, len(data)) for upload, data in zip(uploads, image_bytes)),
        max_side,
        number_of_points,
    )
    _initialize_manual_point_state(selection_signature)

    manual_points = st.session_state["sfm_manual_points"]

    progress_columns = st.columns(4)
    for index, column in enumerate(progress_columns):
        column.metric(
            f"View {index + 1}",
            f"{len(manual_points[index])}/{number_of_points}",
        )

    view_labels = ["View 1", "View 2", "View 3", "View 4"]
    active_view_label = st.radio(
        "Image to annotate",
        view_labels,
        horizontal=True,
        key="sfm_active_view",
    )
    active_view = view_labels.index(active_view_label)
    selected_for_view = manual_points[active_view]

    if len(selected_for_view) < number_of_points:
        next_index = len(selected_for_view)
        st.markdown(f"**Next point:** `{next_index + 1} — {_point_label(next_index)}`")
        if next_index < 4:
            st.caption("Click the corresponding physical corner of the planar object.")
        else:
            st.caption(
                "Click the same distinctive interior feature used for this point "
                "number in every other view."
            )
    else:
        st.success(f"View {active_view + 1} is complete.")

    annotated = _annotate_selected_points(images[active_view], selected_for_view)
    display_width = min(POINT_DISPLAY_WIDTH, int(images[active_view].shape[1]))
    component_nonce = st.session_state.get("sfm_point_component_nonce", 0)

    click_value = streamlit_image_coordinates(
        Image.fromarray(annotated),
        width=display_width,
        key=f"sfm_click_{nonce}_{component_nonce}_{active_view}",
        cursor="crosshair",
    )

    added = _consume_image_click(
        active_view,
        cast(dict[str, Any] | None, click_value),
        images[active_view],
        display_width,
        number_of_points,
    )
    if added:
        st.rerun()

    button_1, button_2, button_3 = st.columns(3)
    with button_1:
        st.button(
            "↶ Undo last point",
            on_click=_undo_view_point,
            args=(active_view,),
            width="stretch",
            disabled=len(selected_for_view) == 0,
        )
    with button_2:
        st.button(
            "Clear this view",
            on_click=_clear_view_points,
            args=(active_view,),
            width="stretch",
            disabled=len(selected_for_view) == 0,
        )
    with button_3:
        any_points = any(len(view) > 0 for view in manual_points)
        st.button(
            "Clear all points",
            on_click=_clear_all_manual_points,
            width="stretch",
            disabled=not any_points,
        )

    if selected_for_view:
        coordinate_rows: list[dict[str, Any]] = []
        for index, point in enumerate(selected_for_view):
            coordinate_rows.append(
                {
                    "Point": index + 1,
                    "Label": _point_label(index),
                    "x (px)": float(point[0]),
                    "y (px)": float(point[1]),
                }
            )

        with st.expander(f"Selected coordinates — View {active_view + 1}"):
            coordinate_df = pd.DataFrame(coordinate_rows)
            st.dataframe(
                coordinate_df.style.format({"x (px)": "{:.2f}", "y (px)": "{:.2f}"}),
                hide_index=True,
                width="stretch",
            )

    all_points_complete = all(
        len(points) == number_of_points for points in manual_points
    )
    if not all_points_complete:
        st.warning(
            "Complete the point selection in all four views before running the reconstruction."
        )

    st.divider()

    # ========================================================
    # 3. Camera intrinsics
    # ========================================================
    st.markdown("### 3. Camera intrinsics")

    intrinsic_option = st.radio(
        "How should camera matrix K be obtained?",
        options=[
            "Approximate from horizontal field of view",
            "Use calibrated camera intrinsics",
        ],
        horizontal=True,
    )

    hfov: float | None = None
    calibration_k: np.ndarray | None = None
    calibration_resolution: tuple[int, int] | None = None

    if intrinsic_option.startswith("Approximate"):
        intrinsic_mode = "hfov"
        hfov = st.slider(
            "Approximate horizontal FOV",
            min_value=45.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
            help=(
                "80° is the default approximation. Calibrated camera intrinsics "
                "are preferred when accurate camera pose is required."
            ),
        )
        st.info(
            "The planar homographies come primarily from the selected correspondences. "
            "The FOV assumption mainly affects recovered camera rotation, translation, "
            "and camera-center positions."
        )
    else:
        intrinsic_mode = "calibrated"
        c1, c2 = st.columns(2)

        with c1:
            fx = st.number_input("fx", min_value=1.0, value=1200.0)
            fy = st.number_input("fy", min_value=1.0, value=1200.0)
            calibration_width = st.number_input(
                "Calibration image width",
                min_value=100,
                value=1920,
            )

        with c2:
            cx = st.number_input("cx", min_value=0.0, value=960.0)
            cy = st.number_input("cy", min_value=0.0, value=540.0)
            calibration_height = st.number_input(
                "Calibration image height",
                min_value=100,
                value=1080,
            )

        calibration_k = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        calibration_resolution = (int(calibration_width), int(calibration_height))

    # ========================================================
    # Cache / result signature
    # ========================================================
    point_signature = tuple(
        tuple((round(float(point[0]), 3), round(float(point[1]), 3)) for point in view)
        for view in manual_points
    )

    if calibration_k is not None:
        calibration_signature: tuple[Any, ...] | None = (
            tuple(float(value) for value in calibration_k.flatten()),
            calibration_resolution,
        )
    else:
        calibration_signature = None

    current_signature: tuple[Any, ...] = (
        selection_signature,
        point_signature,
        float(object_width),
        float(object_height),
        intrinsic_mode,
        hfov,
        calibration_signature,
    )

    # ========================================================
    # Run reconstruction
    # ========================================================
    if st.button(
        "▶ Run Planar SfM Reconstruction",
        type="primary",
        width="stretch",
        disabled=not all_points_complete,
    ):
        try:
            selected_numpy = [
                np.asarray(view, dtype=np.float32) for view in manual_points
            ]

            with st.spinner(
                "Estimating planar homographies, reconstructing the object plane, "
                "and recovering camera poses..."
            ):
                result = process_planar_sfm(
                    images=images,
                    selected_points=selected_numpy,
                    object_width_mm=float(object_width),
                    object_height_mm=float(object_height),
                    intrinsic_mode=intrinsic_mode,
                    hfov_degrees=hfov,
                    calibration_k=calibration_k,
                    calibration_resolution=calibration_resolution,
                )
        except Exception as exc:
            st.error(f"Structure-from-motion processing failed: {exc}")
            return

        st.session_state["sfm_results"] = {
            "signature": current_signature,
            "result": result,
        }

    stored = st.session_state.get("sfm_results")
    if stored is None or stored["signature"] != current_signature:
        return

    result = stored["result"]

    # ========================================================
    # Output 1. Two complementary held-out validation experiments
    # ========================================================
    st.divider()
    st.markdown("## 1. Geometric correspondence and boundary validation")
    st.caption(
        "Both tests use Views 2–4 for aggregate metrics. View 1 defines the "
        "metric reference coordinates and is shown for diagnostic purposes."
    )

    m1, m2 = st.columns(2)
    m1.metric(
        "Corner → interior RMSE",
        f"{result['feature_validation_rmse_target']:.3f} px",
        help="Predicts interior features without including those features in the fitted corner homography.",
    )
    m2.metric(
        "Interior → boundary RMSE",
        f"{result['boundary_validation_rmse_target']:.3f} px",
        help="Predicts the boundary corners from interior features alone. Sensitive to feature distribution and extrapolation.",
    )

    validation_rows: list[dict[str, Any]] = []
    for index, view in enumerate(result["views"], start=1):
        validation_rows.append(
            {
                "View": index,
                "Role": "Reference diagnostic" if index == 1 else "Held-out target",
                "Interior RMSE (px)": view["feature_rmse_px"],
                "Boundary RMSE (px)": view["boundary_rmse_px"],
                "All-point RANSAC inliers": f"{view['all_inliers']}/{number_of_points}",
            }
        )
    st.dataframe(
        pd.DataFrame(validation_rows).style.format(
            {
                "Interior RMSE (px)": "{:.3f}",
                "Boundary RMSE (px)": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    tab_interior, tab_boundary = st.tabs(
        [
            "Corner → interior feature validation",
            "Interior features → boundary validation",
        ]
    )

    with tab_interior:
        st.caption(
            "Green circles: observed interior features · Magenta crosses: predictions "
            "from boundary corners · Yellow segments: errors"
        )
        for start in range(0, 4, 2):
            cols = st.columns(2)
            for offset, col in enumerate(cols):
                idx = start + offset
                with col:
                    st.image(
                        result["views"][idx]["feature_overlay"],
                        caption=f"View {idx + 1}"
                        + (" — reference" if idx == 0 else " — validation"),
                        width="stretch",
                    )

    with tab_boundary:
        st.caption(
            "Green solid polygon: observed boundary · Orange dashed polygon: "
            "predicted boundary using interior features · Yellow segments: corner errors"
        )
        st.info(
            "This is a held-out corner prediction test, not the primary reconstructed "
            "boundary. Its error can become large when interior features occupy a "
            "small region because corner prediction requires extrapolation."
        )
        for start in range(0, 4, 2):
            cols = st.columns(2)
            for offset, col in enumerate(cols):
                idx = start + offset
                with col:
                    st.image(
                        result["views"][idx]["boundary_overlay"],
                        caption=f"View {idx + 1}"
                        + (" — reference" if idx == 0 else " — validation"),
                        width="stretch",
                    )

    # ========================================================
    # Output 2. Reconstructed planar structure
    # ========================================================
    st.markdown("## 2. Reconstructed planar structure")

    st.markdown("""
Each image point is back-projected through its estimated homography into one common
metric object plane. The colored point sets show the reconstructed points from each
view, and the red outline shows the mean reconstructed boundary.
        """)

    st.caption(
        "Perspective rectification (fronto-parallel view) maps the photographed "
        "book cover into the same top-down XY coordinate system used in the "
        "reconstructed-point diagram. It does not correct camera-lens distortion."
    )

    plot_col, photo_col = st.columns(2, gap="large")
    with plot_col:
        st.markdown("**Reconstructed points and boundary (XY plane)**")
        st.image(result["reconstruction_image"], width="stretch")
    with photo_col:
        st.markdown("**Perspective-rectified book cover**")
        selected_rectified_view = st.selectbox(
            "View to display",
            options=[1, 2, 3, 4],
            index=0,
            key="sfm_rectified_view_selection",
            format_func=lambda index: f"View {index}",
        )
        st.image(
            result["rectified_views"][selected_rectified_view - 1],
            caption=f"View {selected_rectified_view}: fronto-parallel rectification",
            width="stretch",
        )

    with st.expander("Compare all four perspective-rectified views"):
        st.caption(
            "All four images are mapped to the same physical object plane. "
            "Differences in sharpness, illumination or small alignment errors may remain."
        )
        rectified_cols = st.columns(4)
        for index, (column, rectified) in enumerate(
            zip(rectified_cols, result["rectified_views"]), start=1
        ):
            with column:
                st.image(
                    rectified,
                    caption=f"Rectified View {index}",
                    width="stretch",
                )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Estimated width", f"{result['reconstructed_width']:.2f} mm")
    r2.metric("Width error", f"{result['width_error_percent']:.2f}%")
    r3.metric("Estimated height", f"{result['reconstructed_height']:.2f} mm")
    r4.metric("Height error", f"{result['height_error_percent']:.2f}%")

    st.caption(
        "Gray is the known physical boundary from your measured object dimensions. "
        "Dashed colored outlines and colored dots are the back-projected points from "
        "Views 1-4. The red polygon and labeled black-outlined points are the mean "
        "reconstructed boundary and mean reconstructed point locations."
    )

    # ========================================================
    # Output 3. Estimated camera positions
    # ========================================================
    st.markdown("## 3. Estimated camera positions")

    st.image(
        result["camera_map"],
        width=680,
        caption="X-Z side view. The planar object lies at Z = 0.",
    )

    camera_rows: list[dict[str, Any]] = []
    for index, view in enumerate(result["views"], start=1):
        camera_center = np.asarray(view["C"], dtype=np.float64)
        translation = np.asarray(view["t"], dtype=np.float64)

        camera_rows.append(
            {
                "View": index,
                "RANSAC inliers": f"{view['all_inliers']}/{number_of_points}",
                "All-point H RMSE (px)": view["homography_rmse"],
                "Pose RMSE (px)": view["pose_rmse"],
                "Camera X (mm)": camera_center[0],
                "Camera Y (mm)": camera_center[1],
                "Camera Z (mm)": camera_center[2],
                "tx": translation[0],
                "ty": translation[1],
                "tz": translation[2],
            }
        )

    st.dataframe(
        pd.DataFrame(camera_rows).style.format(precision=3),
        hide_index=True,
        width="stretch",
    )

    if intrinsic_mode == "hfov":
        st.caption(
            "Camera positions are estimates because K is approximated from the selected "
            "horizontal FOV. Use calibrated intrinsics for stronger quantitative pose accuracy."
        )

    # ========================================================
    # Output 4. Reconstructed point coordinates
    # ========================================================
    st.markdown("## 4. Reconstructed point coordinates")

    point_rows: list[dict[str, Any]] = []
    reconstructed_mean = np.asarray(result["reconstructed_mean"], dtype=np.float64)
    for index, point in enumerate(reconstructed_mean):
        point_rows.append(
            {
                "Point": index + 1,
                "Label": _point_label(index),
                "X (mm)": point[0],
                "Y (mm)": point[1],
            }
        )

    st.dataframe(
        pd.DataFrame(point_rows).style.format(precision=3),
        hide_index=True,
        width="stretch",
    )

    # ========================================================
    # Output 5. Independent ORB/RANSAC evidence
    # ========================================================
    st.markdown("## 5. Independent ORB/RANSAC evidence")

    orb_rows: list[dict[str, Any]] = []
    for target_index, orb_result in enumerate(result["orb_results"], start=2):
        orb_rows.append(
            {
                "Pair": f"View 1 → View {target_index}",
                "Lowe-ratio matches": orb_result["matches"],
                "RANSAC inliers": orb_result["inliers"],
                "Inlier ratio": orb_result["inlier_ratio"],
                "Boundary transfer RMSE (px)": orb_result["corner_rmse"],
            }
        )

    st.dataframe(
        pd.DataFrame(orb_rows).style.format(precision=3),
        hide_index=True,
        width="stretch",
    )

    for target_index, orb_result in enumerate(result["orb_results"], start=2):
        visualization = orb_result["visualization"]
        if visualization is None:
            continue

        with st.expander(f"ORB/RANSAC evidence — View 1 → View {target_index}"):
            st.image(visualization, width="stretch")
            st.caption("Only RANSAC-consistent ORB correspondences are drawn.")

    # ========================================================
    # Output 6. Numerical mathematical workout
    # ========================================================
    st.markdown("## 6. Numerical mathematical workout")

    st.markdown(r"""
For a planar object with $Z=0$:

$$
\lambda\begin{bmatrix}u\\v\\1\end{bmatrix}
=H\begin{bmatrix}X\\Y\\1\end{bmatrix},
\qquad
H \sim K\begin{bmatrix}r_1&r_2&t\end{bmatrix}.
$$

For each view, this implementation uses:

- **$H_{all}$** — all selected correspondences; used for the main planar reconstruction, pose recovery, and back-projection into the object plane.
- **$H_{corners}$** — only TL/TR/BR/BL; predicts held-out interior features.
- **$H_{features}$** — only interior feature points; predicts held-out boundary corners.

View 1 defines the metric coordinates of the selected points. The validation metric at the top therefore uses only Views 2–4.
        """)

    for index, view in enumerate(result["views"], start=1):
        role = "reference coordinate-definition view" if index == 1 else "target view"
        with st.expander(f"View {index}: matrices and errors — {role}"):
            st.markdown("**Camera intrinsic matrix $K$**")
            st.code(_matrix_text(np.asarray(view["K"])))

            st.markdown("**All-point homography $H_{all}$**")
            st.code(_matrix_text(np.asarray(view["H_all"])))

            st.markdown("**Corner-only homography $H_{corners}$**")
            st.code(_matrix_text(np.asarray(view["H_corners"])))

            st.markdown("**Interior-only homography $H_{features}$**")
            st.code(_matrix_text(np.asarray(view["H_features"])))

            st.markdown("**Rotation matrix $R$ recovered from $H_{all}$**")
            st.code(_matrix_text(np.asarray(view["R"])))

            st.markdown("**Translation vector $t$**")
            st.code(_matrix_text(np.asarray(view["t"])))

            st.markdown("**Camera center $C=-R^Tt$**")
            st.code(_matrix_text(np.asarray(view["C"])))

            e1, e2, e3 = st.columns(3)
            e1.metric("All-point H RMSE", f"{view['homography_rmse']:.3f} px")
            e2.metric("Held-out feature RMSE", f"{view['feature_rmse_px']:.3f} px")
            e3.metric("Held-out boundary RMSE", f"{view['boundary_rmse_px']:.3f} px")

            st.markdown("**Held-out interior-feature residuals (pixels)**")
            feature_error_rows = []
            for feature_index, error in enumerate(view["feature_errors_px"], start=1):
                feature_error_rows.append(
                    {"Feature": f"F{feature_index}", "Error (px)": float(error)}
                )
            st.dataframe(
                pd.DataFrame(feature_error_rows).style.format(precision=3),
                hide_index=True,
                width="stretch",
            )
