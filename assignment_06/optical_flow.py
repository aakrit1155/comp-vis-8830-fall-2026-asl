from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from .media_utils import (
    MemoryMP4Writer,
    decode_consecutive_pair,
    iter_video_segment,
    probe_video,
)

MAX_VIDEO_BYTES = 80 * 1024 * 1024


# ============================================================
# Dense optical flow
# ============================================================


def _flow_to_rgb(
    flow: np.ndarray,
    magnitude_clip: float = 6.0,
) -> np.ndarray:
    """
    Standard HSV optical-flow visualization.

    Hue       -> motion direction
    Brightness -> motion magnitude
    """

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=False,
    )

    hsv = np.zeros(
        (*magnitude.shape, 3),
        dtype=np.uint8,
    )

    # OpenCV H range: [0, 179]
    hsv[..., 0] = (angle * 90.0 / np.pi).astype(np.uint8)

    hsv[..., 1] = 255

    hsv[..., 2] = np.clip(
        magnitude / magnitude_clip * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )


def _draw_flow_vectors(
    rgb: np.ndarray,
    flow: np.ndarray,
    step: int = 24,
    min_magnitude: float = 0.7,
) -> np.ndarray:
    vis = rgb.copy()

    h, w = flow.shape[:2]

    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = flow[y, x]

            magnitude = math.hypot(float(u), float(v))

            if magnitude < min_magnitude:
                continue

            end = (
                int(round(x + u)),
                int(round(y + v)),
            )

            cv2.arrowedLine(
                vis,
                (x, y),
                end,
                (50, 235, 95),
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )

    return vis


def _put_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
) -> None:
    x, y = origin

    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _compose_flow_frame(
    rgb: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    arrows = _draw_flow_vectors(
        rgb,
        flow,
    )

    dense = _flow_to_rgb(flow)

    composite = np.concatenate(
        [arrows, dense],
        axis=1,
    )

    _put_label(
        composite,
        "Tracked motion vectors",
        (12, 25),
    )

    _put_label(
        composite,
        "Dense flow: hue = direction, brightness = magnitude",
        (rgb.shape[1] + 12, 25),
    )

    return composite


def _direction_name(angle_degrees: float) -> str:
    directions = [
        "right",
        "down-right",
        "down",
        "down-left",
        "left",
        "up-left",
        "up",
        "up-right",
    ]

    index = int(((angle_degrees + 22.5) % 360) // 45)

    return directions[index]


def process_optical_flow_video(
    video_bytes: bytes,
    start_seconds: float,
    target_fps: int,
    max_side: int,
) -> dict[str, Any]:
    frames = iter_video_segment(
        video_bytes,
        start_seconds=start_seconds,
        duration_seconds=30.0,
        target_fps=target_fps,
        max_side=max_side,
    )

    try:
        first_rgb, _ = next(frames)
    except StopIteration as exc:
        raise ValueError("No frames were decoded from the selected segment.") from exc

    first_gray = cv2.cvtColor(
        first_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    zero_flow = np.zeros(
        (*first_gray.shape, 2),
        dtype=np.float32,
    )

    first_output = _compose_flow_frame(
        first_rgb,
        zero_flow,
    )

    writer = MemoryMP4Writer(
        width=first_output.shape[1],
        height=first_output.shape[0],
        fps=target_fps,
    )

    writer.write(first_output)

    prev_gray = first_gray

    mean_magnitudes: list[float] = []
    p90_magnitudes: list[float] = []
    motion_ratios: list[float] = []

    direction_histogram = np.zeros(8, dtype=np.float64)

    weighted_cos = 0.0
    weighted_sin = 0.0

    frame_count = 1

    angle_bins = np.linspace(
        0,
        2 * np.pi,
        9,
    )

    for rgb, _ in frames:
        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY,
        )

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
        )

        mean_magnitudes.append(float(np.mean(magnitude)))

        p90_magnitudes.append(float(np.percentile(magnitude, 90)))

        motion_ratios.append(float(np.mean(magnitude >= 1.0)))

        hist, _ = np.histogram(
            angle,
            bins=angle_bins,
            weights=magnitude,
        )

        direction_histogram += hist

        weighted_cos += float(np.sum(magnitude * np.cos(angle)))

        weighted_sin += float(np.sum(magnitude * np.sin(angle)))

        output = _compose_flow_frame(
            rgb,
            flow,
        )

        writer.write(output)

        prev_gray = gray
        frame_count += 1

    video_output = writer.close()

    dominant_angle = (
        math.degrees(
            math.atan2(
                weighted_sin,
                weighted_cos,
            )
        )
        + 360.0
    ) % 360.0

    histogram_total = float(direction_histogram.sum())

    if histogram_total > 0:
        histogram_percent = direction_histogram / histogram_total * 100.0
    else:
        histogram_percent = direction_histogram

    mean_motion_ratio = float(np.mean(motion_ratios)) if motion_ratios else 0.0

    if mean_motion_ratio >= 0.35:
        coverage_interpretation = (
            "Motion occupies a large portion of the image. "
            "This is consistent with camera motion or broad "
            "scene motion."
        )
    else:
        coverage_interpretation = (
            "Motion is spatially localized. This is more "
            "consistent with independently moving objects "
            "against a relatively stable background."
        )

    return {
        "video_bytes": video_output,
        "frames": frame_count,
        "mean_magnitude": (float(np.mean(mean_magnitudes)) if mean_magnitudes else 0.0),
        "p90_magnitude": (float(np.mean(p90_magnitudes)) if p90_magnitudes else 0.0),
        "motion_ratio": mean_motion_ratio,
        "dominant_angle": dominant_angle,
        "dominant_direction": _direction_name(dominant_angle),
        "coverage_interpretation": coverage_interpretation,
        "direction_histogram": histogram_percent,
    }


# ============================================================
# Bilinear interpolation
# ============================================================


def _bilinear_sample(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Explicit bilinear interpolation.

    This intentionally implements the mathematical interpolation
    instead of calling cv2.remap so that the code corresponds
    directly to the theory section.
    """

    h, w = image.shape

    x = np.clip(
        x,
        0,
        w - 1.001,
    )

    y = np.clip(
        y,
        0,
        h - 1.001,
    )

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)

    x1 = x0 + 1
    y1 = y0 + 1

    alpha = x - x0
    beta = y - y0

    i00 = image[y0, x0]
    i10 = image[y0, x1]
    i01 = image[y1, x0]
    i11 = image[y1, x1]

    return (
        (1 - alpha) * (1 - beta) * i00
        + alpha * (1 - beta) * i10
        + (1 - alpha) * beta * i01
        + alpha * beta * i11
    )


# ============================================================
# Manual Lucas-Kanade validation
# ============================================================


def _manual_lk_point(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    point: np.ndarray,
    radius: int = 7,
    max_iterations: int = 15,
) -> tuple[np.ndarray | None, int]:
    """
    Translation-only Lucas-Kanade solved by Gauss-Newton.

    Returns the predicted location in frame 2.
    """

    prev_f = prev_gray.astype(np.float32)
    next_f = next_gray.astype(np.float32)

    grad_x = (
        cv2.Sobel(
            next_f,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
        / 8.0
    )

    grad_y = (
        cv2.Sobel(
            next_f,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )
        / 8.0
    )

    px = float(point[0])
    py = float(point[1])

    h, w = prev_gray.shape

    if (
        px < radius + 2
        or py < radius + 2
        or px >= w - radius - 2
        or py >= h - radius - 2
    ):
        return None, 0

    offsets = np.arange(
        -radius,
        radius + 1,
        dtype=np.float32,
    )

    dx_grid, dy_grid = np.meshgrid(
        offsets,
        offsets,
    )

    x_template = px + dx_grid
    y_template = py + dy_grid

    template = _bilinear_sample(
        prev_f,
        x_template,
        y_template,
    ).reshape(-1)

    displacement = np.zeros(
        2,
        dtype=np.float64,
    )

    for iteration in range(max_iterations):
        warped_x = x_template + displacement[0]
        warped_y = y_template + displacement[1]

        if (
            np.min(warped_x) < 1
            or np.min(warped_y) < 1
            or np.max(warped_x) >= w - 2
            or np.max(warped_y) >= h - 2
        ):
            return None, iteration

        warped = _bilinear_sample(
            next_f,
            warped_x,
            warped_y,
        ).reshape(-1)

        ix = _bilinear_sample(
            grad_x,
            warped_x,
            warped_y,
        ).reshape(-1)

        iy = _bilinear_sample(
            grad_y,
            warped_x,
            warped_y,
        ).reshape(-1)

        jacobian = np.column_stack([ix, iy])

        residual = template - warped

        hessian = jacobian.T @ jacobian

        eigenvalues = np.linalg.eigvalsh(hessian)

        if eigenvalues[0] < 1e-4:
            return None, iteration

        delta = np.linalg.solve(
            hessian + 1e-6 * np.eye(2),
            jacobian.T @ residual,
        )

        displacement += delta

        if np.linalg.norm(delta) < 1e-3:
            return (
                np.array(
                    [
                        px + displacement[0],
                        py + displacement[1],
                    ],
                    dtype=np.float32,
                ),
                iteration + 1,
            )

    return (
        np.array(
            [
                px + displacement[0],
                py + displacement[1],
            ],
            dtype=np.float32,
        ),
        max_iterations,
    )


def validate_lucas_kanade(
    frame_1_rgb: np.ndarray,
    frame_2_rgb: np.ndarray,
    max_points: int = 12,
) -> tuple[pd.DataFrame, np.ndarray]:
    gray_1 = cv2.cvtColor(
        frame_1_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    gray_2 = cv2.cvtColor(
        frame_2_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    points_1 = cv2.goodFeaturesToTrack(
        gray_1,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=7,
    )

    if points_1 is None:
        raise ValueError("No reliable Shi-Tomasi feature points were found.")

    lk_params = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }

    points_2, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        gray_1,
        gray_2,
        points_1,
        None,
        **lk_params,
    )

    points_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        gray_2,
        gray_1,
        points_2,
        None,
        **lk_params,
    )

    p0 = points_1.reshape(-1, 2)
    p1 = points_2.reshape(-1, 2)
    pb = points_back.reshape(-1, 2)

    forward_ok = status_forward.reshape(-1) == 1
    backward_ok = status_backward.reshape(-1) == 1

    fb_error = np.linalg.norm(
        p0 - pb,
        axis=1,
    )

    displacement = np.linalg.norm(
        p1 - p0,
        axis=1,
    )

    reliable = forward_ok & backward_ok & (fb_error < 1.0) & (displacement < 12.0)

    reliable_indices = np.where(reliable)[0]

    reliable_indices = reliable_indices[np.argsort(fb_error[reliable_indices])]

    rows: list[dict[str, float | int]] = []

    vis = frame_2_rgb.copy()

    for idx in reliable_indices:
        if len(rows) >= max_points:
            break

        manual_point, iterations = _manual_lk_point(
            gray_1,
            gray_2,
            p0[idx],
        )

        if manual_point is None:
            continue

        endpoint_error = float(np.linalg.norm(manual_point - p1[idx]))

        point_id = len(rows) + 1

        rows.append(
            {
                "Point": point_id,
                "x₁": float(p0[idx, 0]),
                "y₁": float(p0[idx, 1]),
                "x₂ theoretical": float(manual_point[0]),
                "y₂ theoretical": float(manual_point[1]),
                "x₂ OpenCV": float(p1[idx, 0]),
                "y₂ OpenCV": float(p1[idx, 1]),
                "Endpoint error (px)": endpoint_error,
                "Forward-backward error (px)": float(fb_error[idx]),
                "Iterations": iterations,
            }
        )

        start = tuple(np.round(p0[idx]).astype(int))

        actual = tuple(np.round(p1[idx]).astype(int))

        theoretical = tuple(np.round(manual_point).astype(int))

        cv2.arrowedLine(
            vis,
            start,
            actual,
            (35, 220, 70),
            1,
            cv2.LINE_AA,
            tipLength=0.25,
        )

        cv2.circle(
            vis,
            actual,
            4,
            (35, 220, 70),
            2,
            cv2.LINE_AA,
        )

        cv2.circle(
            vis,
            theoretical,
            3,
            (230, 60, 230),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            str(point_id),
            (actual[0] + 4, actual[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if not rows:
        raise ValueError(
            "Reliable Lucas-Kanade points could not be established "
            "for this frame pair. Choose a different validation time."
        )

    return pd.DataFrame(rows), vis


# ============================================================
# Streamlit UI
# ============================================================


def _reset_optical_flow() -> None:
    st.session_state["of_nonce"] = st.session_state.get("of_nonce", 0) + 1

    st.session_state.pop(
        "of_results",
        None,
    )


def _display_video_meta(
    title: str,
    meta,
    byte_count: int,
) -> None:
    st.markdown(f"**{title}**")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Duration",
        f"{meta.duration:.1f} s",
    )

    c2.metric(
        "Resolution",
        f"{meta.width} × {meta.height}",
    )

    c3.metric(
        "Source FPS",
        f"{meta.fps:.1f}",
    )

    c4.metric(
        "Upload",
        f"{byte_count / 1024**2:.1f} MB",
    )


def render_optical_flow() -> None:
    st.markdown(
        """
        <div class="cv-task-hero">
            <h2>Question 1 — Optical Flow & Motion Tracking</h2>
            <p>
                Compute dense optical flow over two 30-second video
                sequences, quantify the observed motion, and validate
                the Lucas–Kanade tracking equations on two consecutive
                source frames from each sequence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cv-input-guide">
        <b>Recommended input.</b>
        Upload two MP4/MOV videos containing at least 30 seconds of
        visible motion. H.264 MP4, 720p and 30 FPS is ideal.
        The application processes only a selected 30-second segment,
        reduces the working resolution, and temporally samples the
        video before dense-flow computation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    header_left, header_right = st.columns([5, 1])

    with header_right:
        st.button(
            "↻ Reset",
            key="of_reset",
            on_click=_reset_optical_flow,
            width="stretch",
        )

    nonce = st.session_state.get(
        "of_nonce",
        0,
    )

    upload_col_1, upload_col_2 = st.columns(2)

    with upload_col_1:
        video_1 = st.file_uploader(
            "Video 1",
            type=["mp4", "mov", "m4v"],
            key=f"of_video_1_{nonce}",
            help=("Prefer a static camera with one independently moving object."),
        )

    with upload_col_2:
        video_2 = st.file_uploader(
            "Video 2",
            type=["mp4", "mov", "m4v"],
            key=f"of_video_2_{nonce}",
            help=(
                "Prefer camera motion or a second, visibly different motion pattern."
            ),
        )

    if video_1 is None or video_2 is None:
        st.info("Upload both videos to configure the 30-second samples.")
        return

    video_bytes = [
        video_1.getvalue(),
        video_2.getvalue(),
    ]

    for index, data in enumerate(
        video_bytes,
        start=1,
    ):
        if len(data) > MAX_VIDEO_BYTES:
            st.error(
                f"Video {index} exceeds the internal 80 MB safety "
                "limit. Compress it with HandBrake before uploading."
            )
            return

    try:
        metadata = [probe_video(data) for data in video_bytes]
    except Exception as exc:
        st.error(f"Unable to decode the uploaded video: {exc}")
        return

    for i, (meta, data) in enumerate(
        zip(metadata, video_bytes),
        start=1,
    ):
        _display_video_meta(
            f"Video {i}",
            meta,
            len(data),
        )

        if meta.duration < 30.0:
            st.error(
                f"Video {i} is only {meta.duration:.1f} seconds. "
                "The assignment requires at least 30 seconds."
            )
            return

    st.divider()

    st.markdown("### Processing configuration")

    cfg1, cfg2 = st.columns(2)

    with cfg1:
        max_side = st.select_slider(
            "Internal maximum frame dimension",
            options=[360, 480, 640],
            value=480,
            help=(
                "480 is recommended for Streamlit Cloud. "
                "640 gives more detail but costs substantially "
                "more CPU."
            ),
        )

    with cfg2:
        target_fps = st.select_slider(
            "Dense-flow analysis FPS",
            options=[5, 6, 8, 10],
            value=8,
            help=(
                "The original source remains unchanged. "
                "Only analysis frames are temporally sampled."
            ),
        )

    starts: list[float] = []

    for index, meta in enumerate(
        metadata,
        start=1,
    ):
        max_start = max(
            0.0,
            meta.duration - 30.0,
        )

        if max_start > 0.5:
            start = st.slider(
                f"Video {index}: start of 30-second sample",
                min_value=0.0,
                max_value=float(max_start),
                value=0.0,
                step=0.5,
                key=f"of_start_{index}_{nonce}",
            )
        else:
            start = 0.0

        starts.append(start)

    validation_offset = st.slider(
        "Frame-pair validation point inside each 30-second segment",
        min_value=1.0,
        max_value=28.0,
        value=10.0,
        step=0.5,
        help=(
            "Two consecutive ORIGINAL source frames are decoded "
            "near this point for Lucas–Kanade validation."
        ),
    )

    st.caption(
        "The dense-flow output is intentionally downsampled for "
        "cloud efficiency. The mathematical validation uses two "
        "consecutive frames from the original source stream."
    )

    current_signature = (
        video_1.name,
        len(video_bytes[0]),
        video_2.name,
        len(video_bytes[1]),
        tuple(starts),
        max_side,
        target_fps,
        validation_offset,
    )

    if st.button(
        "▶ Run Optical Flow Analysis",
        type="primary",
        width="stretch",
    ):
        all_results = []

        try:
            for index in range(2):
                with st.spinner(f"Processing Video {index + 1}..."):
                    dense_result = process_optical_flow_video(
                        video_bytes[index],
                        start_seconds=starts[index],
                        target_fps=target_fps,
                        max_side=max_side,
                    )

                    pair_time = starts[index] + validation_offset

                    (
                        frame_1,
                        frame_2,
                        t1,
                        t2,
                    ) = decode_consecutive_pair(
                        video_bytes[index],
                        target_seconds=pair_time,
                        max_side=640,
                    )

                    table, validation_image = validate_lucas_kanade(
                        frame_1,
                        frame_2,
                    )

                    all_results.append(
                        {
                            "dense": dense_result,
                            "validation_table": table,
                            "validation_image": validation_image,
                            "frame_time_1": t1,
                            "frame_time_2": t2,
                        }
                    )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

        st.session_state["of_results"] = {
            "signature": current_signature,
            "results": all_results,
        }

    stored = st.session_state.get("of_results")

    if stored is None or stored["signature"] != current_signature:
        return

    st.divider()
    st.markdown("## Results")

    histogram_labels = [
        "0–45°",
        "45–90°",
        "90–135°",
        "135–180°",
        "180–225°",
        "225–270°",
        "270–315°",
        "315–360°",
    ]

    for video_index, result in enumerate(
        stored["results"],
        start=1,
    ):
        dense = result["dense"]

        st.markdown(f"### Video {video_index}")

        st.video(
            dense["video_bytes"],
            format="video/mp4",
            width=680,
        )

        st.caption(
            "Left: motion vectors over the RGB image. "
            "Right: dense Farnebäck optical flow. "
            "Hue represents direction; brightness represents "
            "flow magnitude."
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Mean flow",
            f"{dense['mean_magnitude']:.2f} px",
        )

        m2.metric(
            "Mean P90 flow",
            f"{dense['p90_magnitude']:.2f} px",
        )

        m3.metric(
            "Moving pixels",
            f"{100*dense['motion_ratio']:.1f}%",
        )

        m4.metric(
            "Dominant motion",
            (f"{dense['dominant_direction']} " f"({dense['dominant_angle']:.0f}°)"),
        )

        st.markdown(
            f"""
            <div class="cv-result-note">
            <b>Motion inference:</b>
            {dense["coverage_interpretation"]}
            The magnitude and directional statistics provide
            quantitative evidence for this interpretation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        histogram_df = pd.DataFrame(
            {
                "Direction sector": histogram_labels,
                "Weighted flow (%)": (dense["direction_histogram"]),
            }
        ).set_index("Direction sector")

        st.markdown("**Direction evidence**")

        st.bar_chart(histogram_df)

        st.markdown("#### Two-frame Lucas–Kanade validation")

        st.image(
            result["validation_image"],
            width=680,
            caption=(
                "Green = OpenCV pyramidal Lucas–Kanade. "
                "Magenta = point predicted by the manually "
                "implemented Lucas–Kanade normal equations."
            ),
        )

        table = result["validation_table"]

        st.dataframe(
            table.style.format(precision=3),
            hide_index=True,
            width="stretch",
        )

        mean_error = float(table["Endpoint error (px)"].mean())

        st.caption(
            f"Source-frame timestamps: "
            f"{result['frame_time_1']:.4f} s → "
            f"{result['frame_time_2']:.4f} s. "
            f"Mean theoretical/OpenCV endpoint difference: "
            f"{mean_error:.3f} px."
        )
