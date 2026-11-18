from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

DISPLAY_WIDTH = 600


def _decode_image_in_memory(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode the uploaded image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_for_display(
    img_rgb: np.ndarray, target_width: int = DISPLAY_WIDTH
) -> Tuple[Image.Image, float]:
    h, w = img_rgb.shape[:2]
    scale_to_orig = w / float(target_width)
    target_h = int(round(h / scale_to_orig))
    resized = cv2.resize(
        img_rgb, (target_width, target_h), interpolation=cv2.INTER_AREA
    )
    return Image.fromarray(resized), scale_to_orig


def _draw_width_points(
    img_rgb: np.ndarray,
    points: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 220, 130),
) -> np.ndarray:
    vis = img_rgb.copy()
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    for idx, (px, py) in enumerate(pts):
        cv2.circle(vis, (px, py), 7, color, -1)
        cv2.circle(vis, (px, py), 9, (255, 255, 255), 2)
        cv2.putText(
            vis,
            f"P{idx + 1}",
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    if len(pts) == 2:
        cv2.line(vis, pts[0], pts[1], color, 3, cv2.LINE_AA)
    return vis


def _triangulate_point(
    xl: float,
    yl: float,
    xr: float,
    baseline_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Optional[Tuple[float, float, float]]:
    disparity = abs(xl - xr)
    if disparity < 1.0:
        return None
    z = (fx * baseline_m) / disparity
    x = ((xl - cx) * z) / fx
    y = ((yl - cy) * z) / fy
    return (x, y, z)


def _init_q1_state() -> None:
    for k, v in {
        "q1_left_pts": [],
        "q1_right_pts": [],
        "q1_last_l": None,
        "q1_last_r": None,
        "q1_up_key": 0,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_q1_points() -> None:
    st.session_state["q1_left_pts"] = []
    st.session_state["q1_right_pts"] = []
    st.session_state["q1_last_l"] = None
    st.session_state["q1_last_r"] = None


def _reset_q1_all() -> None:
    _reset_q1_points()
    st.session_state["q1_up_key"] += 1


def render_object_size_estimate() -> None:
    _init_q1_state()

    st.markdown(
        '<div class="cv-section-title">📐 Calibrated Stereo — Square Object Width</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cv-note">Upload Left and Right images of a square object taken a known distance apart. '
        "Click the left and right edges (<b>P1</b>, <b>P2</b>) on both images to compute depth <i>Z</i> and width.</div>",
        unsafe_allow_html=True,
    )

    _, r_col = st.columns([3, 1])
    r_col.button(
        "🔄 Reset All",
        key="q1_reset_all",
        on_click=_reset_q1_all,
        width="stretch",
    )

    uk = st.session_state["q1_up_key"]
    u1, u2 = st.columns(2)
    left_file = u1.file_uploader(
        "Upload Left Stereo Image", type=["png", "jpg", "jpeg"], key=f"q1_l_{uk}"
    )
    right_file = u2.file_uploader(
        "Upload Right Stereo Image", type=["png", "jpg", "jpeg"], key=f"q1_r_{uk}"
    )

    if not left_file or not right_file:
        st.info("Upload both Left and Right images to begin.")
        return

    left_rgb = _decode_image_in_memory(left_file)
    right_rgb = _decode_image_in_memory(right_file)
    h, w = left_rgb.shape[:2]

    c1, c2 = st.columns(2)
    baseline_cm = c1.slider(
        "Camera Baseline Distance (cm)", min_value=5, max_value=100, value=30, step=1
    )
    fov_deg = c2.slider(
        "Camera Horizontal FOV (°)",
        min_value=45.0,
        max_value=100.0,
        value=70.0,
        step=0.5,
    )

    baseline_m = baseline_cm / 100.0
    fx = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    fy, cx, cy = fx, w / 2.0, h / 2.0

    b_col, _ = st.columns([1, 4])
    b_col.button("🧹 Clear Points", on_click=_reset_q1_points, width="stretch")

    vis_l = _draw_width_points(left_rgb, st.session_state["q1_left_pts"], (0, 220, 130))
    vis_r = _draw_width_points(
        right_rgb, st.session_state["q1_right_pts"], (0, 180, 255)
    )
    disp_l, scale_l = _resize_for_display(vis_l, DISPLAY_WIDTH)
    disp_r, scale_r = _resize_for_display(vis_r, DISPLAY_WIDTH)

    col_l, col_r = st.columns(2)
    with col_l:
        st.caption(
            f"Left Image — click P1 (left edge) then P2 (right edge) · {DISPLAY_WIDTH}px view"
        )
        click_l = streamlit_image_coordinates(disp_l, key="q1_coord_l")
    with col_r:
        st.caption(
            f"Right Image — click P1 (left edge) then P2 (right edge) · {DISPLAY_WIDTH}px view"
        )
        click_r = streamlit_image_coordinates(disp_r, key="q1_coord_r")

    if click_l is not None:
        raw_l = (click_l["x"], click_l["y"])
        if raw_l != st.session_state["q1_last_l"]:
            st.session_state["q1_last_l"] = raw_l
            if len(st.session_state["q1_left_pts"]) < 2:
                st.session_state["q1_left_pts"].append(
                    (raw_l[0] * scale_l, raw_l[1] * scale_l)
                )
            st.rerun()

    if click_r is not None:
        raw_r = (click_r["x"], click_r["y"])
        if raw_r != st.session_state["q1_last_r"]:
            st.session_state["q1_last_r"] = raw_r
            if len(st.session_state["q1_right_pts"]) < 2:
                st.session_state["q1_right_pts"].append(
                    (raw_r[0] * scale_r, raw_r[1] * scale_r)
                )
            st.rerun()

    l_pts = st.session_state["q1_left_pts"]
    r_pts = st.session_state["q1_right_pts"]
    if len(l_pts) == 2 and len(r_pts) == 2:
        p1 = _triangulate_point(
            l_pts[0][0], l_pts[0][1], r_pts[0][0], baseline_m, fx, fy, cx, cy
        )
        p2 = _triangulate_point(
            l_pts[1][0], l_pts[1][1], r_pts[1][0], baseline_m, fx, fy, cx, cy
        )
        if p1 is None or p2 is None:
            st.error(
                "Disparity |xL - xR| is too small (< 1 px). Please re-click corresponding points."
            )
            return

        mean_z_cm = ((p1[2] + p2[2]) / 2.0) * 100.0
        width_cm = (
            math.sqrt(
                (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
            )
            * 100.0
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Estimated Depth (Z)", f"{mean_z_cm:.2f} cm", f"{mean_z_cm / 100.0:.3f} m"
        )
        m2.metric(
            "Estimated Square Width", f"{width_cm:.2f} cm", f"{width_cm * 10.0:.1f} mm"
        )
        m3.metric("Baseline Used (B)", f"{baseline_cm} cm")

        df = pd.DataFrame(
            [
                {
                    "Point / Metric": "P1 (Left Edge)",
                    "X (cm)": round(p1[0] * 100, 2),
                    "Y (cm)": round(p1[1] * 100, 2),
                    "Z Depth (cm)": round(p1[2] * 100, 2),
                },
                {
                    "Point / Metric": "P2 (Right Edge)",
                    "X (cm)": round(p2[0] * 100, 2),
                    "Y (cm)": round(p2[1] * 100, 2),
                    "Z Depth (cm)": round(p2[2] * 100, 2),
                },
            ]
        )
        st.dataframe(df, width="stretch", hide_index=True)
