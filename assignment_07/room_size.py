from __future__ import annotations

import math
import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

DISPLAY_WIDTH = 600


def _extract_frame(video_bytes: bytes, frame_idx: int) -> Tuple[int, float, np.ndarray]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_idx, max(0, total - 1))))
        ok, bgr = cap.read()
        cap.release()
        if not ok or bgr is None:
            raise ValueError(f"Could not read frame {frame_idx}.")
        return total, fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


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


def _resize_pil(img_rgb: np.ndarray) -> Tuple[Image.Image, float]:
    h, w = img_rgb.shape[:2]
    scale = w / float(DISPLAY_WIDTH)
    resized = cv2.resize(img_rgb, (DISPLAY_WIDTH, int(round(h / scale))))
    return Image.fromarray(resized), scale


def _draw_room_corners(
    img_rgb: np.ndarray, pts: List[Tuple[float, float]]
) -> np.ndarray:
    vis = img_rgb.copy()
    labels = ["P1 (Bot-Left)", "P2 (Bot-Right)", "P3 (Top-Left)"]
    colors = [(0, 220, 130), (0, 180, 255), (255, 90, 160)]
    p_int = [(int(round(x)), int(round(y))) for x, y in pts]

    for i, (px, py) in enumerate(p_int):
        col = colors[i % len(colors)]
        cv2.circle(vis, (px, py), 7, col, -1)
        cv2.circle(vis, (px, py), 9, (255, 255, 255), 2)
        cv2.putText(
            vis,
            labels[i],
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            col,
            2,
            cv2.LINE_AA,
        )

    # Draw Width line (P1 -> P2) and Height line (P1 -> P3)
    if len(p_int) >= 2:
        cv2.line(vis, p_int[0], p_int[1], (0, 220, 130), 3, cv2.LINE_AA)
    if len(p_int) == 3:
        cv2.line(vis, p_int[0], p_int[2], (255, 90, 160), 3, cv2.LINE_AA)

    return vis


def _init_q3_state() -> None:
    for k, v in {
        "q3_f1_pts": [],
        "q3_f2_pts": [],
        "q3_last_1": None,
        "q3_last_2": None,
        "q3_up_key": 0,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_q3_points() -> None:
    st.session_state["q3_f1_pts"] = []
    st.session_state["q3_f2_pts"] = []
    st.session_state["q3_last_1"] = None
    st.session_state["q3_last_2"] = None


def _reset_q3_all() -> None:
    _reset_q3_points()
    st.session_state["q3_up_key"] += 1


def render_room_size_estimate() -> None:
    _init_q3_state()

    st.markdown(
        '<div class="cv-section-title">🧱 Room Size Estimation from Side-Step Video</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cv-note">Stand with your back against <b>Wall D</b> facing <b>Wall B (Front Wall)</b>, '
        "and record a video taking one parallel side-step. On both Frame 1 and Frame 2, click 3 corners on the front wall:<br>"
        "<b>1. P1 (Bottom-Left floor corner)</b> · <b>2. P2 (Bottom-Right floor corner)</b> · <b>3. P3 (Top-Left ceiling corner)</b>.<br>"
        "The app computes <b>Width</b> (P1↔P2), <b>Height</b> (P1↔P3), and <b>Length/Depth</b> (camera distance Z from Wall D to Wall B).</div>",
        unsafe_allow_html=True,
    )

    _, r_col = st.columns([3, 1])
    r_col.button(
        "🔄 Reset All",
        key="q3_reset_all",
        on_click=_reset_q3_all,
        use_container_width=True,
    )

    video_file = st.file_uploader(
        "Upload Room Video (720p MP4 / MOV)",
        type=["mp4", "mov", "avi"],
        key=f"q3_up_{st.session_state['q3_up_key']}",
    )
    if not video_file:
        st.info("Upload your side-step room video to begin.")
        return

    v_bytes = video_file.getvalue()
    total_frames, fps, _ = _extract_frame(v_bytes, 0)

    c1, c2 = st.columns(2)
    side_step_cm = c1.slider(
        "Side-Step Distance (cm)", min_value=20, max_value=100, value=50, step=5
    )
    fov_deg = c2.slider(
        "Camera Wide/Long-Edge FOV (°)",
        min_value=45.0,
        max_value=100.0,
        value=64.0,
        step=0.5,
    )
    baseline_m = side_step_cm / 100.0

    fc1, fc2 = st.columns(2)
    f1_idx = fc1.slider("Frame 1 (Before Side-Step)", 0, max(1, total_frames - 1), 0)
    f2_idx = fc2.slider(
        "Frame 2 (After Side-Step)",
        0,
        max(1, total_frames - 1),
        min(total_frames - 1, int(fps * 2.0)),
    )

    _, _, frame1 = _extract_frame(v_bytes, f1_idx)
    _, _, frame2 = _extract_frame(v_bytes, f2_idx)
    h, w = frame1.shape[:2]

    long_edge = float(max(w, h))
    fx = (long_edge / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    fy, cx, cy = fx, w / 2.0, h / 2.0

    b_col, _ = st.columns([1, 4])
    b_col.button(
        "🧹 Clear Corners", on_click=_reset_q3_points, use_container_width=True
    )

    pil1, s1 = _resize_pil(_draw_room_corners(frame1, st.session_state["q3_f1_pts"]))
    pil2, s2 = _resize_pil(_draw_room_corners(frame2, st.session_state["q3_f2_pts"]))

    col1, col2 = st.columns(2)
    with col1:
        st.caption(
            f"Frame {f1_idx} (Before Step) — Click P1 (Bot-L), P2 (Bot-R), P3 (Top-L) · {DISPLAY_WIDTH}px view"
        )
        click1 = streamlit_image_coordinates(pil1, key="q3_coord_1")
    with col2:
        st.caption(
            f"Frame {f2_idx} (After {side_step_cm} cm Step) — Click P1 (Bot-L), P2 (Bot-R), P3 (Top-L) · {DISPLAY_WIDTH}px view"
        )
        click2 = streamlit_image_coordinates(pil2, key="q3_coord_2")

    if click1 is not None:
        raw1 = (click1["x"], click1["y"])
        if raw1 != st.session_state["q3_last_1"]:
            st.session_state["q3_last_1"] = raw1
            if len(st.session_state["q3_f1_pts"]) < 3:
                st.session_state["q3_f1_pts"].append((raw1[0] * s1, raw1[1] * s1))
            st.rerun()

    if click2 is not None:
        raw2 = (click2["x"], click2["y"])
        if raw2 != st.session_state["q3_last_2"]:
            st.session_state["q3_last_2"] = raw2
            if len(st.session_state["q3_f2_pts"]) < 3:
                st.session_state["q3_f2_pts"].append((raw2[0] * s2, raw2[1] * s2))
            st.rerun()

    f1_pts = st.session_state["q3_f1_pts"]
    f2_pts = st.session_state["q3_f2_pts"]

    if len(f1_pts) == 3 and len(f2_pts) == 3:
        pts_3d = []
        for i in range(3):
            p = _triangulate_point(
                f1_pts[i][0],
                f1_pts[i][1],
                f2_pts[i][0],
                baseline_m,
                fx,
                fy,
                cx,
                cy,
            )
            if p is None:
                st.error(
                    f"Disparity |xL - xR| for P{i+1} is too small (< 1 px). Please re-click."
                )
                return
            pts_3d.append(p)

        # Length (Depth Z) = mean distance from Wall D (camera) to Wall B (front wall)
        room_length_m = float(np.mean([p[2] for p in pts_3d]))

        # Width (P1 Bot-Left <-> P2 Bot-Right) and Height (P1 Bot-Left <-> P3 Top-Left)
        px_width = math.hypot(f1_pts[1][0] - f1_pts[0][0], f1_pts[1][1] - f1_pts[0][1])
        px_height = math.hypot(f1_pts[2][0] - f1_pts[0][0], f1_pts[2][1] - f1_pts[0][1])

        room_width_m = (px_width * room_length_m) / fx
        room_height_m = (px_height * room_length_m) / fy

        area_m2 = room_width_m * room_length_m
        vol_m3 = area_m2 * room_height_m

        st.subheader("Estimated Room Dimensions")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Room Width (Wall B)",
            f"{room_width_m:.2f} m",
            f"{room_width_m * 3.28084:.2f} ft",
        )
        m2.metric(
            "Room Length (Wall D → B)",
            f"{room_length_m:.2f} m",
            f"{room_length_m * 3.28084:.2f} ft",
        )
        m3.metric(
            "Room Height (Floor → Ceiling)",
            f"{room_height_m:.2f} m",
            f"{room_height_m * 3.28084:.2f} ft",
        )

        df = pd.DataFrame(
            [
                {
                    "Dimension": "Room Width (P1 ↔ P2)",
                    "Estimated (m)": round(room_width_m, 2),
                    "Estimated (ft)": round(room_width_m * 3.28084, 2),
                },
                {
                    "Dimension": "Room Length / Depth (Z)",
                    "Estimated (m)": round(room_length_m, 2),
                    "Estimated (ft)": round(room_length_m * 3.28084, 2),
                },
                {
                    "Dimension": "Room Height (P1 ↔ P3)",
                    "Estimated (m)": round(room_height_m, 2),
                    "Estimated (ft)": round(room_height_m * 3.28084, 2),
                },
                {
                    "Dimension": "Floor Area (m² / sq ft)",
                    "Estimated (m)": round(area_m2, 2),
                    "Estimated (ft)": round(area_m2 * 10.7639, 2),
                },
                {
                    "Dimension": "Room Volume (m³ / cu ft)",
                    "Estimated (m)": round(vol_m3, 2),
                    "Estimated (ft)": round(vol_m3 * 35.3147, 2),
                },
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
