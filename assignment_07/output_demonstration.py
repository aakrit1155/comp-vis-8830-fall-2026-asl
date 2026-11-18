from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DISPLAY_WIDTH = 600
ASSETS_DIR = Path(__file__).parent / "assets"


def _show_snap(filename: str, caption: str) -> None:
    path = ASSETS_DIR / filename
    if path.exists():
        st.image(str(path), width=DISPLAY_WIDTH, caption=caption)
    else:
        st.info(
            f"Add `{filename}` to `assignments/assignment_07/assets/` to display this snapshot."
        )


def render_output_demonstrations() -> None:
    st.markdown(
        '<div class="cv-section-title">🪧 Output Demonstrations</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cv-note">Visual snapshots of inputs, intermediate epipolar outputs, '
        "annotated measurements, and derived metric tables for Question 1 and Question 3.</div>",
        unsafe_allow_html=True,
    )

    # --- Question 1 Showcase ---
    st.subheader("1. Question 1 — Calibrated Stereo (Square Object Width)")
    st.write(
        "Input stereo pair captured at a known baseline, followed by annotated width edges (P1–P2) and depth Z."
    )
    c1, c2 = st.columns(2)
    with c1:
        _show_snap("q1_inputs.png", "Q1 Input: Left & Right Stereo Pair")
    with c2:
        _show_snap(
            "q1_output_metrics.png", "Q1 Output: Annotated Edges, Depth Z & Width"
        )

    # --- Question 3 Showcase ---
    st.subheader("2. Question 3 — Video Room Size Estimation (80 cm Side-Step)")
    st.write(
        "Extracted keyframes before/after the side-step, RANSAC epipolar inliers, and triangulated room corners."
    )
    c1, c2 = st.columns(2)
    with c1:
        _show_snap("q3_inputs.png", "Q3 Input: Keyframe 1 & Keyframe 2")

    with c2:
        _show_snap(
            "q3_output_metrics.png", "Q3 Output: Annotated Room Corners (W, L, H)"
        )

    # --- Sample Validation Summary Table (Update with your measured ground truth) ---
    st.subheader("3. Derived Results Summary Table")
    summary_df = pd.DataFrame(
        [
            {
                "Experiment": "Q1: Square Object Width",
                "Baseline / Step": "30 cm",
                "Estimated": "26.7 cm",
                "Ground Truth": "26.3 cm",
            },
            {
                "Experiment": "Q3: Room Width",
                "Baseline / Step": "50 cm",
                "Estimated": "3.76 m",
                "Ground Truth": "4 m",
            },
            {
                "Experiment": "Q3: Room Length",
                "Baseline / Step": "50 cm",
                "Estimated": "3.98 m",
                "Ground Truth": "4 m",
            },
            {
                "Experiment": "Q3: Room Height",
                "Baseline / Step": "50 cm",
                "Estimated": "2.33 m",
                "Ground Truth": "2.5 m",
            },
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
