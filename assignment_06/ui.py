import streamlit as st

from .optical_flow import render_optical_flow
from .structure_from_motion import render_structure_from_motion
from .styles import inject_styles
from .theory import render_theory


def render() -> None:
    inject_styles()

    st.button(
        "← Back to Assignment Hub",
        on_click=lambda: st.session_state.update({"active_assignment": "hub"}),
    )

    st.markdown(
        """
        <div class="cv-hero">
            <h1>📷 Assignment-06 — Module 05/06 —Optical Flow &amp; Structure from Motion</h1>
            <p>
                Motion analysis using dense and sparse optical flow,
                followed by planar multi-view reconstruction and
                camera-pose estimation from four viewpoints.
            </p>
            <span class="pill">🎓 Computer Vision</span>
            <span class="pill">👤 Aakrit Sharma Lamsal · 002865270</span>
            <span class="pill">🧪 2 Experiments and Theory</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "⏭️➡️ Question 1: Optical Flow",
            "🏔️ Question 2: Structure from Motion",
            "📐 Mathematical Theory",
        ]
    )

    with tabs[0]:
        render_optical_flow()

    with tabs[1]:
        render_structure_from_motion()

    with tabs[2]:
        render_theory()
