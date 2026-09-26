import streamlit as st

from .calibration import calibration_render
from .Implementation_Logic import render_theory
from .projection import projection_render, render_validation_stats
from .styles import inject_styles


def render():
    inject_styles()
    st.button(
        "← Back to Assignment Hub",
        on_click=lambda: st.session_state.update({"active_assignment": "hub"}),
    )
    st.markdown(
        """
        <div class="cv-hero">
            <h1>📷 Module 02 — Calibration &amp; Perspective Projection</h1>
            <p>
                Camera calibration, real-world dimension estimation from a single
                view, and empirical validation beyond 2 meters. Work through the
                tabs to explore each task end-to-end.
            </p>
            <span class="pill">🎓 Computer Vision</span>
            <span class="pill">👤 Aakrit Sharma Lamsal · 002865270</span>
            <span class="pill">🧪 4 Tasks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cv-note">
            <b>Tip:</b> Complete <b>Question 1</b> first — the intrinsic matrix
            <code>K</code> computed there should be used for the
            projection tasks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "🔧 Question 1: Calibration",
            "📐 Question 2: 2D Object Dimensions",
            "📊 Question 3: Validation (>2m)",
            "📖 Question 4: Two-View Derivation",
        ]
    )

    # ---------------- TAB 1: CALIBRATION ----------------
    with tabs[0]:
        calibration_render()

    # ---------------- TAB 2: PERSPECTIVE PROJECTION ----------------
    with tabs[1]:
        projection_render()

    # ---------------- TAB 3: VALIDATION STATS ----------------
    with tabs[2]:

        render_validation_stats()

    # ---------------- TAB 4: THEORY ----------------
    with tabs[3]:
        render_theory()
