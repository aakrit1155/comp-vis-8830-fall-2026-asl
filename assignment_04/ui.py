import streamlit as st

from assignment_02.styles import inject_styles

from .boundary_rgb import render_boundary_rgb
from .boundary_thermal import render_boundary_thermal
from .theory import render_theory


def render():
    inject_styles()
    st.button(
        "← Back to Assignment Hub",
        on_click=lambda: st.session_state.update({"active_assignment": "hub"}),
    )
    st.markdown(
        """
        <div class="cv-hero">
            <h1>📷 Assignment-04 -- Module 04/05 — Boundary Extraction &amp; For RGB and Thermal Images</h1>
            <p>
                Applying Boundary Extraction algorithm on digital images for segmentation of human from the images.
                Showcasing how edge detection and segmentation of regions can be achieved using Fourier (frequency) domain analysis.
            </p>
            <span class="pill">🎓 Computer Vision</span>
            <span class="pill">👤 Aakrit Sharma Lamsal · 002865270</span>
            <span class="pill">🧪 3 Tasks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "😬 Question 1: Boundary RGB",
            "🌡️ Question 2: Boundary Thermal",
            "📖 Question 3: Theory",
        ]
    )

    # ---------------- TAB 1: Boundary Extraction RGB ----------------
    with tabs[0]:
        render_boundary_rgb()

    # ---------------- TAB 2: Boundary Extraction Thermal ----------------
    with tabs[1]:
        render_boundary_thermal()

    # ---------------- TAB 3: Theory ----------------
    with tabs[2]:
        render_theory()
