import streamlit as st

from assignment_02.styles import inject_styles

from .blurring import render_blurring
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
            <h1>📷 Module 02 — Filtering &amp; Blurring Projection</h1>
            <p>
                Applying Image Processing on digital images for blurring and filtering.
                Showcasing that using spatial filters is same as using the fourier domain equivalent filters.
            </p>
            <span class="pill">🎓 Computer Vision</span>
            <span class="pill">👤 Aakrit Sharma Lamsal · 002865270</span>
            <span class="pill">🧪 2 Tasks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "🔧 Question 1: Blurring",
            "📖 Question 2: Theory",
        ]
    )

    # ---------------- TAB 1: Blurring ----------------
    with tabs[0]:
        render_blurring()

    # ---------------- TAB 2: Theory ----------------
    with tabs[1]:
        render_theory()
