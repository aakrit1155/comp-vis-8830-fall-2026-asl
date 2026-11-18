import streamlit as st

from assignment_02.styles import inject_styles

from .object_size import render_object_size_estimate
from .output_demonstration import render_output_demonstrations
from .room_size import render_room_size_estimate
from .submission_document import render_submission_document


def render() -> None:
    inject_styles()

    st.button(
        "← Back to Assignment Hub",
        on_click=lambda: st.session_state.update({"active_assignment": "hub"}),
    )

    st.markdown(
        """
        <div class="cv-hero">
            <h1>📷 Assignment-07 — Module 07 —Stereo Vision</h1>
            <p>
                Process stereo images to estimate 3D structure and size dimensions.
                Process videos to estimate size of the room.
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
            "📐 Question 1: Object Size Estimation",
            "📐🧱 Question 2: Room Size Estimation",
            "🪧 Output Demonstrations",
            "📄 Submission Document",
        ]
    )

    with tabs[0]:
        render_object_size_estimate()

    with tabs[1]:
        render_room_size_estimate()

    with tabs[2]:
        render_output_demonstrations()

    with tabs[3]:
        render_submission_document()
