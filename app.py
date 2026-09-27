import streamlit as st

from assignment_02 import ui as assignment_02_ui
from assignment_03 import ui as assignment_03_ui
from styles import inject_styles

st.set_page_config(
    page_title="Computer Vision Hub",
    page_icon="👁️",
    layout="wide",
)

# --- Session state ---
if "calibration_matrix" not in st.session_state:
    st.session_state.calibration_matrix = None
if "dist_coeffs" not in st.session_state:
    st.session_state.dist_coeffs = None
if "active_assignment" not in st.session_state:
    st.session_state.active_assignment = "hub"


ASSIGNMENTS = [
    {
        "id": "assignment_02",
        "name": "Module 02",
        "desc": "Camera Calibration, Perspective Projection & Stereo Geometry",
        "icon": "📷",
        "active": True,
    },
    {
        "id": "assignment_03",
        "name": "Module 03",
        "desc": "Image Blurring Using Filtering Approach",
        "icon": "🖼️",
        "active": True,
    },
    {
        "id": "assignment_04",
        "name": "Module 04",
        "desc": "Image Segmentation(In Progress)",
        "icon": "👀",
        "active": False,
    },
]


def show_hub():
    inject_styles()

    # ---------------------- Hero ----------------------
    st.markdown(
        """
        <div class="cv-hero">
            <h1>👁️ Computer Vision — Assignment Hub </h1>
            <h5> — Aakrit Sharma Lamsal </h5>
            <p>
                A curated showcase of assignment tasks for the course
                <b>Computer Vision</b>, under <b>Prof. Ashok Aswin</b>.
                Each module below walks through a self-contained pipeline —
                from spatial filtering to camera calibration and beyond.
            </p>
            <span class="pill">👤 Aakrit Sharma Lamsal · 002865270 </span>
            <span class="pill">📧 Google Classroom: sharmalamsalaakrit@gmail.com </span>
            <span class="pill">📧 GSU: asharmalamsal1@student.gsu.edu </span>
            <span class="pill">🎓 Course: Computer Vision</span>
            <span class="pill">🧪 Interactive Demos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------- Quick stats ----------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modules", "02", "CV pipeline")
    c2.metric("Available Now", "02", "Assignment 03 live")
    c3.metric("Focus", "Image → 3D, 2D", "Vision stack")
    c4.metric("Stack", "Python + CV", "OpenCV · NumPy")

    st.divider()

    # ---------------------- About section ----------------------
    about_col, tips_col = st.columns([1.4, 1])

    with about_col:
        st.markdown(
            '<div class="cv-section-title">📖 About this project</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
            This project serves the purpose of showcasing the assignment tasks
            for the course **Computer Vision** under **Prof. Ashok Aswin**.
            Each module is designed to be an end-to-end demonstration of a
            core computer-vision concept, from classical image processing
            all the way to geometry-driven 3D reconstruction.

            Use the cards below to navigate to the respective assignment.
            Each module page exposes its own inputs and interactive controls
            so you can experiment with your own images and observe how the
            pipeline responds.
            """)

    with tips_col:
        st.markdown(
            '<div class="cv-section-title">💡 How to use</div>',
            unsafe_allow_html=True,
        )
        st.info("""
            • **Pick a module** from the cards below.  
            • **Upload your own images** when prompted.  
            • Follow the on-screen steps for each task.  
            • Results update **live** as you tweak parameters.
            """)

    st.divider()

    # ---------------------- Assignment cards ----------------------
    st.markdown(
        '<div class="cv-section-title">🗂️ Assignment Modules</div>',
        unsafe_allow_html=True,
    )
    st.caption("Click on an active module to launch its pipeline.")

    max_cols = 3
    for i in range(0, len(ASSIGNMENTS), max_cols):
        cols = st.columns(max_cols)
        for j, col in enumerate(cols):
            index = i + j
            if index < len(ASSIGNMENTS):
                module = ASSIGNMENTS[index]
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f"<div class='cv-card-title'>{module['icon']} {module['name']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(module["desc"])

                        if module["active"]:
                            st.success("Ready", icon="✅")
                            if st.button(
                                f"Open {module['name']}  →",
                                type="primary",
                                width="stretch",
                                key=f"btn_{module['id']}",
                            ):
                                st.session_state.active_assignment = module["id"]
                                st.rerun()
                        else:
                            st.warning("Coming soon", icon="🚧")
                            st.button(
                                f"Open {module['name']}",
                                disabled=True,
                                width="stretch",
                                key=f"btn_{module['id']}",
                            )

    st.divider()

    # ---------------------- Footer ----------------------
    st.caption(
        "Built by Aakrit Sharma Lamsal with ❤️ using Streamlit · OpenCV · NumPy — "
        "for the Computer Vision course under Prof. Ashok Aswin."
    )


if st.session_state.active_assignment == "hub":
    show_hub()
elif st.session_state.active_assignment == "assignment_02":
    assignment_02_ui.render()
elif st.session_state.active_assignment == "assignment_03":
    assignment_03_ui.render()
