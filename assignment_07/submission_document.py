from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

PDF_PATH = Path(__file__).parent / "assets" / "assignment_07.pdf"


def render_submission_document() -> None:
    st.markdown(
        '<div class="cv-section-title">📄 Submission Document (Hand-Written Derivation & Report)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cv-note">View or download the compiled submission PDF containing the '
        "by-hand derivation for Question 2 (Uncalibrated Stereo) and supporting documentation.</div>",
        unsafe_allow_html=True,
    )

    if not PDF_PATH.exists():
        st.warning(
            "PDF not found. Place `assignment_07.pdf` inside `assignments/assignment_07/assets/` to render it here."
        )
        return

    pdf_bytes = PDF_PATH.read_bytes()

    st.download_button(
        label="⬇️ Download assignment_07.pdf",
        data=pdf_bytes,
        file_name="assignment_07.pdf",
        mime="application/pdf",
    )

    try:
        st.pdf(pdf_bytes, height=850)
    except Exception:
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f'<embed src="data:application/pdf;base64,{b64_pdf}" '
            'type="application/pdf" width="100%" height="850px" />',
            unsafe_allow_html=True,
        )
