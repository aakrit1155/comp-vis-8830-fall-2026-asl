import streamlit as st

from assignment_02.styles import inject_styles as inject_base_styles
from assignment_02.styles import inject_theory_styles as inject_base_theory_styles


def _inject_assignment_styles() -> None:
    st.markdown(
        """
        <style>
        .cv-task-hero {
            background:
                linear-gradient(
                    135deg,
                    rgba(30,60,114,0.12),
                    rgba(106,17,203,0.10)
                );
            border: 1px solid rgba(42,82,152,0.18);
            padding: 1.15rem 1.25rem;
            border-radius: 14px;
            margin-bottom: 1rem;
        }

        .cv-task-hero h2 {
            font-size: 1.30rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
        }

        .cv-task-hero p {
            margin: 0;
            line-height: 1.55;
            font-size: 0.94rem;
        }

        .cv-input-guide {
            background: rgba(42, 82, 152, 0.07);
            border: 1px solid rgba(42, 82, 152, 0.13);
            border-left: 4px solid #2a5298;
            padding: 0.85rem 1rem;
            border-radius: 10px;
            margin: 0.65rem 0 1rem 0;
            font-size: 0.91rem;
            line-height: 1.55;
        }

        .cv-result-note {
            background: rgba(106,17,203,0.06);
            border-left: 4px solid #6a11cb;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin: 0.7rem 0;
            font-size: 0.91rem;
        }

        .cv-caption {
            opacity: 0.75;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.16);
            border-radius: 12px;
            padding: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_styles() -> None:
    inject_base_styles()
    _inject_assignment_styles()


def inject_theory_styles() -> None:
    inject_base_theory_styles()
    _inject_assignment_styles()
