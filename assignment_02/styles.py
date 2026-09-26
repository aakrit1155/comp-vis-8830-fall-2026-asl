import streamlit as st


def inject_styles():
    st.markdown(
        """
        <style>
        .cv-hero {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a11cb 100%);
            padding: 1.8rem 1.8rem;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .cv-hero h1 {
            color: #ffffff;
            font-size: 1.75rem;
            margin-bottom: 0.25rem;
            font-weight: 700;
        }
        .cv-hero p {
            color: #e6ecff;
            font-size: 0.98rem;
            margin: 0;
            line-height: 1.5;
        }
        .cv-hero .pill {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.35);
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            margin-right: 8px;
            margin-top: 12px;
        }
        .cv-section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .cv-note {
            background: rgba(42, 82, 152, 0.08);
            border-left: 4px solid #2a5298;
            padding: 0.7rem 1rem;
            border-radius: 8px;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_theory_styles():
    """Local styles for the theory panel — mirrors hub theme."""
    st.markdown(
        """
        <style>
        .cv-theory-hero {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a11cb 100%);
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            color: #ffffff;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.16);
        }
        .cv-theory-hero h2 {
            color: #ffffff;
            font-size: 1.35rem;
            margin: 0 0 0.35rem 0;
            font-weight: 700;
        }
        .cv-theory-hero p {
            color: #e6ecff;
            font-size: 0.94rem;
            margin: 0;
            line-height: 1.5;
        }
        .cv-step {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 1.2rem 0 0.5rem 0;
        }
        .cv-step .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2a5298, #6a11cb);
            color: #fff;
            font-weight: 700;
            font-size: 0.9rem;
            box-shadow: 0 3px 10px rgba(42,82,152,0.35);
            flex-shrink: 0;
        }
        .cv-step .label {
            font-size: 1.1rem;
            font-weight: 700;
            color: inherit;
        }
        .cv-note {
            background: rgba(42, 82, 152, 0.08);
            border-left: 4px solid #2a5298;
            padding: 0.7rem 1rem;
            border-radius: 8px;
            font-size: 0.92rem;
            margin: 0.8rem 0;
        }
        .cv-kv {
            background: rgba(106, 17, 203, 0.06);
            border-left: 4px solid #6a11cb;
            padding: 0.7rem 1rem;
            border-radius: 8px;
            font-size: 0.92rem;
            margin: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
