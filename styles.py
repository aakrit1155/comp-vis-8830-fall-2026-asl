import streamlit as st


def inject_styles():
    st.markdown(
        """
        <style>
        .cv-hero {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a11cb 100%);
            padding: 2.2rem 2rem;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 1.4rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .cv-hero h1 {
            color: #ffffff;
            font-size: 2.1rem;
            margin-bottom: 0.3rem;
            font-weight: 700;
        }
        .cv-hero p {
            color: #e6ecff;
            font-size: 1.02rem;
            margin: 0;
            line-height: 1.5;
        }
        .cv-hero .pill {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.35);
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            margin-right: 8px;
            margin-top: 12px;
        }
        .cv-card-title {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
        }
        .cv-section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
