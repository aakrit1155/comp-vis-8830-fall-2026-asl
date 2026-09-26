import streamlit as st

from assignment_02.styles import inject_theory_styles


def _step(num: int, title: str):
    """Render a numbered step header matching the theme."""
    st.markdown(
        f"""
        <div class="cv-step">
            <div class="badge">{num}</div>
            <div class="label">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_theory():
    inject_theory_styles()
    st.markdown(
        '<div class="cv-section-title">📖 Two-View Geometry — Derivation</div>',
        unsafe_allow_html=True,
    )
    st.caption("Mathematical background for the two-view reconstruction task.")
    # ---------- Hero ----------
    st.markdown(
        """
        <div class="cv-theory-hero">
            <h2>📖 Theoretical Derivation: Two-View Geometric Relationship</h2>
            <p>
                A step-by-step walkthrough from camera-frame projections to the
                Fundamental matrix — establishing the epipolar constraint that
                links correspondences across two views.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Setup note ----------
    st.markdown(
        """
        <div class="cv-note">
            <b>Setup:</b> A static 3D world coordinate system is aligned with
            <b>Camera 1</b>. <b>Camera 2</b> is positioned at a relative rotation
            <b>R</b> and translation <b>t</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Step 1 ----------
    _step(1, "Camera Frame Projections")
    st.latex(
        r"\mathbf{P}_1 = \begin{bmatrix} X_1 \\ Y_1 \\ Z_1 \end{bmatrix}, \quad \mathbf{P}_2 = \mathbf{R} \mathbf{P}_1 + \mathbf{t}"
    )
    st.markdown(
        "Let the intrinsic matrices for Camera 1 and Camera 2 be "
        "$\\mathbf{K}_1$ and $\\mathbf{K}_2$. In homogeneous pixel coordinates:"
    )
    st.latex(
        r"\tilde{\mathbf{p}}_1 = \begin{bmatrix} u_1 \\ v_1 \\ 1 \end{bmatrix} = \frac{1}{Z_1} \mathbf{K}_1 \mathbf{P}_1 \implies \mathbf{P}_1 = Z_1 \mathbf{K}_1^{-1} \tilde{\mathbf{p}}_1"
    )
    st.latex(
        r"\tilde{\mathbf{p}}_2 = \begin{bmatrix} u_2 \\ v_2 \\ 1 \end{bmatrix} = \frac{1}{Z_2} \mathbf{K}_2 \mathbf{P}_2 \implies \mathbf{P}_2 = Z_2 \mathbf{K}_2^{-1} \tilde{\mathbf{p}}_2"
    )

    # ---------- Step 2 ----------
    _step(2, "Epipolar Geometry & Coplanarity Constraint")
    st.markdown(
        "Vectors $\\mathbf{P}_2$, $\\mathbf{t}$, and $\\mathbf{R}\\mathbf{P}_1$ "
        "lie in the same epipolar plane. Their scalar triple product is zero:"
    )
    st.latex(r"\mathbf{P}_2^T \cdot (\mathbf{t} \times (\mathbf{R} \mathbf{P}_1)) = 0")
    st.markdown(
        "Expressing the cross product as matrix multiplication via "
        "skew-symmetric matrix $[\\mathbf{t}]_\\times$:"
    )
    st.latex(
        r"[\mathbf{t}]_\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}"
    )
    st.latex(r"\mathbf{P}_2^T [\mathbf{t}]_\times \mathbf{R} \mathbf{P}_1 = 0")

    # ---------- Step 3 ----------
    _step(3, "Essential & Fundamental Matrices")
    st.latex(
        r"\mathbf{E} = [\mathbf{t}]_\times \mathbf{R} \quad \implies \quad \mathbf{P}_2^T \mathbf{E} \mathbf{P}_1 = 0"
    )
    st.markdown("Substituting normalized coordinates into pixel coordinates:")
    st.latex(
        r"(Z_2 \mathbf{K}_2^{-1} \tilde{\mathbf{p}}_2)^T \mathbf{E} (Z_1 \mathbf{K}_1^{-1} \tilde{\mathbf{p}}_1) = 0"
    )
    st.latex(
        r"\tilde{\mathbf{p}}_2^T (\mathbf{K}_2^{-T} \mathbf{E} \mathbf{K}_1^{-1}) \tilde{\mathbf{p}}_1 = 0"
    )
    st.latex(
        r"\mathbf{F} = \mathbf{K}_2^{-T} [\mathbf{t}]_\times \mathbf{R} \mathbf{K}_1^{-1}"
    )
    st.latex(r"\tilde{\mathbf{p}}_2^T \mathbf{F} \tilde{\mathbf{p}}_1 = 0")

    # ---------- Step 4 ----------
    _step(4, "Parameter Determination & Computation")
    st.markdown(
        """
        <div class="cv-kv">
            <b>Intrinsic Matrices (K₁, K₂):</b> Focal lengths (f<sub>x</sub>, f<sub>y</sub>),
            principal point (c<sub>x</sub>, c<sub>y</sub>), and skew (s). Determined via
            Zhang's calibration method.
        </div>
        <div class="cv-kv">
            <b>Extrinsics (R, t):</b> Obtained via stereo camera calibration
            (<code>cv2.stereoCalibrate</code>) or decomposed from the essential
            matrix <b>E</b> computed via 8-point / 5-point matching.
        </div>
        <div class="cv-kv">
            <b>Epipolar Line Relation:</b> For any point <b>p1</b> in Camera 1, its
            corresponding point in Camera 2 lies along the epipolar line
            <b>l₂ = Fp1</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Summary callout ----------
    st.markdown(
        """
        <div class="cv-note" style="margin-top:1.2rem;">
            <b>Takeaway:</b> The Fundamental matrix <b>F</b> is a single 3×3 entity
            that encodes both the intrinsics of each camera and their relative pose —
            reducing the two-view correspondence problem to a single linear constraint
            on homogeneous pixel coordinates.
        </div>
        """,
        unsafe_allow_html=True,
    )
