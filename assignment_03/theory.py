## assignment_02/theory.py
import time

import cv2
import numpy as np
import streamlit as st

from assignment_02.styles import inject_theory_styles


def _generate_synthetic_test_image(size: int = 256) -> np.ndarray:
    """Create an in-memory grayscale synthetic test image if no user image is uploaded."""
    img = np.zeros((size, size), dtype=np.uint8)
    # Checkerboard background
    block = 32
    for r in range(0, size, block):
        for c in range(0, size, block):
            if ((r // block) + (c // block)) % 2 == 0:
                img[r : r + block, c : c + block] = 50

    # High-contrast geometric shapes to test sharp edge & frequency response
    cv2.rectangle(img, (40, 40), (120, 120), 220, -1)
    cv2.circle(img, (180, 90), 42, 255, -1)
    cv2.line(img, (30, 210), (225, 160), 240, 5)
    cv2.putText(img, "FFT = CONV", (35, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 255, 2)
    return img


def _get_gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """Construct a normalized 2D Gaussian filter kernel h[m, n]."""
    g1d = cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_64F)
    kernel_2d = g1d @ g1d.T
    return kernel_2d / np.sum(kernel_2d)


def _run_equivalence_experiment(gray_img: np.ndarray, ksize: int, sigma: float) -> dict:
    """
    Compare 2D linear spatial convolution against zero-padded 2D DFT multiplication
    in double precision (float64) to verify the 2D Convolution Theorem.
    """
    f = gray_img.astype(np.float64)
    M, N = f.shape
    h = _get_gaussian_kernel_2d(ksize, sigma)
    kh, kw = h.shape
    pad_y, pad_x = kh // 2, kw // 2

    # 1. Spatial Domain Filtering: Linear Convolution with zero-padded boundary
    t0 = time.perf_counter()
    f_padded_spatial = cv2.copyMakeBorder(
        f, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_CONSTANT, value=0.0
    )
    spatial_full = cv2.filter2D(
        f_padded_spatial, ddepth=cv2.CV_64F, kernel=h, borderType=cv2.BORDER_CONSTANT
    )
    spatial_out = spatial_full[pad_y : pad_y + M, pad_x : pad_x + N]
    spatial_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Fourier Domain Filtering: Zero-padded 2D FFT Multiplication
    # Padded dimensions P >= M + kh - 1, Q >= N + kw - 1 prevent circular wrap-around
    t1 = time.perf_counter()
    P, Q = M + kh - 1, N + kw - 1

    f_pad = np.zeros((P, Q), dtype=np.float64)
    f_pad[:M, :N] = f

    # Embed kernel and circularly shift its center (pad_y, pad_x) to (0, 0) for zero-phase alignment
    h_pad = np.zeros((P, Q), dtype=np.float64)
    h_pad[:kh, :kw] = h
    h_pad = np.roll(h_pad, shift=(-pad_y, -pad_x), axis=(0, 1))

    # Forward 2D DFTs
    F_uv = np.fft.fft2(f_pad)
    H_uv = np.fft.fft2(h_pad)

    # Pointwise multiplication in Frequency Domain
    G_uv = F_uv * H_uv

    # Inverse 2D DFT and crop back to original M x N support
    g_pad = np.real(np.fft.ifft2(G_uv))
    freq_out = g_pad[:M, :N]
    freq_ms = (time.perf_counter() - t1) * 1000.0

    # 3. Error & Spectrum Diagnostics
    abs_diff = np.abs(spatial_out - freq_out)
    mse = float(np.mean((spatial_out - freq_out) ** 2))
    max_err = float(np.max(abs_diff))

    # Log-magnitude spectrum visualization of |G(u,v)| = |F(u,v) * H(u,v)|
    G_shifted = np.fft.fftshift(G_uv)
    mag_spec = np.log1p(np.abs(G_shifted))
    mag_spec_norm = cv2.normalize(
        mag_spec, np.empty_like(mag_spec), 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Normalize absolute error heatmap for visual inspection
    err_vis = cv2.normalize(
        abs_diff, np.empty_like(mag_spec), 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    err_heatmap = cv2.applyColorMap(err_vis, cv2.COLORMAP_INFERNO)
    err_heatmap = cv2.cvtColor(err_heatmap, cv2.COLOR_BGR2RGB)

    return {
        "spatial_u8": np.clip(spatial_out, 0, 255).astype(np.uint8),
        "freq_u8": np.clip(freq_out, 0, 255).astype(np.uint8),
        "spectrum_u8": mag_spec_norm,
        "err_heatmap": err_heatmap,
        "mse": mse,
        "max_err": max_err,
        "spatial_ms": spatial_ms,
        "freq_ms": freq_ms,
        "padded_shape": (P, Q),
    }


def render_theory() -> None:
    """Render Tab 2: Mathematical Derivation & Empirical Validation of the Convolution Theorem."""
    inject_theory_styles()

    st.markdown(
        """
        <div class="cv-theory-hero">
            <h2>📖 Theoretical Proof &amp; Empirical Validation: The 2D Convolution Theorem</h2>
            <p>
                Proving mathematically and experimentally that spatial domain filtering (convolution)
                is strictly equivalent to pointwise multiplication in the Fourier frequency domain.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- SECTION 1: CONTINUOUS DERIVATION ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">1</span>
            <span class="label">Continuous 2D Convolution &amp; Fourier Transform Definitions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "Let $f(x, y)$ denote a 2D spatial image and $h(x, y)$ denote a linear spatial filter kernel. "
        "The 2D linear convolution $g(x, y) = (f * h)(x, y)$ is defined by the superposition integral:"
    )
    st.latex(
        r"g(x, y) = (f * h)(x, y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau, \eta)\, h(x - \tau, y - \eta)\, d\tau\, d\eta"
    )
    st.markdown(
        "The forward 2D Continuous Fourier Transform "
        "$\\mathcal{F}\\{g(x, y)\\} = G(u, v)$ "
        "maps spatial coordinates $(x, y)$ into spatial frequencies $(u, v)$:"
    )
    st.latex(
        r"G(u, v) = \mathcal{F}\{g(x, y)\} = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} g(x, y)\, e^{-j 2\pi (ux + vy)}\, dx\, dy"
    )

    # ---------------- SECTION 2: STEP-BY-STEP PROOF ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">2</span>
            <span class="label">Step-by-Step Derivation of the 2D Convolution Theorem</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Step 2.1 — Substitute the convolution integral into the Fourier Transform:**"
    )
    st.latex(
        r"\mathcal{F}\{f * h\}(u, v) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} "
        r"\left[ \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau, \eta)\, h(x - \tau, y - \eta)\, d\tau\, d\eta \right] "
        r"e^{-j 2\pi (ux + vy)}\, dx\, dy"
    )

    st.markdown(
        "**Step 2.2 — Interchange the order of integration (by Fubini's Theorem):**"
    )
    st.latex(
        r"\mathcal{F}\{f * h\}(u, v) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau, \eta) "
        r"\left[ \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} h(x - \tau, y - \eta)\, e^{-j 2\pi (ux + vy)}\, dx\, dy \right] d\tau\, d\eta"
    )

    st.markdown(
        "**Step 2.3 — Apply change of variables:** Let $\\alpha = x - \\tau \\implies x = \\alpha + \\tau$ ($dx = d\\alpha$) "
        "and $\\beta = y - \\eta \\implies y = \\beta + \\eta$ ($dy = d\\beta$):"
    )
    st.latex(
        r"\mathcal{F}\{f * h\}(u, v) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau, \eta) "
        r"\left[ \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} h(\alpha, \beta)\, e^{-j 2\pi (u(\alpha + \tau) + v(\beta + \eta))}\, d\alpha\, d\beta \right] d\tau\, d\eta"
    )

    st.markdown(
        "**Step 2.4 — Factor the complex exponential $e^{-j 2\\pi(u\\tau + v\\eta)}$ out of the inner integral:**"
    )
    st.latex(
        r"\mathcal{F}\{f * h\}(u, v) = \underbrace{\left[ \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau, \eta)\, e^{-j 2\pi (u\tau + v\eta)}\, d\tau\, d\eta \right]}_{F(u, v)} "
        r"\cdot \underbrace{\left[ \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} h(\alpha, \beta)\, e^{-j 2\pi (u\alpha + v\beta)}\, d\alpha\, d\beta \right]}_{H(u, v)}"
    )

    st.markdown(
        "**Step 2.5 — Final Equivalence Identity:** Taking the Inverse Fourier Transform $\\mathcal{F}^{-1}$ on both sides yields:"
    )
    st.latex(
        r"\boxed{(f * h)(x, y) = \mathcal{F}^{-1}\big\{ F(u, v) \cdot H(u, v) \big\}} \quad \Longleftrightarrow \quad \boxed{f(x, y) * h(x, y) \xleftrightarrow{\mathcal{F}} F(u, v) \cdot H(u, v)}"
    )

    # ---------------- SECTION 3: DISCRETE DOMAIN & ZERO PADDING ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">3</span>
            <span class="label">2D Discrete Fourier Transform (DFT) &amp; Zero-Padding Condition</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "For a discrete digital image $f[m, n]$ of size $M \\times N$ and a discrete filter kernel $h[m, n]$ of size $k_h \\times k_w$, "
        "the 2D Discrete Fourier Transform (DFT) is periodic and inherently computes **circular convolution** ($\\circledast$). "
        "To make frequency-domain multiplication strictly identical to **linear spatial convolution** without wrap-around boundary aliasing, "
        "both $f$ and $h$ are zero-padded to dimensions $P \\times Q$:"
    )
    st.latex(r"P \ge M + k_h - 1, \qquad Q \ge N + k_w - 1")
    st.latex(
        r"F[u, v] = \sum_{m=0}^{P-1} \sum_{n=0}^{Q-1} f_{\text{pad}}[m, n]\, e^{-j 2\pi \left(\frac{um}{P} + \frac{vn}{Q}\right)}, \qquad "
        r"g[m, n] = \frac{1}{PQ} \sum_{u=0}^{P-1} \sum_{v=0}^{Q-1} F[u, v]\, H[u, v]\, e^{j 2\pi \left(\frac{um}{P} + \frac{vn}{Q}\right)}"
    )

    st.markdown(
        """
    <div class="cv-kv">
        <b>Gaussian Dual Property:</b>
        The Fourier Transform of a spatial Gaussian filter is another Gaussian
        in the frequency domain.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.latex(r"h(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}")

    st.latex(r"H(u,v) = e^{-2\pi^2\sigma^2(u^2+v^2)}")

    st.markdown(
        """
        <div class="cv-kv">
            Thus, a wider spatial Gaussian (<code>σ ↑</code>) corresponds to a narrower
            low-pass frequency filter (<code>D₀ ↓</code>), explaining why both produce
            smooth, ring-free blurring.
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ---------------- SECTION 4: EMPIRICAL VALIDATION ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">4</span>
            <span class="label">Experimental Validation &amp; Numerical Verification</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Use the uploaded image from Tab 1 if available; otherwise use synthetic pattern
    uploaded_img = st.session_state.get("cv_uploaded_img", None)
    if uploaded_img is not None:
        if uploaded_img.ndim == 3:
            test_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_RGB2GRAY)
        else:
            test_gray = uploaded_img.copy()
        source_caption = "Using your uploaded image from Tab 1 (converted to single-channel luminance for exact matrix comparison)."
    else:
        test_gray = _generate_synthetic_test_image(256)
        source_caption = "No image uploaded in Tab 1 yet — using an auto-generated 256×256 synthetic test image in memory."

    st.markdown(
        f'<div class="cv-note"><b>Active Experimental Input:</b> {source_caption}</div>',
        unsafe_allow_html=True,
    )

    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        exp_ksize = st.slider(
            "Validation Gaussian Kernel Size (k × k)",
            min_value=3,
            max_value=31,
            value=15,
            step=2,
            key="theory_ksize",
        )
    with exp_c2:
        exp_sigma = st.slider(
            "Validation Gaussian Sigma (σ)",
            min_value=0.5,
            max_value=8.0,
            value=3.0,
            step=0.5,
            key="theory_sigma",
        )

    exp_results = _run_equivalence_experiment(
        test_gray, ksize=exp_ksize, sigma=exp_sigma
    )

    # Quantitative Verification Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Squared Error (MSE)", f"{exp_results['mse']:.2e}")
    m2.metric("Max Pixel Diff (L∞)", f"{exp_results['max_err']:.2e}")
    m3.metric("Spatial Conv Time", f"{exp_results['spatial_ms']:.2f} ms")
    m4.metric("2D FFT Mult Time", f"{exp_results['freq_ms']:.2f} ms")

    st.markdown(
        f"""
    <div class="cv-kv">
        <b>Empirical Conclusion:</b>
        At padded dimensions
        <code>{exp_results['padded_shape'][0]} × {exp_results['padded_shape'][1]}</code>,
        the Mean Squared Error between spatial convolution
        (<code>cv2.filter2D</code>) and Fourier multiplication
        (<code>ifft2(fft2(f) * fft2(h))</code>) is
        <b>{exp_results['mse']:.3e}</b>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        Machine floating-point precision:
        $$
        \mathrm{MSE} \approx 0
        $$
        """,
    )

    st.markdown(
        """
        <div class="cv-kv">
            This confirms that, under the same padding and boundary assumptions,
            both operations produce the same filtered image.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2x2 Visual Validation Grid (width=400 per requirement)
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown("**A. Spatial Domain Output: $(f * h)(x, y)$**")
        st.image(
            exp_results["spatial_u8"],
            caption=f"Computed via 2D spatial convolution (cv2.filter2D, {exp_ksize}×{exp_ksize}, σ={exp_sigma}).",
            width=400,
        )
    with row1_col2:
        st.markdown(
            "**B. Fourier Domain Output: $\\mathcal{F}^{-1}\\{F(u,v) \\cdot H(u,v)\\}$**"
        )
        st.image(
            exp_results["freq_u8"],
            caption=f"Computed via zero-padded 2D FFT pointwise multiplication ({exp_ksize}×{exp_ksize}, σ={exp_sigma}).",
            width=400,
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown(
            "**C. Filtered Log-Magnitude Spectrum: $\\log(1 + |F(u,v)H(u,v)|)$**"
        )
        st.image(
            exp_results["spectrum_u8"],
            caption="Centered 2D FFT spectrum showing high-frequency attenuation around the DC center.",
            width=400,
        )
    with row2_col2:
        st.markdown(
            "**D. Amplified Residual Error Map: $|g_{\\text{spatial}} - g_{\\text{fourier}}|$**"
        )
        st.image(
            exp_results["err_heatmap"],
            caption=f"Min-Max normalized floating-point residual (Max absolute error = {exp_results['max_err']:.2e}).",
            width=400,
        )
