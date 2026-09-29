## assignment_02/theory.py
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from assignment_02.styles import inject_theory_styles


def _load_demo_image_in_memory() -> tuple[np.ndarray, str, str]:
    """Load assets/fourier_demo.(jpg|png) into RAM, or synthesize a fallback phantom."""
    assets_dir = Path(__file__).resolve().parent / "assets"
    candidate = "fourier_demo.jpg"
    img_path = assets_dir / candidate
    if img_path.exists():
        raw_bytes = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded, candidate, f"{len(raw_bytes) / 1024.0:.1f} KB"
    return np.array([]), "", ""


def _execute_fourier_pipeline(
    bgr_img: np.ndarray, d0_hp: int = 28, d0_lp: int = 35
) -> dict:
    """Execute 2D FFT High-Pass Edge Detection and Low-Pass Region Segmentation in RAM."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    f_float = gray.astype(np.float32)

    # 1. 2D Forward FFT and DC Centering
    dft_shift = np.fft.fftshift(np.fft.fft2(f_float))
    mag_vis = cv2.normalize(
        20.0 * np.log10(np.abs(dft_shift) + 1.0),
        np.empty_like(20.0 * np.log10(np.abs(dft_shift) + 1.0)),
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    mag_vis_color = cv2.cvtColor(
        cv2.applyColorMap(mag_vis, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB
    )

    # 2. Centered Radial Frequency Grid D(u, v)
    crow, ccol = rows // 2, cols // 2
    u, v = np.ogrid[:rows, :cols]
    d_squared = (u - crow) ** 2 + (v - ccol) ** 2

    # 3. Gaussian High-Pass Filter H_GHPF(u,v) for Edge Detection
    h_ghpf = 1.0 - np.exp(-d_squared / (2.0 * (max(1, d0_hp) ** 2)))
    g_hp_shift = dft_shift * h_ghpf
    img_edges = np.abs(np.fft.ifft2(np.fft.ifftshift(g_hp_shift)))
    edges_norm = cv2.normalize(
        img_edges, np.empty_like(img_edges), 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, edges_binary = cv2.threshold(
        edges_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Gaussian Low-Pass Filter H_GLPF(u,v) for Homogeneous Region Segmentation
    h_glpf = np.exp(-d_squared / (2.0 * (max(1, d0_lp) ** 2)))
    g_lp_shift = dft_shift * h_glpf
    img_smooth_regions = np.abs(np.fft.ifft2(np.fft.ifftshift(g_lp_shift)))
    lp_norm = cv2.normalize(
        img_smooth_regions, np.empty_like(img_smooth_regions), 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 5. Correct Mask Polarity & Filter out White Backgrounds
    _, region_mask = cv2.threshold(lp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Auto-invert if the background (corners) is light/white
    corners_sum = (
        int(region_mask[0, 0])
        + int(region_mask[0, -1])
        + int(region_mask[-1, 0])
        + int(region_mask[-1, -1])
    )
    if corners_sum > 510:  # If majority of corners are white
        region_mask = cv2.bitwise_not(region_mask)

    # Morphological closing to solidify the human silhouette
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Annotate central human subject
    annotated_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).copy()
    contours, _ = cv2.findContours(
        region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours:
        # Ignore contours that map to the entire image bounds
        valid_contours = [
            c for c in contours if cv2.contourArea(c) < 0.95 * (rows * cols)
        ]
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(annotated_rgb, [largest], -1, (0, 255, 128), 3)

    # 6. Parseval Spectral Energy Partitioning Metrics
    total_energy = float(np.sum(np.abs(dft_shift) ** 2))
    lp_energy_pct = (
        ((float(np.sum(np.abs(g_lp_shift) ** 2)) / total_energy) * 100.0)
        if total_energy > 0
        else 0.0
    )
    hp_energy_pct = (
        ((float(np.sum(np.abs(g_hp_shift) ** 2)) / total_energy) * 100.0)
        if total_energy > 0
        else 0.0
    )

    return {
        "mag_vis_color": mag_vis_color,
        "edges_norm": edges_norm,
        "edges_binary": edges_binary,
        "lp_norm": lp_norm,
        "region_mask": region_mask,
        "annotated_rgb": annotated_rgb,
        "lp_energy_pct": lp_energy_pct,
        "hp_energy_pct": hp_energy_pct,
    }


def _render_fourier_demo() -> None:
    """Interactive UI section at the bottom of Tab 3 (Theory)."""
    if "theory_demo_active" not in st.session_state:
        st.session_state["theory_demo_active"] = False

    st.markdown("---")
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">6</span>
            <span class="label">Interactive Verification: 2D FFT Edge Detection &amp; Region Segmentation Demo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="cv-note">
            <strong>Live Frequency-Domain Benchmark:</strong> Applies the derived Gaussian High-Pass Filter
             to isolate spatial boundaries and the Gaussian Low-Pass Filter
             + Otsu thresholding to segment homogeneous regions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
                Gaussian High-Pass Filter: 
            $$H_{\\text{GHPF}}(u,v) = 1 - e^{-D^2(u,v)/2D_{0,\\text{HP}}^2}$$\n
            Gaussian Low-Pass Filter:
            $$H_{\\text{GLPF}}(u,v) = e^{-D^2(u,v)/2D_{0,\\text{LP}}^2}$$
            """)

    bgr_img, file_name, size_str = _load_demo_image_in_memory()
    h, w, ch = bgr_img.shape
    rgb_input = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # ---------------- Basic Info Above Original Image ----------------
    st.markdown("#### 📋 Demo Input Image Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source Image", file_name)
    m2.metric("Dimensions (W × H)", f"{w} × {h} px")
    m3.metric("Channels", f"{ch} Channels (RGB)")
    m4.metric("In-Memory Footprint", size_str)

    st.image(
        rgb_input,
        caption=f"Input Image for 2D FFT Analysis — {file_name} ({w}×{h} px)",
        width=600,
    )

    # ---------------- Cutoff Frequency Controls ----------------
    c1, c2 = st.columns(2)
    d0_hp = c1.slider(
        "High-Pass Edge Cutoff Radius (D₀_HP)",
        min_value=5,
        max_value=100,
        value=25,
        help="Attenuates low frequencies inside D₀_HP; retains high-frequency edge transitions.",
        key="theory_d0_hp",
    )
    d0_lp = c2.slider(
        "Low-Pass Region Cutoff Radius (D₀_LP)",
        min_value=5,
        max_value=100,
        value=30,
        help="Integrates smooth spatial energy inside D₀_LP for region segmentation.",
        key="theory_d0_lp",
    )

    # ---------------- Run Demo & Reset Buttons ----------------
    b1, b2, _ = st.columns([1.2, 1, 3.8])
    with b1:
        run_pressed = st.button(
            "Run Demo",
            key="theory_run_demo_btn",
            type="primary",
            use_container_width=True,
        )
    with b2:
        if st.button("🔄 Reset", key="theory_reset_demo_btn", use_container_width=True):
            st.session_state["theory_demo_active"] = False
            st.session_state.pop("theory_demo_results", None)
            st.rerun()

    if run_pressed:
        with st.spinner(
            "Computing 2D Fast Fourier Transform and inverse spatial reconstructions..."
        ):
            st.session_state["theory_demo_results"] = _execute_fourier_pipeline(
                bgr_img, d0_hp=d0_hp, d0_lp=d0_lp
            )
            st.session_state["theory_demo_active"] = True

    # ---------------- Render Outputs Only When Activated ----------------
    if (
        st.session_state.get("theory_demo_active")
        and "theory_demo_results" in st.session_state
    ):
        res = st.session_state["theory_demo_results"]
        st.markdown("#### 🎯 Frequency-Domain Segmentation & Edge Extraction Outputs")

        st.image(
            res["annotated_rgb"],
            caption="Annotated Region Boundary Extracted via Low-Pass Fourier Energy + Otsu Thresholding (Green)",
            width=600,
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Low-Pass Region Energy (Parseval)", f"{res['lp_energy_pct']:.2f}%")
        k2.metric("High-Pass Edge Energy (Parseval)", f"{res['hp_energy_pct']:.4f}%")
        k3.metric("FFT Grid Size", f"{w} × {h} Bins")

        demo_tabs = st.tabs(
            [
                "1️⃣ 2D FFT Log-Magnitude Spectrum",
                "2️⃣ High-Pass Edge Reconstruction",
                "3️⃣ Low-Pass Region Segmentation",
            ]
        )
        with demo_tabs[0]:
            st.image(
                res["mag_vis_color"],
                caption="Centered 2D Fourier Log-Magnitude Spectrum: 20·log₁₀(1 + |F(u,v)|)",
                width=600,
            )
        with demo_tabs[1]:
            st.image(
                res["edges_norm"],
                caption=f"Spatial Edge Magnitude Reconstructed from High-Pass Filter (D₀_HP = {d0_hp})",
                width=600,
            )
        with demo_tabs[2]:
            st.image(
                res["region_mask"],
                caption=f"Segmented Binary Region Mask from Low-Pass IFFT Reconstruction (D₀_LP = {d0_lp})",
                width=600,
            )


def render_theory() -> None:
    inject_theory_styles()

    st.markdown(
        """
        <div class="cv-theory-hero">
            <h2>📖 Mathematical Derivations: Edge Detection &amp; Region Segmentation in the Fourier Domain</h2>
            <p>
                A rigorous analytical treatment of how spatial differential operators, multi-scale boundary detectors,
                and region-based spectral segmentation map to transfer functions in the 2D frequency domain.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- STEP 1 ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">1</span>
            <span class="label">2D Fourier Transform &amp; The Convolution Theorem</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(r"""
        Let $f(x, y)$ denote a continuous 2D spatial image intensity function. Its **2D Continuous Fourier Transform** $\mathcal{F}\{f(x,y)\} = F(u, v)$ and inverse transform are defined as:

        $$F(u, v) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f(x, y)\, e^{-j2\pi(ux + vy)} \,dx\,dy$$

        $$f(x, y) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} F(u, v)\, e^{j2\pi(ux + vy)} \,du\,dv$$

        For a discrete digital image of size $M \times N$, the **2D Discrete Fourier Transform (DFT)** is given by:

        $$F(u, v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x, y)\, e^{-j2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)}, \quad u \in [0, M-1],\; v \in [0, N-1]$$

        By the **2D Convolution Theorem**, spatial linear filtering of an image $f(x,y)$ with a spatial kernel $h(x,y)$ is equivalent to point-wise multiplication in the frequency domain:

        $$g(x, y) = f(x, y) * h(x, y) \Longleftrightarrow G(u, v) = F(u, v)\cdot H(u, v)$$
        """)

    # ---------------- STEP 2 ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">2</span>
            <span class="label">First-Order Derivative Theorem &amp; Gradient Edge Detection</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(r"""
        Edges in an image correspond to sharp spatial discontinuities (high-frequency transitions) in intensity $f(x,y)$. To derive the frequency-domain equivalent of spatial differentiation, differentiate the inverse Fourier integral with respect to $x$:

        $$\frac{\partial f(x, y)}{\partial x} = \frac{\partial}{\partial x} \left[ \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} F(u, v)\, e^{j2\pi(ux + vy)} \,du\,dv \right] = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} (j2\pi u)\, F(u, v)\, e^{j2\pi(ux + vy)} \,du\,dv$$

        By induction for the $n$-th partial derivatives along $x$ and $y$:

        $$\mathcal{F}\left\{\frac{\partial^n f(x, y)}{\partial x^n}\right\} = (j2\pi u)^n F(u, v), \qquad \mathcal{F}\left\{\frac{\partial^n f(x, y)}{\partial y^n}\right\} = (j2\pi v)^n F(u, v)$$

        Therefore, the **spatial gradient vector** $\nabla f(x,y)$ maps directly to a pair of directional frequency-domain transfer functions $H_x(u,v) = j2\pi u$ and $H_y(u,v) = j2\pi v$:

        $$\nabla f(x, y) = \begin{bmatrix} g_x(x,y) \\ g_y(x,y) \end{bmatrix} = \begin{bmatrix} \mathcal{F}^{-1}\{j2\pi u\, F(u,v)\} \\ \mathcal{F}^{-1}\{j2\pi v\, F(u,v)\} \end{bmatrix}$$

        The **edge gradient magnitude** $M(x, y)$ is then recovered in the spatial domain via:

        $$M(x, y) = \|\nabla f(x, y)\|_2 = \sqrt{g_x^2(x, y) + g_y^2(x, y)}$$
        """)
    st.markdown(
        """
        <div class="cv-kv">
            <strong>Physical Intuition:</strong> Because the transfer function amplitude $|H_x(u,v)| = 2\\pi |u|$ grows linearly with radial frequency, differentiation acts as a pure <strong>High-Pass Filter</strong>—suppressing DC/constant regions ($u=v=0$) and amplifying sharp intensity transitions (edges).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- STEP 3 ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">3</span>
            <span class="label">Second-Order Laplacian &amp; Laplacian of Gaussian (LoG) Derivation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(r"""
        The isotropic second-order **Laplacian operator** $\nabla^2 f(x, y)$ locates exact edge boundaries via zero-crossings:

        $$\nabla^2 f(x, y) = \frac{\partial^2 f(x, y)}{\partial x^2} + \frac{\partial^2 f(x, y)}{\partial y^2}$$

        Applying the second-order Fourier derivative property ($n=2$):

        $$\mathcal{F}\{\nabla^2 f(x, y)\} = (j2\pi u)^2 F(u, v) + (j2\pi v)^2 F(u, v) = -4\pi^2(u^2 + v^2)\, F(u, v)$$

        Letting $D(u, v) = \sqrt{(u - M/2)^2 + (v - N/2)^2}$ represent the centered radial frequency distance, the **Laplacian Frequency Transfer Function** is:

        $$H_{\text{Lap}}(u, v) = -4\pi^2 D^2(u, v)$$

        Because $D^2(u,v)$ amplifies high-frequency sensor noise quadratically, **Marr-Hildreth (Laplacian of Gaussian — LoG)** regularizes the operator with a 2D Gaussian smoothing filter $H_{\text{LP}}(u,v) = e^{-2\pi^2 \sigma^2 (u^2 + v^2)}$:

        $$H_{\text{LoG}}(u, v) = -4\pi^2 D^2(u, v)\, e^{-2\pi^2 \sigma^2 D^2(u, v)}$$

        The exact edge boundary contour $\mathcal{E}$ is extracted as the zero-crossing locus of the inverse transform:

        $$\mathcal{E} = \left\{ (x, y) \;\middle|\; \mathcal{F}^{-1}\left\{ H_{\text{LoG}}(u, v)\, F(u, v) \right\} = 0 \right\}$$
        """)

    # ---------------- STEP 4 ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">4</span>
            <span class="label">Classical High-Pass Edge Extraction Transfer Functions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(r"""
        Beyond differential operators, edges are isolated by attenuating low frequencies below a cutoff radius $D_0$:

        1. **Ideal High-Pass Filter (IHPF):** Causes spatial ringing artifacts due to convolution with a $\text{sinc}$ impulse response (Gibbs phenomenon):
           $$H_{\text{IHPF}}(u, v) = \begin{cases} 0, & \text{if } D(u, v) \le D_0 \\ 1, & \text{if } D(u, v) > D_0 \end{cases}$$

        2. **Butterworth High-Pass Filter (BHPF) of order $n$:** Provides a smooth transition controlled by order $n$:
           $$H_{\text{BHPF}}(u, v) = \frac{1}{1 + \left[\dfrac{D_0}{D(u, v)}\right]^{2n}}$$

        3. **Gaussian High-Pass Filter (GHPF):** Eliminates ringing completely because the Fourier transform of a Gaussian remains Gaussian:
           $$H_{\text{GHPF}}(u, v) = 1 - e^{-\frac{D^2(u, v)}{2D_0^2}}$$
        """)

    # ---------------- STEP 5 ----------------
    st.markdown(
        """
        <div class="cv-step">
            <span class="badge">5</span>
            <span class="label">Region Segmentation via Frequency-Domain Spectral Partitioning</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(r"""
        While edges occupy the high-frequency periphery of the Fourier plane, **homogeneous regions** and **periodic textures** are segmented by partitioning radial frequency bands and angular orientations:

        #### A. Low-Pass & Band-Pass Difference of Gaussians (DoG) Region Isolation
        Smooth foreground objects of characteristic scale $\in [\sigma_2, \sigma_1]$ are isolated from both slow illumination drift and high-frequency edge noise using a **Band-Pass Difference of Gaussians (DoG)** filter:

        $$H_{\text{DoG}}(u, v) = e^{-\frac{D^2(u, v)}{2\sigma_2^2}} - e^{-\frac{D^2(u, v)}{2\sigma_1^2}}, \quad (\sigma_1 > \sigma_2)$$

        Reconstructing the filtered spatial region map $g_{\text{reg}}(x, y) = \mathcal{F}^{-1}\{H_{\text{DoG}}(u, v)\,F(u, v)\}$ and applying global thresholding yields closed connected regions:

        $$R(x, y) = \begin{cases} 1, & \text{if } \mathcal{F}^{-1}\{H_{\text{DoG}}(u, v)\,F(u, v)\} \ge \tau \\ 0, & \text{otherwise} \end{cases}$$

        #### B. Multi-Channel Gabor & Wedge-Ring Spectral Texture Segmentation
        When regions differ in texture rather than mean intensity, a 2D **Gabor filter bank** acts as a localized directional band-pass filter centered at frequency $(u_0, v_0)$:

        $$H_{\text{Gabor}}(u, v) = e^{-2\pi^2 \left[ \sigma_x^2 (u - u_0)^2 + \sigma_y^2 (v - v_0)^2 \right]}$$

        By **Parseval's Theorem**, total spatial energy is preserved in the spectral domain:

        $$\sum_{x=0}^{M-1}\sum_{y=0}^{N-1} |f(x, y)|^2 = \frac{1}{MN} \sum_{u=0}^{M-1}\sum_{v=0}^{N-1} |F(u, v)|^2$$

        Partitioning the Fourier plane into radial rings $S_{r_1, r_2}$ (scale) and angular wedges $S_{\theta_1, \theta_2}$ (orientation) in polar coordinates $(D, \theta)$ extracts a spectral feature vector $\mathbf{z}(x,y)$ for clustering distinct image regions:

        $$E(r_1, r_2, \theta_1, \theta_2) = \int_{r_1}^{r_2}\int_{\theta_1}^{\theta_2} |F(D, \theta)|^2 \, D\, dD\, d\theta$$
        """)
    st.markdown(
        """
        <div class="cv-note">
            <strong>Synthesis — Unifying Edges and Regions in Fourier Space:</strong>
            Low-pass filtering extracts region interiors by integrating homogeneous spatial energy, whereas high-pass/band-pass filtering extracts region boundaries. Combining both yields a complete dual segmentation framework in the frequency domain.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
            Low-pass filtering: ($$D(u,v) \\le D_0$$)\n
            High-pass/band-pass filtering: ($$D(u,v) > D_0$$)
            """)

    _render_fourier_demo()
