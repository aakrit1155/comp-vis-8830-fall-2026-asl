## assignment_02/blurring.py
import cv2
import numpy as np
import streamlit as st


def _init_session_state() -> None:
    """Initialize session state keys for the blurring module."""
    if "blur_uploader_key" not in st.session_state:
        st.session_state["blur_uploader_key"] = 0
    if "blur_applied" not in st.session_state:
        st.session_state["blur_applied"] = False
    if "cv_uploaded_img" not in st.session_state:
        st.session_state["cv_uploaded_img"] = None
    if "cv_is_gray" not in st.session_state:
        st.session_state["cv_is_gray"] = False


def _reset_blurring_state() -> None:
    """Callback to reset uploaded image and processed filter outputs in memory."""
    st.session_state["blur_uploader_key"] += 1
    st.session_state["blur_applied"] = False
    st.session_state["cv_uploaded_img"] = None
    st.session_state["cv_is_gray"] = False


def _decode_uploaded_image(uploaded_file) -> tuple[np.ndarray, bool]:
    """Decode uploaded image bytes directly in memory without touching disk."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if raw_img is None:
        raise ValueError("Unable to decode the uploaded image file.")

    # Handle Grayscale (2D), BGR (3-channel), and BGRA (4-channel)
    if raw_img.ndim == 2:
        return raw_img, True
    elif raw_img.shape[2] == 4:
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2RGB)
        return rgb_img, False
    else:
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        return rgb_img, False


def _apply_fft_lowpass(
    img: np.ndarray, cutoff: float, mode: str = "gaussian"
) -> np.ndarray:
    """
    Apply a 2D Frequency-Domain Low-Pass Filter (Ideal or Gaussian) per channel
    entirely in memory using NumPy/OpenCV DFT equivalents.
    """

    def _filter_channel(channel: np.ndarray) -> np.ndarray:
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # 2D FFT and shift DC component to center
        dft = np.fft.fft2(channel.astype(np.float32))
        dft_shift = np.fft.fftshift(dft)

        # Construct frequency distance matrix D(u, v)
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        V, U = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)

        if mode == "ideal":
            mask = (D <= cutoff).astype(np.float32)
        else:
            # Gaussian Low-Pass Filter: H(u,v) = exp(-D^2 / (2 * D0^2))
            mask = np.exp(-(D**2) / (2.0 * (cutoff**2) + 1e-8)).astype(np.float32)

        filtered_shift = dft_shift * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
        img_real = np.real(img_back)
        return np.clip(img_real, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _filter_channel(img)

    channels = cv2.split(img)
    filtered_channels = [_filter_channel(ch) for ch in channels]
    return cv2.merge(filtered_channels)


def _compute_all_blurs(
    img: np.ndarray, ksize: int, sigma: float, cutoff: float
) -> list[dict]:
    """Compute a comprehensive suite of spatial and frequency blurring filters."""
    # 1. Averaging (Box) Filter
    box_blur = cv2.blur(img, (ksize, ksize))

    # 2. Spatial Gaussian Filter
    gaussian_blur = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # 3. Median Filter (Non-linear order-statistic filter)
    median_blur = cv2.medianBlur(img, ksize)

    # 4. Bilateral Filter (Edge-preserving spatial + intensity filter)
    bilateral_blur = cv2.bilateralFilter(
        img, d=min(ksize, 15), sigmaColor=75, sigmaSpace=75
    )

    # 5. Custom 2D Motion Blur Kernel via cv2.filter2D
    motion_kernel = np.zeros((ksize, ksize), dtype=np.float32)
    motion_kernel[ksize // 2, :] = 1.0 / ksize
    motion_blur = cv2.filter2D(img, ddepth=-1, kernel=motion_kernel)

    # 6. Frequency Domain: Ideal Low-Pass Filter (ILPF)
    ideal_lpf = _apply_fft_lowpass(img, cutoff=cutoff, mode="ideal")

    # 7. Frequency Domain: Gaussian Low-Pass Filter (GLPF)
    gaussian_lpf = _apply_fft_lowpass(img, cutoff=cutoff, mode="gaussian")

    return [
        {
            "title": "1. Original Image (Baseline)",
            "image": img,
            "caption": f"Unfiltered input ({img.shape[1]}×{img.shape[0]} px).",
            "desc": "Reference image prior to spatial convolution or Fourier domain attenuation.",
        },
        {
            "title": f"2. Averaging / Box Filter ({ksize}×{ksize})",
            "image": box_blur,
            "caption": f"cv2.blur — Uniform {ksize}×{ksize} box kernel (1/{ksize*ksize}).",
            "desc": "Uniformly averages all neighbor pixels; blurs edges and introduces box artifacts.",
        },
        {
            "title": f"3. Gaussian Blur ({ksize}×{ksize}, σ={sigma})",
            "image": gaussian_blur,
            "caption": f"cv2.GaussianBlur — 2D isotropic Gaussian kernel (σ={sigma}).",
            "desc": "Weights center pixels more heavily than distant neighbors for smooth, natural blurring.",
        },
        {
            "title": f"4. Median Filter ({ksize}×{ksize})",
            "image": median_blur,
            "caption": f"cv2.medianBlur — Non-linear rank filter (window {ksize}×{ksize}).",
            "desc": "Replaces each pixel with the neighborhood median; suppresses impulse noise while preserving step edges.",
        },
        {
            "title": "5. Bilateral Filter (Edge-Preserving)",
            "image": bilateral_blur,
            "caption": f"cv2.bilateralFilter — d={min(ksize, 15)}, σ_color=75, σ_space=75.",
            "desc": "Combines a spatial Gaussian kernel with an intensity-difference Gaussian to smooth flat regions while keeping sharp edges intact.",
        },
        {
            "title": f"6. Custom 2D Convolution: Motion Blur ({ksize}×{ksize})",
            "image": motion_blur,
            "caption": f"cv2.filter2D — Horizontal directional kernel of length {ksize}.",
            "desc": "Demonstrates custom 2D linear spatial convolution simulating horizontal camera motion.",
        },
        {
            "title": f"7. Fourier Ideal Low-Pass Filter (D₀={int(cutoff)})",
            "image": ideal_lpf,
            "caption": f"2D FFT — Hard circular frequency cutoff at radius D₀={int(cutoff)}.",
            "desc": "Zeroes all frequencies outside radius D₀; causes visible spatial ringing (Gibbs phenomenon) due to sinc convolution.",
        },
        {
            "title": f"8. Fourier Gaussian Low-Pass Filter (D₀={int(cutoff)})",
            "image": gaussian_lpf,
            "caption": f"2D FFT — Smooth Gaussian frequency attenuation (D₀={int(cutoff)}).",
            "desc": "Smoothly attenuates high frequencies in the Fourier domain; produces ring-free blurring equivalent to a spatial Gaussian filter.",
        },
    ]


def render_blurring() -> None:
    """Render Tab 1: Interactive Image Blurring via Spatial & Frequency Filtering."""
    _init_session_state()

    st.markdown(
        '<div class="cv-section-title">📤 Step 1: Upload Image & Configure Filter Parameters</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="cv-note">
            Upload any <b>RGB</b> or <b>Grayscale</b> image. All decoding and filtering operations
            are executed strictly <b>in-memory</b> using OpenCV and NumPy buffers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image (PNG, JPG, JPEG, BMP, WEBP, TIFF)",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"],
        key=f"blur_uploader_{st.session_state['blur_uploader_key']}",
    )

    if uploaded_file is not None:
        try:
            img, is_gray = _decode_uploaded_image(uploaded_file)
            st.session_state["cv_uploaded_img"] = img
            st.session_state["cv_is_gray"] = is_gray
        except Exception as exc:
            st.error(f"Error decoding image: {exc}")
            return

        mode_label = "Grayscale (1-Channel)" if is_gray else "RGB Color (3-Channel)"
        st.caption(
            f"**Loaded in memory:** `{uploaded_file.name}` · "
            f"**Dimensions:** `{img.shape[1]} × {img.shape[0]}` · "
            f"**Format:** `{mode_label}`"
        )

        # Filter tuning controls
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            ksize = st.slider(
                "Spatial Kernel Size (k × k)",
                min_value=3,
                max_value=31,
                value=11,
                step=2,
                help="Odd kernel dimension used for Box, Gaussian, Median, and Motion filters.",
            )
        with col_p2:
            sigma = st.slider(
                "Gaussian Sigma (σ)",
                min_value=0.5,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Standard deviation for the 2D Gaussian spatial filter.",
            )
        with col_p3:
            cutoff = st.slider(
                "Fourier Cutoff Frequency (D₀)",
                min_value=5,
                max_value=150,
                value=35,
                step=5,
                help="Cutoff radius in the 2D FFT frequency domain for Ideal and Gaussian LPFs.",
            )

        # Action Buttons: Apply Blurring Filters & Reset
        btn_col1, btn_col2, _ = st.columns([1.4, 1.0, 3.6])
        with btn_col1:
            if st.button(
                "✨ Apply Blurring Filters",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["blur_applied"] = True
        with btn_col2:
            st.button(
                "🔄 Reset",
                on_click=_reset_blurring_state,
                use_container_width=True,
            )

        # Render 2-Column Comparison Grid once button is pressed
        if st.session_state["blur_applied"]:
            st.markdown("---")
            st.markdown(
                '<div class="cv-section-title">🔍 Step 2: Spatial & Frequency Blurring Comparison Grid</div>',
                unsafe_allow_html=True,
            )

            with st.spinner(
                "Computing spatial and frequency-domain blurring filters in memory..."
            ):
                results = _compute_all_blurs(
                    img, ksize=ksize, sigma=sigma, cutoff=float(cutoff)
                )

            # Display in a 2-column grid with fixed width=400
            for row_idx in range(0, len(results), 2):
                cols = st.columns(2)
                for col_idx in range(2):
                    item_idx = row_idx + col_idx
                    if item_idx < len(results):
                        item = results[item_idx]
                        with cols[col_idx]:
                            st.markdown(f"**{item['title']}**")
                            st.image(
                                item["image"],
                                caption=item["caption"],
                                width=400,
                            )
                            st.markdown(
                                f'<div class="cv-note">{item["desc"]}</div>',
                                unsafe_allow_html=True,
                            )
    else:
        st.info("👆 Please upload an image above to unlock the blurring controls.")
