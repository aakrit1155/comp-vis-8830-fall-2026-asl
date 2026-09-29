import cv2
import numpy as np
import streamlit as st


def _init_thermal_state() -> None:
    if "thermal_uploader_key" not in st.session_state:
        st.session_state["thermal_uploader_key"] = 0
    if "thermal_applied" not in st.session_state:
        st.session_state["thermal_applied"] = False
    if "thermal_last_file_id" not in st.session_state:
        st.session_state["thermal_last_file_id"] = None


def _reset_thermal_state() -> None:
    st.session_state["thermal_uploader_key"] += 1
    st.session_state["thermal_applied"] = False
    st.session_state["thermal_last_file_id"] = None
    st.session_state.pop("thermal_results", None)


def _decode_thermal_image(uploaded_file) -> tuple[np.ndarray, str, int]:
    """Decode thermal image strictly in memory."""
    raw_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Could not decode the uploaded thermal image.")

    if decoded.ndim == 2:
        mode = "Radiometric Grayscale (1-Channel)"
        channels = 1
        bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    elif decoded.shape[2] == 4:
        mode = "False-Color Thermal RGBA (4-Channel)"
        channels = 4
        bgr = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
    else:
        mode = "Thermal RGB / Pseudo-Color (3-Channel)"
        channels = 3
        bgr = decoded

    return bgr, mode, channels


def _extract_human_boundary_thermal(
    bgr_img: np.ndarray,
    polarity: str = "White-Hot / Warm-Bright (Standard)",
    morph_kernel_size: int = 5,
    clahe_clip: float = 3.0,
    boundary_thickness: int = 3,
) -> dict:
    """Fast Classical OpenCV boundary extraction tailored for Thermal Infrared imagery."""
    h, w = bgr_img.shape[:2]

    # 1. OPTIMIZATION: Downscale image to max 320px for fast bilateral filtering
    scale = 1.0
    max_dim = 320.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        work_img = cv2.resize(
            bgr_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        work_img = bgr_img.copy()

    # 2. Extract Thermal Luminance & Enhance (on downscaled image)
    lab = cv2.cvtColor(work_img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    norm_gray = cv2.normalize(
        l_channel, np.empty_like(l_channel), 0, 255, cv2.NORM_MINMAX
    )

    if polarity.startswith("Black-Hot"):
        norm_gray = cv2.bitwise_not(norm_gray)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced_thermal_small = clahe.apply(norm_gray)
    smoothed = cv2.bilateralFilter(
        enhanced_thermal_small, d=5, sigmaColor=50, sigmaSpace=50
    )

    # 3. Thresholding
    otsu_thresh, _ = cv2.threshold(
        smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    high_pct_val = np.percentile(smoothed, 75)
    effective_thresh = max(float(otsu_thresh), float(high_pct_val * 0.85))
    _, thermal_mask_small = cv2.threshold(
        smoothed, int(effective_thresh), 255, cv2.THRESH_BINARY
    )

    # 4. Upscale mask and enhanced thermal back to original High-Res dimensions
    if scale != 1.0:
        thermal_mask = cv2.resize(
            thermal_mask_small, (w, h), interpolation=cv2.INTER_NEAREST
        )
        enhanced_thermal = cv2.resize(
            enhanced_thermal_small, (w, h), interpolation=cv2.INTER_CUBIC
        )
    else:
        thermal_mask = thermal_mask_small
        enhanced_thermal = enhanced_thermal_small

    # 5. Morphological Reconstruction (on full resolution)
    k = max(3, morph_kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(thermal_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. Filter Human-Scale Connected Thermal Contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clean_mask = np.zeros_like(opened)
    valid_contours = []
    min_area = 0.005 * (h * w)

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid_contours:
            valid_contours = [max(contours, key=cv2.contourArea)]
        cv2.drawContours(clean_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

    # 7. Exact Morphological Boundary: Beta(A) = A - (A erode B)
    b_k = max(3, boundary_thickness | 1)
    b_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b_k, b_k))
    eroded = cv2.erode(clean_mask, b_kern, iterations=1)
    exact_boundary = cv2.subtract(clean_mask, eroded)

    # 8. Build Annotated Outputs
    annotated_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).copy()
    total_area = sum(cv2.contourArea(c) for c in valid_contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in valid_contours)

    for idx, cnt in enumerate(valid_contours, start=1):
        cv2.drawContours(
            annotated_rgb, [cnt], -1, (0, 255, 180), thickness=boundary_thickness
        )
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(annotated_rgb, (x, y), (x + bw, y + bh), (255, 140, 0), 2)
        cv2.putText(
            annotated_rgb,
            f"Thermal Human #{idx}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )

    pseudo_heatmap = cv2.cvtColor(
        cv2.applyColorMap(enhanced_thermal, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB
    )

    return {
        "annotated_rgb": annotated_rgb,
        "exact_boundary": exact_boundary,
        "clean_mask": clean_mask,
        "pseudo_heatmap": pseudo_heatmap,
        "area_px": float(total_area),
        "perimeter_px": float(total_perimeter),
        "subject_count": len(valid_contours),
        "otsu_thresh": float(otsu_thresh),
    }


def render_boundary_thermal() -> None:
    _init_thermal_state()

    st.markdown(
        '<div class="cv-section-title">Question 2: Classical Human Boundary Extraction (Thermal Infrared Camera)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="cv-note">
            <strong>Thermal Pipeline Overview:</strong> Segments human thermal signatures from LWIR/MWIR imagery
            using radiometric normalization &rarr; CLAHE local contrast boosting &rarr; Bilateral edge-preserving
            smoothing &rarr; Otsu + Percentile Radiometric Thresholding &rarr; Morphological Boundary Extraction
            <code>&beta;(A) = A &minus; (A &ominus; B)</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "🌡️ Upload a Thermal Infrared Image (Grayscale or Pseudo-Color)",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        key=f"thermal_file_{st.session_state['thermal_uploader_key']}",
    )

    if uploaded_file is None:
        st.info(
            "Upload a thermal infrared image above to inspect its metadata and extract human boundaries."
        )
        return

    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state["thermal_last_file_id"] != file_id:
        st.session_state["thermal_last_file_id"] = file_id
        st.session_state["thermal_applied"] = False
        st.session_state.pop("thermal_results", None)

    bgr_img, color_mode, channels = _decode_thermal_image(uploaded_file)
    h, w = bgr_img.shape[:2]
    size_kb = len(uploaded_file.getvalue()) / 1024.0
    rgb_original = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # ---------------- Original Image Metadata (Displayed Above Image) ----------------
    st.markdown("#### 📋 Original Thermal Image Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("File Name", uploaded_file.name)
    m2.metric("Dimensions (W × H)", f"{w} × {h} px")
    m3.metric("Sensor Encoding", f"{color_mode} ({channels}ch)")
    m4.metric("In-Memory Size", f"{size_kb:.1f} KB")

    st.image(
        rgb_original,
        caption=f"Original Thermal Input — {uploaded_file.name} ({w}×{h} px)",
        width=600,
    )

    # ---------------- Hyperparameters ----------------
    with st.expander("⚙️ Thermal Pipeline Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        polarity = c1.selectbox(
            "Thermal Polarity",
            [
                "White-Hot / Warm-Bright (Standard)",
                "Black-Hot / Warm-Dark (Inverted)",
            ],
            key="thermal_polarity",
        )
        clahe_clip = c2.slider(
            "CLAHE Clip Limit", 1.0, 6.0, 3.0, 0.5, key="thermal_clahe"
        )
        morph_k = c3.slider(
            "Morphological Kernel (px)", 3, 15, 5, step=2, key="thermal_kernel"
        )

    # ---------------- Action Buttons ----------------
    btn_col1, btn_col2, _ = st.columns([1.4, 1, 3.6])
    with btn_col1:
        apply_clicked = st.button(
            "Apply Boundary Extraction",
            key="thermal_apply_btn",
            type="primary",
            use_container_width=True,
        )
    with btn_col2:
        st.button(
            "🔄 Reset",
            key="thermal_reset_btn",
            on_click=_reset_thermal_state,
            use_container_width=True,
        )

    if apply_clicked:
        with st.spinner("Extracting thermal human boundaries in memory..."):
            st.session_state["thermal_results"] = _extract_human_boundary_thermal(
                bgr_img,
                polarity=polarity,
                morph_kernel_size=morph_k,
                clahe_clip=clahe_clip,
            )
            st.session_state["thermal_applied"] = True

    # ---------------- Results Display ----------------
    if (
        st.session_state.get("thermal_applied")
        and "thermal_results" in st.session_state
    ):
        res = st.session_state["thermal_results"]
        st.markdown("---")
        st.markdown("#### 🎯 Annotated Thermal Human Boundary")

        st.image(
            res["annotated_rgb"],
            caption="Annotated Thermal Human Contour (Cyan-Green) & Bounding ROI (Orange)",
            width=600,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Detected Thermal Subjects", f"{res['subject_count']}")
        r2.metric("Boundary Perimeter", f"{res['perimeter_px']:,.1f} px")
        r3.metric("Segmented Area", f"{res['area_px']:,.0f} px²")
        r4.metric("Otsu Intensity Cutoff", f"{res['otsu_thresh']:.0f} / 255")

        st.markdown("#### 🔍 Intermediate Thermal Processing Stages")
        sub_tabs = st.tabs(
            [
                "1️⃣ Exact Morphological Boundary β(A)",
                "2️⃣ Binary Thermal Mask",
                "3️⃣ CLAHE Radiometric Heatmap",
            ]
        )
        with sub_tabs[0]:
            st.image(
                res["exact_boundary"],
                caption="Exact Thermal Silhouette Boundary: β(A) = A − (A ⊖ B)",
                width=600,
            )
        with sub_tabs[1]:
            st.image(
                res["clean_mask"],
                caption="Segmented Human Thermal Mask (Otsu + Morphological Closing)",
                width=600,
            )
        with sub_tabs[2]:
            st.image(
                res["pseudo_heatmap"],
                caption="CLAHE-Enhanced Radiometric Distribution (Inferno Colormap)",
                width=600,
            )

        st.markdown("#### ⚖️ Comparison with Meta SAM2 on Thermal Imagery")
        st.markdown("""
            | Criterion | Classical OpenCV Thermal Pipeline (Ours) | Meta SAM2 on Thermal Imagery |
            | :--- | :--- | :--- |
            | **Signal Exploitation** | Directly thresholds emitted Planck radiation intensity ($8\\text{--}14\\,\\mu\\text{m}$ LWIR) where humans naturally decouple from ambient backgrounds | Treats thermal grayscale/false-color images as out-of-distribution RGB textures |
            | **Boundary Fidelity** | **Near-identical to SAM2** on unobstructed pedestrians because thermal images lack RGB clothing/shadow clutter | Strong contour closure across heavy thermal halo/blooming artifacts |
            | **Failure Modes** | Ambient heat sources (car engines, hot pavement) close to $37^\\circ\\text{C}$ can merge into the contour | Zero-shot prompts occasionally over-segment false-color thermal background gradients |
            """)
