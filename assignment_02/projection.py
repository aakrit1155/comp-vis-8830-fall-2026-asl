import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

IMG_WIDTH = 400


def resize_for_annotation(img_bgr, max_width=400):
    """
    Resize image only for UI interaction while preserving aspect ratio.

    Returns:
        resized_image
        scale = resized_width / original_width
    """
    h, w = img_bgr.shape[:2]

    if w <= max_width:
        return img_bgr.copy(), 1.0

    scale = max_width / w

    new_width = int(w * scale)
    new_height = int(h * scale)

    resized = cv2.resize(
        img_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def calculate_real_width_from_points(
    point1,
    point2,
    depth_z,
    camera_matrix,
    dist_coeffs,
):
    """
    Estimate real-world distance between two image points.

    Assumption:
        Both selected points lie on the same plane at camera
        depth Z (parallel to the image plane).

    Returns:
        pixel_distance      - distance in original image pixels
        real_distance       - estimated physical distance in mm
        undistorted_points  - normalized camera coordinates
    """

    points = np.array(
        [[point1], [point2]],
        dtype=np.float32,
    )

    # Remove lens distortion and convert pixel coordinates
    # into normalized camera coordinates:
    #
    # [u, v] -> [x_n, y_n]
    #
    # With P omitted, OpenCV returns normalized coordinates.
    normalized_points = cv2.undistortPoints(
        points,
        camera_matrix,
        dist_coeffs,
    )

    x1, y1 = normalized_points[0, 0]
    x2, y2 = normalized_points[1, 0]

    # Pixel distance for display/reference
    pixel_distance = np.linalg.norm(
        np.asarray(point2, dtype=np.float32) - np.asarray(point1, dtype=np.float32)
    )

    # Reconstruct the two 3D points assuming both have
    # the same camera-space depth Z.
    P1 = np.array(
        [
            x1 * depth_z,
            y1 * depth_z,
            depth_z,
        ]
    )

    P2 = np.array(
        [
            x2 * depth_z,
            y2 * depth_z,
            depth_z,
        ]
    )

    # Euclidean 3D distance
    real_distance = np.linalg.norm(P2 - P1)

    return pixel_distance, real_distance


def draw_two_point_measurement(
    img_bgr,
    point1,
    point2,
    real_width,
    unit="mm",
):
    """Draw measurement annotation directly on the displayed image."""

    annotated = img_bgr.copy()

    p1 = tuple(map(int, point1))
    p2 = tuple(map(int, point2))

    # Measurement line
    cv2.line(
        annotated,
        p1,
        p2,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    # Larger end points
    cv2.circle(
        annotated,
        p1,
        3,
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        annotated,
        p2,
        3,
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )

    label = f"Width: {real_width:.2f} {unit}"

    # Larger, more readable text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    # Place label above the measurement line
    label_x = min(p1[0], p2[0])
    label_y = max(45, min(p1[1], p2[1]) - 20)

    # Keep label inside image boundaries
    h, w = annotated.shape[:2]

    if label_x + tw + 20 > w:
        label_x = max(5, w - tw - 20)

    if label_y - th - 15 < 0:
        label_y = th + 20

    # Background rectangle for readability
    cv2.rectangle(
        annotated,
        (
            label_x - 5,
            label_y - th - 12,
        ),
        (
            label_x + tw + 10,
            label_y + baseline + 8,
        ),
        (0, 255, 0, 0.6),
        -1,
    )

    cv2.putText(
        annotated,
        label,
        (label_x, label_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    return annotated


def projection_render():
    st.markdown(
        '<div class="cv-section-title">📐 Compute Real-World 2D Dimensions</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Select two points on the object. "
        "The real-world width is estimated using the pinhole camera model: "
        "W = (pixel distance × Z) / fx."
    )

    # Automatically fetch fx and fy from session state
    if "proj_fx" not in st.session_state:
        st.session_state["proj_fx"] = 1200.0

    if "proj_fy" not in st.session_state:
        st.session_state["proj_fy"] = 1200.0

    if st.session_state.get("calibration_matrix") is not None:
        st.success(
            f"Intrinsic values loaded from calibration: "
            f"**fx = {st.session_state['proj_fx']:.2f} px**, "
            f"**fy = {st.session_state['proj_fy']:.2f} px**",
            icon="✅",
        )
    else:
        st.warning(
            "No calibration found — using default focal lengths (1200 px). "
            "Run Question 1 for accurate results.",
            icon="⚠️",
        )

    col_fx, col_fy = st.columns(2)

    with col_fx:
        fx = st.number_input(
            "Focal Length fx (px)",
            min_value=1.0,
            key="proj_fx",
        )

    with col_fy:
        fy = st.number_input(
            "Focal Length fy (px)",
            min_value=1.0,
            key="proj_fy",
        )

    obj_file = st.file_uploader(
        "Upload Object Target Image",
        type=["jpg", "jpeg", "png"],
        key="proj_img",
    )

    if obj_file is not None:

        # ---------------------------------------------------------
        # IMPORTANT:
        # Use getvalue() instead of read().
        # getvalue() does not consume the UploadedFile stream.
        # ---------------------------------------------------------
        file_bytes = np.frombuffer(
            obj_file.getvalue(),
            dtype=np.uint8,
        )

        raw_img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR,
        )

        if raw_img is None:
            st.error("Unable to read the uploaded image.")
            return

        # ---------------------------------------------------------
        # Measurement distance
        # ---------------------------------------------------------
        depth_z = st.number_input(
            "Measured Distance Z (mm)",
            min_value=1.0,
            value=2500.0,
            step=10.0,
            key="measurement_z",
        )

        st.markdown("### Select two points on the object")

        st.caption(
            "Click the first point and then the second point. "
            "The distance between the two points will be converted "
            "to real-world width using Z and fx."
        )

        # ---------------------------------------------------------
        # Initialize session state
        # ---------------------------------------------------------
        if "measurement_points" not in st.session_state:
            st.session_state["measurement_points"] = []

        # ---------------------------------------------------------
        # Reset
        # ---------------------------------------------------------
        if st.button(
            "↻ Reset Points",
            key="reset_measurement",
        ):
            st.session_state["measurement_points"] = []
            st.rerun()

        # ---------------------------------------------------------
        # Resize ONLY for display / interaction
        # ---------------------------------------------------------
        display_img_bgr, display_scale = resize_for_annotation(
            raw_img,
            max_width=IMG_WIDTH,
        )

        points = st.session_state["measurement_points"]

        if len(points) == 1:
            display_point1 = (
                int(round(points[0][0] * display_scale)),
                int(round(points[0][1] * display_scale)),
            )

            cv2.circle(
                display_img_bgr,
                display_point1,
                2,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

        display_img_rgb = cv2.cvtColor(
            display_img_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # ---------------------------------------------------------
        # Interactive image
        # ---------------------------------------------------------
        points_count = len(st.session_state["measurement_points"])
        if points_count != 2:
            click = streamlit_image_coordinates(
                display_img_rgb,
                key=f"measurement_image_{points_count}",
                cursor="crosshair",
            )

            # ---------------------------------------------------------
            # Accept the click
            # ---------------------------------------------------------
            if click is not None:

                display_point = (
                    int(click["x"]),
                    int(click["y"]),
                )

                # Convert display coordinates back to original-image
                # coordinates.
                original_point = (
                    int(round(display_point[0] / display_scale)),
                    int(round(display_point[1] / display_scale)),
                )

                if points_count < 2:
                    st.session_state["measurement_points"].append(original_point)

                else:
                    # Third click starts a new measurement.
                    st.session_state["measurement_points"] = [original_point]

                st.rerun()

        points = st.session_state["measurement_points"]

        # ---------------------------------------------------------
        # Show the selected points / measurement
        # ---------------------------------------------------------
        if len(points) == 0:

            st.info("Click the first point on the object.")

        elif len(points) == 1:

            point1 = points[0]

            st.info(
                f"First point selected: "
                f"({point1[0]}, {point1[1]}) px. "
                "Click the second point."
            )

        elif len(points) == 2:

            point1, point2 = points

            camera_matrix = st.session_state.get("calibration_matrix")
            dist_coeffs = st.session_state.get("dist_coeffs")

            if camera_matrix is None or dist_coeffs is None:
                st.error(
                    "Camera calibration is required before measuring "
                    "real-world dimensions."
                )
                st.stop()

            pixel_width, real_width = calculate_real_width_from_points(
                point1,
                point2,
                depth_z,
                camera_matrix,
                dist_coeffs,
            )

            # Resize first for UI display
            annotated_display, display_scale = resize_for_annotation(
                raw_img,
                max_width=IMG_WIDTH,
            )

            # Convert original-image coordinates to resized-image coordinates
            display_point1 = (
                int(round(point1[0] * display_scale)),
                int(round(point1[1] * display_scale)),
            )

            display_point2 = (
                int(round(point2[0] * display_scale)),
                int(round(point2[1] * display_scale)),
            )

            # Draw annotation directly on the resized image
            annotated_display = draw_two_point_measurement(
                annotated_display,
                display_point1,
                display_point2,
                real_width,
                unit="mm",
            )

            st.success("Measurement completed.")

            st.image(
                cv2.cvtColor(
                    annotated_display,
                    cv2.COLOR_BGR2RGB,
                ),
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Pixel Distance",
                    f"{pixel_width:.2f} px",
                )

            with col2:
                st.metric(
                    "Real-World Width",
                    f"{real_width:.2f} mm ({real_width/10.0:.2f} cm, {real_width/25.4:.2f} in)",
                )


def render_validation_stats():
    # ---------------- TAB 3: VALIDATION STATS ----------------
    st.markdown(
        '<div class="cv-section-title">📊 Validation Experiment (>2.0 m)</div>',
        unsafe_allow_html=True,
    )

    # Top tip notice
    st.markdown(
        """
        <div class="cv-note">
            <b>Tip:</b> The CSV file is expected in the format of the given sample table.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5-Row Sample Experiment Data
    st.subheader("Sample Experiment Data")
    sample_df = pd.DataFrame(
        {
            "Sample": ["Obj_01", "Obj_02", "Obj_03", "Obj_04", "Obj_05"],
            "Distance_mm": [2500.0, 2500.0, 2600.0, 2750.0, 3000.0],
            "Ground_Truth_Width_mm": [150.0, 200.0, 180.0, 240.0, 300.0],
            "Projected_Width_mm": [148.2, 203.5, 177.8, 244.1, 295.4],
        }
    )
    st.dataframe(sample_df, width="stretch", hide_index=True)

    # Optional helper: allow downloading the sample structure directly as a CSV template
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_validation_data.csv",
        mime="text/csv",
    )

    st.divider()

    # CSV Uploader
    uploaded_csv = st.file_uploader(
        "Upload Validation CSV File", type=["csv"], key="val_csv_uploader"
    )

    if uploaded_csv is not None:
        try:
            val_df = pd.read_csv(uploaded_csv)

            required_cols = {
                "Sample",
                "Ground_Truth_Width_mm",
                "Projected_Width_mm",
            }
            if not required_cols.issubset(val_df.columns):
                missing = required_cols - set(val_df.columns)
                st.error(
                    f"Missing required columns in uploaded CSV: {', '.join(missing)}. "
                    "Please match the Sample Experiment Data format above."
                )
            else:
                # Compute error & accuracy columns
                gt = val_df["Ground_Truth_Width_mm"].astype(float)
                proj = val_df["Projected_Width_mm"].astype(float)

                val_df["Abs_Error_mm"] = (proj - gt).abs()
                val_df["Signed_Error_mm"] = proj - gt
                val_df["Rel_Error_%"] = (val_df["Abs_Error_mm"] / gt) * 100.0
                val_df["Accuracy_%"] = (100.0 - val_df["Rel_Error_%"]).clip(lower=0.0)

                # Summary Statistical Metrics
                mae = val_df["Abs_Error_mm"].mean()
                rmse = np.sqrt((val_df["Signed_Error_mm"] ** 2).mean())
                mape = val_df["Rel_Error_%"].mean()
                mean_acc = val_df["Accuracy_%"].mean()

                st.subheader("Statistical Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean Abs Error (MAE)", f"{mae:.2f} mm")
                m2.metric("Root Mean Sq Error (RMSE)", f"{rmse:.2f} mm")
                m3.metric("Mean Rel Error (MAPE)", f"{mape:.2f} %")
                m4.metric("Mean Accuracy", f"{mean_acc:.2f} %")

                # Display Processed Data Table
                with st.expander("📋 View Full Evaluated Dataset", expanded=True):
                    st.dataframe(
                        val_df.style.format(
                            {
                                "Ground_Truth_Width_mm": "{:.2f}",
                                "Projected_Width_mm": "{:.2f}",
                                "Abs_Error_mm": "{:.2f}",
                                "Signed_Error_mm": "{:+.2f}",
                                "Rel_Error_%": "{:.2f}%",
                                "Accuracy_%": "{:.2f}%",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                st.subheader("Accuracy & Error Visualizations")

                # --- CHART 1 & 2: Side-by-Side Comparison and Parity Plot ---
                col_ch1, col_ch2 = st.columns(2)

                with col_ch1:
                    # Grouped Bar Chart: Ground Truth vs Projected Width
                    comp_df = val_df.melt(
                        id_vars=["Sample"],
                        value_vars=["Ground_Truth_Width_mm", "Projected_Width_mm"],
                        var_name="Measurement Type",
                        value_name="Width (mm)",
                    )
                    comp_df["Measurement Type"] = comp_df["Measurement Type"].replace(
                        {
                            "Ground_Truth_Width_mm": "Ground Truth (mm)",
                            "Projected_Width_mm": "Projected (mm)",
                        }
                    )

                    comparison_chart = (
                        alt.Chart(comp_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Sample:N", title="Sample"),
                            y=alt.Y("Width (mm):Q", title="Width (mm)"),
                            color=alt.Color(
                                "Measurement Type:N",
                                scale=alt.Scale(range=["#2563eb", "#10b981"]),
                            ),
                            xOffset="Measurement Type:N",
                            tooltip=[
                                "Sample",
                                "Measurement Type",
                                alt.Tooltip("Width (mm):Q", format=".2f"),
                            ],
                        )
                        .properties(
                            title="Ground Truth vs. Projected Width",
                            height=320,
                        )
                    )
                    st.altair_chart(comparison_chart, width="stretch")

                with col_ch2:
                    # Parity Plot (Scatter + Ideal y = x line)
                    min_val = float(min(gt.min(), proj.min()) * 0.9)
                    max_val = float(max(gt.max(), proj.max()) * 1.1)
                    line_df = pd.DataFrame(
                        {
                            "Ground_Truth_Width_mm": [min_val, max_val],
                            "Projected_Width_mm": [min_val, max_val],
                        }
                    )

                    ideal_line = (
                        alt.Chart(line_df)
                        .mark_line(strokeDash=[5, 5], color="gray")
                        .encode(
                            x=alt.X(
                                "Ground_Truth_Width_mm:Q",
                                scale=alt.Scale(domain=[min_val, max_val]),
                                title="Ground Truth Width (mm)",
                            ),
                            y=alt.Y(
                                "Projected_Width_mm:Q",
                                scale=alt.Scale(domain=[min_val, max_val]),
                                title="Projected Width (mm)",
                            ),
                        )
                    )

                    scatter_points = (
                        alt.Chart(val_df)
                        .mark_circle(size=90, color="#6366f1")
                        .encode(
                            x="Ground_Truth_Width_mm:Q",
                            y="Projected_Width_mm:Q",
                            tooltip=[
                                "Sample",
                                alt.Tooltip("Ground_Truth_Width_mm:Q", format=".2f"),
                                alt.Tooltip("Projected_Width_mm:Q", format=".2f"),
                                alt.Tooltip("Abs_Error_mm:Q", format=".2f"),
                                alt.Tooltip("Rel_Error_%:Q", format=".2f"),
                            ],
                        )
                    )

                    parity_chart = (ideal_line + scatter_points).properties(
                        title="Parity Plot (Projected vs. Ground Truth)",
                        height=320,
                    )
                    st.altair_chart(parity_chart, width="stretch")

                # --- CHART 3: Relative Error (%) Across Samples with Mean Line ---
                err_bars = (
                    alt.Chart(val_df)
                    .mark_bar(color="#f59e0b")
                    .encode(
                        x=alt.X("Sample:N", title="Sample"),
                        y=alt.Y("Rel_Error_%:Q", title="Relative Error (%)"),
                        tooltip=[
                            "Sample",
                            alt.Tooltip(
                                "Abs_Error_mm:Q", title="Abs Error (mm)", format=".2f"
                            ),
                            alt.Tooltip(
                                "Rel_Error_%:Q", title="Rel Error (%)", format=".2f"
                            ),
                        ],
                    )
                )

                mean_rule = (
                    alt.Chart(val_df)
                    .mark_rule(color="#ef4444", strokeDash=[4, 4], size=2)
                    .encode(y="mean(Rel_Error_%):Q")
                )

                error_chart = (err_bars + mean_rule).properties(
                    title=f"Relative Error (%) per Sample (Red Dashed Line = Mean: {mape:.2f}%)",
                    height=300,
                )
                st.altair_chart(error_chart, width="stretch")

        except Exception as e:
            st.error(f"Failed to read or process CSV file: {e}")
