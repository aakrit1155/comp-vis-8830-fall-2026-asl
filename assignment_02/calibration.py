import cv2
import numpy as np
import pandas as pd
import streamlit as st


def run_camera_calibration(uploaded_files, pattern_size=(9, 6), square_size=25.0):
    """
    Calibrates camera using uploaded checkerboard image buffers in memory.
    pattern_size: (internal_corners_x, internal_corners_y)
    square_size: physical side length of checkerboard square in mm
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        * square_size
    )

    objpoints = []
    imgpoints = []
    annotated_previews = []
    img_shape = None

    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = (int(gray.shape[1]), int(gray.shape[0]))
            # img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(refined_corners)

            preview = img.copy()
            cv2.drawChessboardCorners(preview, pattern_size, refined_corners, ret)
            annotated_previews.append((file.name, preview))
        else:
            annotated_previews.append((file.name, None))

    if len(objpoints) == 0:
        return None, None, None, None, annotated_previews

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # Compute mean re-projection error
    total_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        observed = np.asarray(imgpoints[i], dtype=np.float32).reshape(-1, 2)
        projected = np.asarray(imgpoints2, dtype=np.float32).reshape(-1, 2)

        error = np.linalg.norm(observed - projected) / len(observed)
        total_error += error
    mean_error = total_error / len(objpoints)

    return ret, mtx, dist, mean_error, annotated_previews


def calibration_render():
    st.markdown(
        '<div class="cv-section-title">🔧 Camera Calibration via OpenCV</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload 10–20 checkerboard images captured from varied angles for best results."
    )
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        cb_cols = st.number_input(
            "Inner Corners Width (X)", min_value=3, max_value=20, value=9
        )
    with col_cfg2:
        cb_rows = st.number_input(
            "Inner Corners Height (Y)", min_value=3, max_value=20, value=6
        )
    with col_cfg3:
        sq_size = st.number_input(
            "Square Size (mm)", min_value=1.0, max_value=200.0, value=21.0
        )

    uploaded_calib_imgs = st.file_uploader(
        "Upload Checkerboard Images (JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="calib_uploader",
    )

    if st.button("Execute Calibration", key="btn_run_calib"):
        if not uploaded_calib_imgs or len(uploaded_calib_imgs) < 3:
            st.error("Upload at least 3 calibration patterns (10-20 recommended).")
        else:
            with st.spinner("Processing calibration..."):
                ret, mtx, dist, err, previews = run_camera_calibration(
                    uploaded_calib_imgs,
                    pattern_size=(cb_cols, cb_rows),
                    square_size=sq_size,
                )
                if mtx is not None:
                    st.session_state.calibration_matrix = mtx
                    st.session_state.dist_coeffs = dist
                    st.session_state["proj_fx"] = float(mtx[0, 0])
                    st.session_state["proj_fy"] = float(mtx[1, 1])
                    st.success(
                        f"Calibration successful! Mean Re-projection Error: {err:.4f} pixels"
                    )

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.write("**Intrinsic Matrix (K):**")
                        st.dataframe(pd.DataFrame(mtx, columns=["X", "Y", "Z"]))
                    with col_m2:
                        st.write("**Distortion Coefficients:**")
                        st.dataframe(pd.DataFrame(dist.reshape(1, -1)))

                    st.write("**Corner Detection Overlays:**")
                    for idx, (fname, pimg) in enumerate(previews):
                        if pimg is not None:
                            st.image(
                                cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB),
                                caption=f"{fname}: Found",
                                width="stretch",
                            )
                        else:
                            st.caption(f"{fname}: Corners Not Found")
                else:
                    st.error("Corner detection failed on all uploaded images.")
