# Assignment 06 — Optical Flow & Structure from Motion

**Course:** Computer Vision
**Student:** Aakrit Sharma Lamsal — 002865270

This Streamlit application demonstrates motion estimation using optical flow and planar multi-view reconstruction using Structure from Motion concepts.

## Question 1 — Optical Flow

The application processes two user-supplied videos and analyzes a 30-second sample from each.

It provides:

* dense Farnebäck optical flow visualization,
* motion magnitude and dominant-direction statistics,
* motion-coverage analysis,
* sparse pyramidal Lucas–Kanade tracking,
* manual Lucas–Kanade tracking derived from the theoretical equations,
* bilinear interpolation for sub-pixel sampling,
* comparison of theoretical and OpenCV pixel locations.

### Recommended Video Inputs

Use two MP4/MOV videos containing at least 30 seconds of continuous visible motion.

For best results, use H.264 MP4 video at approximately 720p and 30 FPS.

A useful experiment is:

**Video 1 — Object Motion**

Keep the camera stationary and move a textured object or person across the scene. This should produce localized optical flow around the moving object.

**Video 2 — Camera Motion**

Slowly pan or translate the camera across a textured scene such as a room, bookshelf, or outdoor environment. This should produce flow across a larger portion of the image.

Avoid excessive motion blur, very dark scenes, featureless surfaces, abrupt camera shake, and extremely fast movement.

### Optical Flow Parameters

**30-second sample start**

Selects which 30-second portion of the uploaded video is processed.

Choose a segment containing continuous and meaningful motion.

**Internal maximum frame dimension**

Recommended:

`480`

The video is internally downscaled while preserving its aspect ratio. This reduces memory and computational cost without modifying the original uploaded file.

Use `640` only when additional spatial detail is required.

**Dense-flow analysis FPS**

Recommended:

`8 FPS`

This controls temporal sampling during dense Farnebäck optical-flow computation.

Use approximately:

* `5–6 FPS` for slow motion and lower computational cost,
* `8 FPS` for normal motion,
* `10 FPS` for faster motion.

**Frame-pair validation point**

Select a time inside the 30-second sample at which the application extracts two consecutive original video frames for Lucas–Kanade validation.

Choose a timestamp containing visible textured features, moderate motion, minimal blur, and no severe occlusion.

The displayed endpoint error compares the manually derived Lucas–Kanade result with OpenCV's pyramidal Lucas–Kanade tracker.

---

## Question 2 — Planar Structure from Motion

The application uses four photographs of the same planar object captured from different camera viewpoints.

It estimates:

* the planar object boundary,
* plane-to-image homographies,
* points reconstructed into a common physical coordinate system,
* camera rotation and translation,
* camera centers,
* reprojection error,
* ORB feature correspondences,
* RANSAC inliers,
* multi-view geometric consistency.

### Recommended Object

Use a flat, rectangular, textured object such as:

* a book cover,
* printed A4 sheet,
* poster,
* magazine cover,
* rectangular sign.

Only one physical plane should be reconstructed.

For a book, use the **front cover only**. The spine, pages, and back cover belong to different 3D planes and should not form part of the detected boundary.

### Capturing the Four Views

Keep the object stationary and move only the camera.

A recommended set of views is:

* View 1 — nearly frontal,
* View 2 — approximately 20–30° from the left,
* View 3 — approximately 20–30° from the right,
* View 4 — approximately 15–25° vertically displaced.

The complete object boundary must remain visible in every image.

Use the same physical camera and lens for all four photographs. A smartphone's `1×` main camera is recommended.

Avoid ultrawide lenses, digital zoom, portrait mode, severe perspective angles, glare, strong shadows, and changing camera lenses between photographs.

A contrasting background greatly improves automatic boundary detection.

### Object Width and Height

Enter the **actual measured physical dimensions of the planar surface**.

For example, if a book cover measures:

```text
Width  = 158 mm
Height = 235 mm
```

enter:

```text
158
235
```

Do not leave the default `210 × 297 mm` unless the physical object is actually A4-sized.

The physical dimensions establish the metric coordinate system used by the reconstruction.

### Internal Image Maximum Dimension

Recommended:

`1000`

Uploaded photographs are internally downscaled while preserving their aspect ratio.

Use:

`1200`

when the object occupies a relatively small part of the image or additional feature detail is required.

### Camera Intrinsics

Two modes are available.

#### Approximate Horizontal Field of View

Use this when calibrated camera parameters are unavailable.

The entered field of view should approximately correspond to the physical lens used to capture the photographs.

For a normal smartphone `1×` main camera, a value in approximately the mid-60s to low-70s degrees may be more realistic than a very narrow value such as 45°, although the correct value depends on the specific camera and crop.

Camera positions obtained using an approximate FOV should also be interpreted as approximate.

#### Calibrated Camera Intrinsics

This mode is preferred for the final experiment.

Enter:

```text
fx
fy
cx
cy
calibration image width
calibration image height
```

corresponding to

$$
K =
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}.
$$

Camera calibration can be performed using multiple photographs of a known chessboard calibration pattern and OpenCV's camera-calibration functions.

Calibration and SfM photographs should use the same physical camera, lens, zoom level, and preferably the same image resolution and aspect ratio.

### Boundary Detection Check

Before interpreting the reconstruction, inspect the detected `TL`, `TR`, `BR`, and `BL` object corners displayed for every viewpoint.

The detected polygon should closely follow the actual planar-object boundary.

If the polygon includes the background, a shadow, book spine, pages, or another object, the resulting reconstruction should not be considered reliable.

Retake the photograph with better contrast or lighting when necessary.

### Interpreting Results

Feature alignment across all four reconstructed views provides evidence that the estimated homographies are geometrically consistent.

The ORB/RANSAC section provides additional independent evidence using multi-view feature correspondences.

Lower reprojection and boundary-transfer errors generally indicate better geometric consistency.

Camera-position estimates depend strongly on the accuracy of the camera intrinsic parameters.

---

## Technologies

* Python
* Streamlit
* OpenCV
* NumPy
* Pandas
* Pillow
* PyAV

## Running the Application

```bash
streamlit run app.py
```

All uploaded media and generated results are processed in memory. Large images and videos are internally downscaled or temporally sampled to improve performance on resource-constrained deployments such as Streamlit Community Cloud.
