# Assignment 07 — Module 07: Stereo Vision

**Student:** Aakrit Sharma Lamsal (`002865270`)  
**Course:** Computer Vision (CSC 8830)  
**Instructor:** Dr. Ashok Aswin  
**Module:** Module 07 — Stereo Vision (Calibrated Stereo, Uncalibrated Stereo, and Video-Based 3D Room Estimation)

---

## Table of Contents
1. [Overview & System Architecture](#1-overview--system-architecture)
2. [Question 1: Calibrated Stereo Object-Size Estimation](#2-question-1-calibrated-stereo-object-size-estimation)
3. [Question 2: Uncalibrated Stereo Object-Size Derivation (By Hand)](#3-question-2-uncalibrated-stereo-object-size-derivation-by-hand)
4. [Question 3: Room Size Estimation from Video](#4-question-3-room-size-estimation-from-video)
5. [Tab 3 & Tab 4: Output Demonstrations and Submission PDF](#5-tab-3--tab-4-output-demonstrations-and-submission-pdf)
6. [How to Reproduce Locally](#6-how-to-reproduce-locally)
7. [References](#7-references)

---

## 1. Overview & System Architecture

This Streamlit module implements classical, first-principles **Stereo Vision** pipelines (pure pinhole geometry and epipolar algebra via OpenCV and NumPy—no deep learning or AI depth estimators) across four tabs in `ui.py`:

| Tab | Module | Purpose |
| :--- | :--- | :--- |
| **Tab 1 — Question 1** | `object_size.py` | Calibrated parallel-axis stereo depth $Z$ and square object width estimation from 2 images. |
| **Tab 2 — Question 3** | `room_size.py` | 3D room dimension estimation (Width, Length, Height, Area, Volume) from a single side-step video. |
| **Tab 3 — Output Demonstrations** | `output_demonstration.py` | Curated input/output visual snapshots and derived validation metrics table. |
| **Tab 4 — Submission Document** | `submission_document.py` | Renders and provides download access for `assets/assignment_07.pdf` (containing Question 2 by-hand proof). |

### Core Design & UX Principles
- **Zero Persistent Storage (Volatile In-Memory Processing):** Designed for ephemeral cloud deployment (Streamlit Community Cloud). Uploaded images are decoded directly in RAM via `io.BytesIO` and `PIL.ImageOps.exif_transpose` (preserving smartphone EXIF orientation). Uploaded videos are accessed via `tempfile.NamedTemporaryFile(delete=False)` only long enough for `cv2.VideoCapture` to extract the two keyframes, then immediately deleted in a `finally:` block.
- **Display vs. Native Coordinate Mapping:** All images and video frames are displayed at a fixed `600 px` width (`DISPLAY_WIDTH = 600`) for clean side-by-side UI alignment. Click coordinates $(x_{\text{disp}}, y_{\text{disp}})$ captured by `streamlit-image-coordinates` are mapped back to original pixel coordinates via:
  $$x_{\text{orig}} = x_{\text{disp}} \cdot \frac{w_{\text{orig}}}{600}, \qquad y_{\text{orig}} = y_{\text{disp}} \cdot \frac{h_{\text{orig}}}{600}$$
- **Session State & Reset Controls:** Clicked points, last-click deduplication guards, and dynamic `file_uploader` keys are stored in `st.session_state`, allowing instant **Clear Points / Clear Corners** and **Reset All** actions.

---

## 2. Question 1: Calibrated Stereo Object-Size Estimation

**Implementation File:** `assignments/assignment_07/object_size.py` (`render_object_size_estimate`)

### 2.1 Inputs
1. **Left & Right Stereo Images (`pic_1.jpg`, `pic_2.jpg`):**
   - **Device:** Google Pixel 10 Pro XL (50 MP 1x main wide camera, $82^\circ$ diagonal FOV, $4:3$ sensor).
   - **Resolution:** Captured at `4080 × 3072` (landscape) and downscaled to `1280 × 964` (preserving the $4:3$ aspect ratio so FOV is unchanged).
   - **Supported formats:** `.png`, `.jpg`, `.jpeg`.
2. **Camera Baseline Distance ($B$):**
   - Controlled via drag slider (`5 cm` to `100 cm`, default `15 cm`).
   - Represents the exact pure horizontal translation $T_x = B$ between the Left and Right camera positions.
3. **Camera Long-Edge Horizontal Field of View ($\text{FOV}$):**
   - Controlled via slider (`45.0°` to `100.0°`, default `73.0°`; for Pixel 10 Pro XL $82^\circ$ diagonal on a $4:3$ sensor, exact horizontal FOV is $2\arctan(\frac{4}{5}\tan(41^\circ)) \approx 69.6^\circ\text{–}70.0^\circ$).
4. **Interactive Point Clicks (4 clicks total):**
   - **Left Image:** `P1` (left edge of object) and `P2` (right edge of object).
   - **Right Image:** `P1` (corresponding left edge) and `P2` (corresponding right edge).

### 2.2 Capture & Reproduction Procedure
1. Place a square/rectangular target object (e.g., a notebook of known width $26.7\text{ cm}$) on a flat surface facing the camera at roughly $0.8\text{ m}–1.5\text{ m}$ distance.
2. Hold the phone in **landscape orientation** (1x lens) with its back flat/parallel to a straight guide (e.g., a table edge or ruler) so the optical axis is perpendicular to the object plane.
3. Capture the **Left** image.
4. Slide the phone **purely sideways** to the right by a measured baseline $B$ (e.g., `15 cm` or `20 cm`) **without rotating or tilting the wrist** (parallel optical axes).
5. Capture the **Right** image. *(Note: In parallel stereo, the object must shift noticeably toward the left of the Right image; rotating the phone inward to re-center the object cancels disparity and inflates depth $Z$.)*
6. Upload both images in Tab 1, set the **Baseline** and **FOV** sliders, and click `P1` then `P2` on both images.

### 2.3 Mathematical Formulation & Calculation Steps

1. **Camera Intrinsic Parameters ($K$):**  
   From image dimensions $(w, h)$ and long-edge $\text{FOV}$, the focal length in pixels is:
   $$f_x = f_y = \frac{\max(w, h) / 2}{\tan(\text{FOV} / 2)}, \qquad c_x = \frac{w}{2}, \qquad c_y = \frac{h}{2}$$
   Using $\max(w, h)$ ensures $\text{FOV}$ maps consistently whether images are landscape or portrait.

2. **Horizontal Stereo Disparity ($d$):**  
   For each corresponding point pair $i \in \{1, 2\}$:
   $$d_i = |x_{L,i} - x_{R,i}|$$
   *(Guarded against $d_i < 1.0\text{ px}$ to avoid division-by-zero instability.)*

3. **Depth Recovery ($Z$) via Parallel-Axis Triangulation:**  
   By similar triangles between the baseline $B$ and image disparity $d_i$:
   $$Z_i = \frac{f_x \cdot B}{d_i}$$
   where $B$ is converted from centimeters to meters ($B_{\text{m}} = B_{\text{cm}} / 100$).

4. **3D Back-Projection $(X_i, Y_i, Z_i)$:**  
   Each clicked point is projected from 2D pixel space into 3D camera coordinates:
   $$X_i = \frac{(x_{L,i} - c_x) \cdot Z_i}{f_x}, \qquad Y_i = \frac{(y_{L,i} - c_y) \cdot Z_i}{f_y}$$

5. **Object Width ($W$):**  
   Using the mean depth $\bar{Z} = \frac{Z_1 + Z_2}{2}$ (which prevents minor 1–2 px click jitter in disparity from skewing $Z_2 - Z_1$) and the Euclidean pixel span $\Delta u$ on the Left image:
   $$\Delta u = \sqrt{(x_{L,2} - x_{L,1})^2 + (y_{L,2} - y_{L,1})^2}, \qquad W = \frac{\Delta u \cdot \bar{Z}}{f_x}$$

### 2.4 Outputs
- **Annotated Left & Right Previews (`600 px` width):** Numbered circular markers (`P1`, `P2`) connected by a line segment.
- **Metric Cards:** Estimated Depth $Z$ (cm and m), Estimated Object Width (cm and mm), and Baseline Used $B$ (cm).
- **3D Point Coordinates Table:** `pandas.DataFrame` showing $(X, Y, Z)$ in cm for both `P1` and `P2`.

---

## 3. Question 2: Uncalibrated Stereo Object-Size Derivation (By Hand)

**Rendered File:** `assignments/assignment_07/assets/assignment_07.pdf` (displayed in **Tab 4: Submission Document**)

### 3.1 Problem Statement
Derive the end-to-end procedure to estimate the depth $Z$ and physical dimensions (rectangle, circle, polygon) of an object from two images when the **extrinsic camera pose $(R, \mathbf{t})$ is unknown** (**Uncalibrated Stereo**).

### 3.2 Summary of Derivation Steps (Detailed in `assignment_07.pdf`)
1. **Epipolar Constraint & Fundamental Matrix ($F$):**  
   For $N \ge 8$ corresponding homogeneous image points $\tilde{\mathbf{u}}_L = [u_L, v_L, 1]^T$ and $\tilde{\mathbf{u}}_R = [u_R, v_R, 1]^T$:
   $$\tilde{\mathbf{u}}_R^T \, F \, \tilde{\mathbf{u}}_L = 0 \implies A \mathbf{f} = \mathbf{0}$$
   Solve for $\mathbf{f}$ as the last right singular vector of $A$ via SVD, reshape to $3 \times 3$, and enforce $\operatorname{rank}(F) = 2$ by zeroing its smallest singular value.
2. **Essential Matrix ($E$) & Relative Pose $(R, \hat{\mathbf{t}})$:**  
   Using intrinsic matrices $K_L, K_R$:
   $$E = K_R^T \, F \, K_L = [\hat{\mathbf{t}}]_\times R$$
   Decompose $E = U \Sigma V^T$ via SVD to recover rotation $R \in SO(3)$ and unit translation $\hat{\mathbf{t}}$ ($\|\hat{\mathbf{t}}\| = 1$), selecting the physically valid solution via the cheirality condition ($Z > 0$).
3. **Linear 3D Triangulation (Up to Scale $\lambda$):**  
   Construct projection matrices $P_L = K_L [I \mid \mathbf{0}]$ and $P_R = K_R [R \mid \hat{\mathbf{t}}]$. For each object vertex correspondence $(\tilde{\mathbf{u}}_L, \tilde{\mathbf{u}}_R)$, stack $\tilde{\mathbf{u}}_L \times (P_L \tilde{\mathbf{X}}) = \mathbf{0}$ and $\tilde{\mathbf{u}}_R \times (P_R \tilde{\mathbf{X}}) = \mathbf{0}$ into $M \tilde{\mathbf{X}} = \mathbf{0}$ and solve via SVD to obtain unscaled 3D coordinates $\mathbf{X}'_i$.
4. **Metric Scale Recovery ($\lambda$) & Object Dimensions:**  
   Because $\hat{\mathbf{t}}$ is unit-norm, true 3D coordinates are $\mathbf{X}_i = \lambda \mathbf{X}'_i$. Recover $\lambda$ from a known baseline ($\lambda = B$) or a known scene reference length $L_{\text{known}}$ ($\lambda = L_{\text{known}} / \|\mathbf{X}'_b - \mathbf{X}'_a\|_2$), then compute object dimensions as Euclidean distances $d_{ij} = \lambda \|\mathbf{X}'_j - \mathbf{X}'_i\|_2$.

---

## 4. Question 3: Room Size Estimation from Video

**Implementation File:** `assignments/assignment_07/room_size.py` (`render_room_size_estimate`)

### 4.1 Inputs
1. **Single Room Walkthrough Video (`.mp4`, `.mov`, `.avi`):**
   - **Capture Resolution:** Recorded at `1920 × 1080` (landscape, 1x main lens, **Video Stabilization OFF**).
   - **HandBrake Preprocessing:** Encoded using HandBrake (`Fast 720p30`, `1280 × 720`, Constant Framerate `30 fps`, H.264, audio removed) to reduce file size for fast in-memory processing on Streamlit Cloud while preserving the 16:9 aspect ratio.
2. **Side-Step Distance / Baseline ($B$):**
   - Controlled via drag slider (`5 cm` to `150 cm`, default `50 cm`; tested at `85 cm`).
   - Represents the pure lateral translation of the camera along the back wall (**Wall D**) between Keyframe 1 and Keyframe 2.
3. **Effective Video Horizontal FOV ($\text{FOV}_{\text{video}}$):**
   - Controlled via slider (`45.0°` to `100.0°`, default `73.0°`; calibrated to **`64.0°`** for Pixel 10 Pro XL video due to its ~8–10% video sensor readout/OIS margin crop even when EIS is toggled off).
4. **Keyframe Indices (`Frame 1` & `Frame 2`):**
   - Two sliders spanning `[0, total_frames - 1]` to select a settled frame **before** the side-step and a settled frame **after** the side-step.
5. **Interactive Front-Wall Corner Clicks (3 clicks per frame):**
   - **`P1` (Bottom-Left):** Junction of Left Wall (A), Front Wall (B), and Floor.
   - **`P2` (Bottom-Right):** Junction of Front Wall (B), Right Wall (C), and Floor.
   - **`P3` (Top-Left):** Junction of Left Wall (A), Front Wall (B), and Ceiling.

### 4.2 Capture & Reproduction Procedure
1. Stand with your back against the back wall (**Wall D**) facing the opposite front wall (**Wall B**) so all 3 corners (`P1`, `P2`, `P3`) of **Wall B** are clearly visible in the 16:9 landscape frame.
2. Mark two floor positions separated laterally by $B$ (e.g., `85 cm`) parallel to **Wall B**.
3. **Seconds 0–2 (Position A):** Hold the phone steady against a flat reference/guide (avoiding rotation) and begin recording.
4. **Seconds 2–4 (Lateral Side-Step):** Slide/step smoothly sideways by $B = 85\text{ cm}$ along **Wall D** while keeping the phone pointed straight ahead (**zero yaw/pitch rotation**).
5. **Seconds 4–6 (Position B):** Hold steady for 2 seconds, then stop recording.
6. Compress to `720p` (`1280 × 720`) in HandBrake and upload to Tab 2.
7. Set `Frame 1` inside the steady Position A window, `Frame 2` inside the steady Position B window, set **Side-Step Distance** to `85 cm`, and **FOV** to `64.0°`.
8. Click `P1`, `P2`, and `P3` in exact order on Frame 1 and Frame 2.

### 4.3 Mathematical Formulation & Calculation Steps

Because Frame 1 and Frame 2 are separated by a pure lateral translation $B$ with parallel optical axes, the two keyframes form a **calibrated motion-stereo pair**:

1. **Focal Length from Video FOV:**
   $$f_x = f_y = \frac{\max(w, h) / 2}{\tan(\text{FOV}_{\text{video}} / 2)}, \qquad c_x = \frac{w}{2}, \qquad c_y = \frac{h}{2}$$

2. **Front-Wall Depth / Room Length ($L = \bar{Z}$):**  
   For each of the 3 clicked corners $i \in \{P_1, P_2, P_3\}$ with horizontal disparity $d_i = |x_{1,i} - x_{2,i}|$:
   $$Z_i = \frac{f_x \cdot B}{d_i}, \qquad L = \bar{Z} = \frac{Z_1 + Z_2 + Z_3}{3}$$
   Here, $L$ represents the perpendicular distance from the camera plane (near Wall D) to the front wall (Wall B).

3. **Room Width ($W$) and Room Height ($H$):**  
   Using the front-wall pixel spans in Frame 1:
   $$\Delta u_W = \sqrt{(x_{1,P2} - x_{1,P1})^2 + (y_{1,P2} - y_{1,P1})^2}, \qquad W = \frac{\Delta u_W \cdot L}{f_x}$$
   $$\Delta v_H = \sqrt{(x_{1,P3} - x_{1,P1})^2 + (y_{1,P3} - y_{1,P1})^2}, \qquad H = \frac{\Delta v_H \cdot L}{f_y}$$

4. **Floor Area ($A$) and Room Volume ($V$):**
   $$A = W \times L \quad (\text{m}^2), \qquad V = W \times L \times H \quad (\text{m}^3)$$
   *(Metric outputs are also converted to feet using $1\text{ m} = 3.28084\text{ ft}$, $1\text{ m}^2 = 10.7639\text{ ft}^2$, and $1\text{ m}^3 = 35.3147\text{ ft}^3$.)*

### 4.4 Error & Sensitivity Analysis (Observed During Testing)
Substituting $L = \frac{f_x B}{d}$ into the Width and Height equations shows that $f_x$ cancels out of $W$ and $H$:
$$W = \frac{\Delta u_W \cdot B}{d}, \qquad H = \frac{\Delta v_H \cdot B}{d}$$
Consequently:
- **Width and Height depend exclusively on $\frac{B}{d}$**, while **Length ($L$) depends on both $\text{FOV}$ ($f_x$) and $\frac{B}{d}$**.
- Even a small **$2^\circ–3^\circ$ inward wrist rotation** during a freehand side-step reduces disparity $d$ by ~25%, inflating $\frac{B}{d}$ (and therefore **all three dimensions** $W, L, H$) by an identical scale factor ($\approx 1.35\times$). Keeping the camera back parallel to a flat guide eliminates this rotation bias.

### 4.5 Outputs
- **Annotated Keyframe Previews (`600 px` width):** `P1`, `P2`, `P3` markers with green Width line (`P1 ↔ P2`) and pink Height line (`P1 ↔ P3`).
- **Metric Cards:** Room Width, Room Length, and Room Height in both meters and feet.
- **Summary Table (`pandas.DataFrame`):** Consolidated table of Width, Length, Height, Floor Area ($\text{m}^2$ / $\text{sq ft}$), and Room Volume ($\text{m}^3$ / $\text{cu ft}$).

---

## 5. Tab 3 & Tab 4: Output Demonstrations and Submission PDF

- **Tab 3 (`output_demonstration.py`):** Displays side-by-side `600 px` snapshots from `assets/` (`q1_inputs.png`, `q1_output_metrics.png`, `q3_input_frames.png`, `q3_room_annotated.png`, `q3_metrics_table.png`) alongside a derived validation summary table.
- **Tab 4 (`submission_document.py`):** Loads `assets/assignment_07.pdf`, provides a one-click `st.download_button`, and renders the PDF inline (using `st.pdf` if `streamlit-pdf` is installed, with an automatic `<embed>` base64 fallback).

---

## 6. How to Reproduce Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt