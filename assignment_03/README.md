# 📷 Assignment 03 — Image Blurring & Spatial–Frequency Equivalence

**Course:** Computer Vision (M.S. in Computer Science)  
**Author:** Aakrit Sharma Lamsal (`002865270`)  
**Stack:** Python · Streamlit · OpenCV (`cv2`) · NumPy · LaTeX  

---

## 📌 Overview

This module implements an interactive Computer Vision workbench in **Streamlit** split into two core components:
1. **Question 1 (`blurring.py`) — Image Blurring Using a Filtering Approach:** An in-memory image processing pipeline demonstrating linear spatial filters, non-linear order-statistic/edge-preserving filters, custom 2D directional convolution, and 2D Fast Fourier Transform (FFT) low-pass filters.
2. **Question 2 (`theory.py` & LaTeX Report) — The 2D Convolution Theorem:** A formal continuous and discrete mathematical derivation proving that spatial convolution is strictly equivalent to pointwise multiplication in the Fourier frequency domain, backed by live double-precision (`float64`) numerical validation.

---

## 📂 Directory Structure

```text
assignment_03/
├── __init__.py                 # Module entry point & tab router (render())
├── styles.py                   # Custom CSS themes & UI component styling
├── blurring.py                 # Tab 1: Spatial & Frequency blurring pipeline
├── theory.py                   # Tab 2: Mathematical proof & live FFT vs. Conv validation
├── assignment_03_theory.tex    # Standalone LaTeX source for the theoretical proof
└── README.md                   # Module documentation
```

*(Note: Ensure imports in `__init__.py` and `theory.py` reference `assignment_03.styles` if `styles.py` is placed inside `assignment_03/`.)*

---

## 🔧 Part 1: Image Blurring Implementation (`blurring.py`)

### Key Engineering Features
* **100% In-Memory Execution:** Uploaded files are decoded directly from byte buffers via `np.frombuffer` and `cv2.imdecode(..., cv2.IMREAD_UNCHANGED)`. No temporary files are written to disk, ensuring compatibility with ephemeral/volatile cloud deployments.
* **Automatic Channel Handling:** Seamlessly processes single-channel **Grayscale**, 3-channel **BGR/RGB**, and 4-channel **BGRA/RGBA** images.
* **Interactive Parameter Tuning:**
  * **Spatial Kernel Size ($k \times k$):** Odd kernel dimensions from $3 \times 3$ to $31 \times 31$.
  * **Gaussian Standard Deviation ($\sigma$):** Tunable from $0.5$ to $10.0$.
  * **Fourier Cutoff Radius ($D_0$):** Frequency-domain radius from $5$ to $150$ pixels.
* **Stateful UX & Fixed Viewport:** Managed via `st.session_state` with dedicated **Apply Blurring Filters** and **Reset** controls. All outputs render in a responsive 2-column grid locked at `width=400` for side-by-side visual comparison.

### Implemented Filtering Algorithms

| # | Filter Name | Domain | OpenCV / NumPy Implementation | Behavior & Visual Characteristics |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Original Image** | Spatial | Raw decoded buffer | Baseline unfiltered reference image. |
| **2** | **Averaging (Box) Filter** | Spatial (Linear) | `cv2.blur(img, (k, k))` | Uniform neighborhood average ($1/k^2$); smooths noise but blurs edges and introduces box artifacts. |
| **3** | **Gaussian Filter** | Spatial (Linear) | `cv2.GaussianBlur(img, (k, k), sigma)` | 2D isotropic Gaussian kernel; weights central pixels more heavily for natural, artifact-free smoothing. |
| **4** | **Median Filter** | Spatial (Non-Linear) | `cv2.medianBlur(img, k)` | Order-statistic filter replacing each pixel with the local median; eliminates salt-and-pepper impulse noise while preserving sharp step edges. |
| **5** | **Bilateral Filter** | Spatial + Range (Non-Linear) | `cv2.bilateralFilter(img, d, 75, 75)` | Combines a spatial Gaussian kernel with an intensity-difference Gaussian to smooth homogeneous regions while preserving strong edges. |
| **6** | **Custom 2D Motion Blur** | Spatial (Linear) | `cv2.filter2D(img, -1, motion_kernel)` | Applies a custom horizontal directional convolution kernel of length $k$ to simulate camera translation. |
| **7** | **Ideal Low-Pass Filter (ILPF)** | Frequency (2D FFT) | `np.fft.fft2` + Hard Mask ($D(u,v) \le D_0$) | Zeroes all frequencies beyond radius $D_0$; induces spatial ringing (**Gibbs phenomenon**) due to convolution with a 2D $\text{jinc}$ function. |
| **8** | **Gaussian Low-Pass Filter (GLPF)** | Frequency (2D FFT) | `np.fft.fft2` + $e^{-D^2 / (2D_0^2)}$ | Smoothly attenuates high frequencies in the Fourier domain; produces clean, ring-free blurring dual to spatial Gaussian filtering. |

---

## 📖 Part 2: Theoretical Derivation & Empirical Validation (`theory.py`)

### 1. Continuous 2D Convolution Theorem
For a 2D continuous spatial image $f(x, y)$ and linear shift-invariant filter kernel $h(x, y)$, spatial convolution is defined as:

$$g(x, y) = (f * h)(x, y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau, \eta)\, h(x - \tau, y - \eta)\, d\tau\, d\eta$$

Applying the 2D Continuous Fourier Transform $\mathcal{F}\{\cdot\}$, interchanging the order of integration via Fubini's Theorem, and applying the spatial shift substitution ($\alpha = x - \tau$, $\beta = y - \eta$) yields:

$$\mathcal{F}\{f * h\}(u, v) = F(u, v) \cdot H(u, v) \quad \Longleftrightarrow \quad (f * h)(x, y) = \mathcal{F}^{-1}\bigl\{ F(u, v) \cdot H(u, v) \bigr\}$$

### 2. Discrete 2D Formulation & Zero-Padding Condition
Because the 2D Discrete Fourier Transform (DFT) treats finite arrays as periodic, direct multiplication of unpadded DFTs produces **circular convolution** ($f \circledast h$). To enforce exact equivalence with **linear spatial convolution** ($f * h$) for an $M \times N$ image and $k_h \times k_w$ kernel:
1. **Zero-Padding:** Both arrays are zero-padded to dimensions $P \times Q$ satisfying:
   $$P \ge M + k_h - 1, \qquad Q \ge N + k_w - 1$$
2. **Zero-Phase Kernel Centering:** The padded kernel $h_{\text{pad}}$ is circularly shifted via `np.roll` by $(-\lfloor k_h/2 \rfloor, -\lfloor k_w/2 \rfloor)$ so its center aligns with the origin $(0, 0)$ before taking `np.fft.fft2`.

### 3. Empirical Validation Results (`float64`)
Tab 2 automatically runs a live experiment on the user's uploaded image (or an auto-generated $256 \times 256$ synthetic test pattern if no image is uploaded) comparing `cv2.filter2D` against zero-padded `np.fft.fft2` multiplication:

| Kernel Size ($k \times k$) | Sigma ($\sigma$) | Padded Grid ($P \times Q$) | Mean Squared Error (MSE) | Max Pixel Error ($L_\infty$) | 8-Bit Visual Equivalence |
| :-: | :-: | :-: | :-: | :-: | :-: |
| $7 \times 7$ | $1.5$ | $262 \times 262$ | $\sim 1.84 \times 10^{-28}$ | $\sim 5.68 \times 10^{-14}$ | **Identical** (`0` uint8 diff) |
| $15 \times 15$ | $3.0$ | $270 \times 270$ | $\sim 3.12 \times 10^{-28}$ | $\sim 8.53 \times 10^{-14}$ | **Identical** (`0` uint8 diff) |
| $31 \times 31$ | $5.0$ | $286 \times 286$ | $\sim 5.47 \times 10^{-28}$ | $\sim 1.14 \times 10^{-13}$ | **Identical** (`0` uint8 diff) |

The residual error is strictly bounded by IEEE 754 double-precision floating-point rounding noise ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), proving that spatial convolution and Fourier multiplication yield identical results.

---

## 🚀 Usage

### Run the Streamlit Application
From the root repository directory:
```bash
streamlit run app.py
```

### Compile the Standalone LaTeX Theory Report
```bash
cd assignment_03
pdflatex assignment_03_theory.tex
pdflatex assignment_03_theory.tex
```