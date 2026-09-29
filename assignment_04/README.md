# Computer Vision: Human Boundary Extraction

**Author:** Aakrit Sharma Lamsal
**Course:** Computer Vision (MS in Computer Science)
**Institution:** Georgia State University

## Overview
This project is a Streamlit-based web application built to extract the exact boundaries of human subjects from both regular (RGB/Grayscale) and Thermal Infrared images. 

Per the assignment constraints, **no Deep Learning or Machine Learning models are used**. The entire pipeline relies strictly on classical Computer Vision algorithms and mathematical morphology using OpenCV. All image processing and byte-decoding are handled entirely in memory to ensure fast, volatile execution suitable for cloud deployment.

## Features & Implementation

The application is divided into three main modules:

### 1. RGB Boundary Extraction (`boundary_rgb.py`)
*   **Pipeline:** Bilateral Filtering $\rightarrow$ LAB CLAHE Illumination Normalization $\rightarrow$ Iterative GrabCut (Graph-Cut Optimization) $\rightarrow$ Morphological Refinement.
*   **Boundary Math:** Computes the exact 8-connected morphological boundary using the formula: `Beta(A) = A - (A erode B)`.
*   **SAM2 Comparison:** Includes a feature to upload an exported mask from Meta's SAM2 to calculate in-memory Intersection over Union (IoU) and Dice Similarity metrics against the classical OpenCV result.

### 2. Thermal Boundary Extraction (`boundary_thermal.py`)
*   **Pipeline:** Radiometric Grayscale Extraction $\rightarrow$ CLAHE Contrast Boosting $\rightarrow$ Edge-Preserving Bilateral Smoothing $\rightarrow$ Adaptive Otsu + Top-Percentile Thresholding.
*   **Design:** Tailored for thermal cameras (LWIR/MWIR), exploiting natural radiometric heat contrast (both white-hot and black-hot polarities) to isolate human silhouettes without complex background interference.

### 3. Fourier Domain Theory & Demo (`theory.py`)
*   **Mathematical Proofs:** Provides rigorous derivations of 2D Continuous/Discrete Fourier Transforms, the Fourier Derivative Theorem, and Laplacian of Gaussian (LoG) spatial-to-frequency mappings.
*   **Interactive 2D FFT Demo:** A live, interactive demo that computes the 2D Fast Fourier Transform of an uploaded image, allowing users to apply High-Pass (Edge Extraction) and Low-Pass (Region Segmentation) Gaussian filters directly in the frequency domain.

## Tech Stack
*   **Language:** Python 3
*   **UI Framework:** Streamlit
*   **Computer Vision:** OpenCV (`cv2`)
*   **Numerical Computation:** NumPy

## How to Run Locally

1. Install the required dependencies:
   ```bash
   pip install streamlit opencv-python-headless numpy