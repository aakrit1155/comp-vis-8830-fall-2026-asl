import streamlit as st

from .styles import inject_theory_styles


def render_theory() -> None:
    inject_theory_styles()

    st.markdown(
        """
        <div class="cv-theory-hero">
            <h2>
                Mathematical Foundations — Motion and Multi-View Geometry
            </h2>
            <p>
                Derivation of the optical-flow constraint,
                Lucas-Kanade tracking, bilinear interpolation,
                planar projective geometry, homography-based
                reconstruction, and camera-pose estimation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # OPTICAL FLOW
    # ========================================================

    st.markdown(r"""
## 1. Optical Flow

### 1.1 Brightness constancy assumption

Suppose a scene point appears at pixel location

$$
\mathbf{x}
=
\begin{bmatrix}
x\\
y
\end{bmatrix}
$$

at time $t$ and moves by

$$
\mathbf{d}
=
\begin{bmatrix}
u\\
v
\end{bmatrix}
$$

between two nearby frames.

The fundamental optical-flow assumption is that the brightness
of the same physical point remains approximately constant:

$$
I(x,y,t)
=
I(x+u,\;y+v,\;t+\Delta t).
$$

For a sufficiently small displacement, expand the right-hand
side using a first-order Taylor approximation:

$$
I(x+u,y+v,t+\Delta t)
\approx
I(x,y,t)
+
I_xu
+
I_yv
+
I_t\Delta t.
$$

Using brightness constancy and taking $\Delta t=1$ frame,

$$
I_xu + I_yv + I_t = 0.
$$

This is the **optical-flow constraint equation**.

A single pixel provides one equation but two unknowns,
$u$ and $v$. Therefore the flow cannot be uniquely determined
from one pixel alone. This is the aperture problem.

---

### 1.2 Lucas-Kanade local motion model

Lucas-Kanade assumes that all pixels inside a small neighborhood
$\Omega$ undergo approximately the same displacement:

$$
\mathbf{d}
=
\begin{bmatrix}
u\\
v
\end{bmatrix}.
$$

For $n$ pixels in the neighborhood,

$$
I_{x_i}u + I_{y_i}v = -I_{t_i},
\qquad
i=1,\ldots,n.
$$

Stacking the equations gives

$$
A\mathbf{d}=\mathbf{b}
$$

where

$$
A=
\begin{bmatrix}
I_{x_1} & I_{y_1}\\
I_{x_2} & I_{y_2}\\
\vdots & \vdots\\
I_{x_n} & I_{y_n}
\end{bmatrix},
\qquad
\mathbf{b}
=
-
\begin{bmatrix}
I_{t_1}\\
I_{t_2}\\
\vdots\\
I_{t_n}
\end{bmatrix}.
$$

Because there are usually more equations than unknowns,
the displacement is obtained using least squares:

$$
\hat{\mathbf{d}}
=
\arg\min_{\mathbf{d}}
\|A\mathbf{d}-\mathbf{b}\|_2^2.
$$

Setting the derivative to zero produces the normal equations:

$$
A^TA\mathbf{d}
=
A^T\mathbf{b}.
$$

Therefore,

$$
\boxed{
\mathbf{d}
=
(A^TA)^{-1}A^T\mathbf{b}
}
$$

provided that $A^TA$ is well conditioned.

Expanding $A^TA$,

$$
A^TA
=
\begin{bmatrix}
\sum I_x^2 & \sum I_xI_y\\
\sum I_xI_y & \sum I_y^2
\end{bmatrix}.
$$

Strong corners contain significant intensity variation in both
directions, causing both eigenvalues of $A^TA$ to be large.
This explains why Shi-Tomasi corners are effective points for
Lucas-Kanade tracking.

---

### 1.3 Tracking between two discrete image frames

Let the template patch in frame 1 be $I_1$ and the next frame
be $I_2$.

The translation-only tracking problem is

$$
\hat{\mathbf{d}}
=
\arg\min_{\mathbf{d}}
\sum_{\mathbf{x}\in\Omega}
\left[
I_2(\mathbf{x}+\mathbf{d})
-
I_1(\mathbf{x})
\right]^2.
$$

Define the residual

$$
r_i(\mathbf{d})
=
I_2(\mathbf{x}_i+\mathbf{d})
-
I_1(\mathbf{x}_i).
$$

Linearizing the residual,

$$
r_i(\mathbf{d}+\Delta\mathbf{d})
\approx
r_i(\mathbf{d})
+
\nabla I_2^T
\Delta\mathbf{d}.
$$

The Gauss-Newton update is

$$
\boxed{
\Delta\mathbf{d}
=
-
(J^TJ)^{-1}J^T\mathbf{r}
}
$$

where

$$
J_i=
\begin{bmatrix}
I_x(\mathbf{x}_i+\mathbf{d}) &
I_y(\mathbf{x}_i+\mathbf{d})
\end{bmatrix}.
$$

The estimate is iteratively updated:

$$
\mathbf{d}
\leftarrow
\mathbf{d}
+
\Delta\mathbf{d}.
$$

The implementation in the Optical Flow tab performs this
iteration manually and compares its predicted pixel location

$$
\mathbf{x}_2^{(\mathrm{theory})}
=
\mathbf{x}_1+\hat{\mathbf{d}}
$$

with the location obtained from OpenCV's pyramidal
Lucas-Kanade implementation.

The validation error reported by the application is

$$
e_{\mathrm{endpoint}}
=
\left\|
\mathbf{x}_2^{(\mathrm{theory})}
-
\mathbf{x}_2^{(\mathrm{OpenCV})}
\right\|_2.
$$

A small endpoint difference provides experimental support for
the theoretical derivation.

---

### 1.4 Why an image pyramid is useful

The first-order approximation assumes small displacement.
Large motion violates this assumption.

A Gaussian pyramid reduces the image resolution:

$$
I^{(0)},I^{(1)},\ldots,I^{(L)}.
$$

If an object moves $16$ pixels at the original resolution,
the same motion may correspond to only $2$ pixels at a level
that is downsampled by a factor of $8$.

Lucas-Kanade can therefore estimate the displacement first at
a coarse level and progressively refine it at finer levels.
This is the principle behind pyramidal Lucas-Kanade tracking.

---

## 2. Bilinear Interpolation

Tracking typically produces sub-pixel locations such as

$$
(x,y)=(124.37,\;82.64).
$$

Because a digital image is only defined at integer pixel
coordinates, the intensity at a fractional position must be
estimated.

Let

$$
x=x_0+\alpha,
\qquad
y=y_0+\beta,
$$

where

$$
0\leq\alpha<1,
\qquad
0\leq\beta<1.
$$

The four neighboring pixels are

$$
I_{00}=I(x_0,y_0),
$$

$$
I_{10}=I(x_0+1,y_0),
$$

$$
I_{01}=I(x_0,y_0+1),
$$

and

$$
I_{11}=I(x_0+1,y_0+1).
$$

Bilinear interpolation gives

$$
\boxed{
I(x,y)
=
(1-\alpha)(1-\beta)I_{00}
+
\alpha(1-\beta)I_{10}
+
(1-\alpha)\beta I_{01}
+
\alpha\beta I_{11}
}
$$

The four interpolation weights sum to one:

$$
(1-\alpha)(1-\beta)
+
\alpha(1-\beta)
+
(1-\alpha)\beta
+
\alpha\beta
=1.
$$

This interpolation is explicitly implemented in
`_bilinear_sample()` and is used during the manual
Lucas-Kanade iterations.

---

## 3. Dense Optical Flow and Motion Interpretation

For every image location $(x,y)$, dense optical flow produces

$$
\mathbf{F}(x,y)
=
\begin{bmatrix}
u(x,y)\\
v(x,y)
\end{bmatrix}.
$$

Its magnitude is

$$
M(x,y)
=
\sqrt{
u(x,y)^2+v(x,y)^2
}
$$

and its direction is

$$
\theta(x,y)
=
\operatorname{atan2}
\left(
v(x,y),u(x,y)
\right).
$$

The visualization in this application maps direction to hue and
flow magnitude to brightness.

The application additionally reports:

$$
\text{mean magnitude},
$$

$$
P_{90}(M),
$$

$$
\text{motion ratio}
=
\frac{
\#\{M(x,y)\geq1\}
}{
\#\{\text{pixels}\}
},
$$

and a magnitude-weighted dominant direction.

Localized flow generally indicates independent object motion,
whereas coherent flow over a large part of the frame provides
evidence of camera motion or global scene motion. These are
interpretations of the measured flow field rather than absolute
semantic classifications.
""")

    st.divider()

    # ========================================================
    # STRUCTURE FROM MOTION
    # ========================================================

    st.markdown(r"""
# 4. Structure from Motion for a Planar Object

## 4.1 Pinhole camera model

A 3-D world point is written in homogeneous coordinates as

$$
\tilde{\mathbf{X}}
=
\begin{bmatrix}
X\\
Y\\
Z\\
1
\end{bmatrix}.
$$

Its image projection is

$$
s\tilde{\mathbf{x}}
=
K
\begin{bmatrix}
R & \mathbf{t}
\end{bmatrix}
\tilde{\mathbf{X}}
$$

where

$$
K
=
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
$$

is the camera intrinsic matrix,

$R$ is the camera rotation matrix, and $\mathbf{t}$ is the
translation vector.

---

## 4.2 Reduction to a planar homography

For the selected object, define its physical plane as

$$
Z=0.
$$

Then

$$
\tilde{\mathbf{X}}
=
\begin{bmatrix}
X\\
Y\\
0\\
1
\end{bmatrix}.
$$

Substituting into the camera equation gives

$$
s\tilde{\mathbf{x}}
=
K
\begin{bmatrix}
\mathbf{r}_1 &
\mathbf{r}_2 &
\mathbf{t}
\end{bmatrix}
\begin{bmatrix}
X\\
Y\\
1
\end{bmatrix}.
$$

Therefore

$$
\boxed{
s\tilde{\mathbf{x}}
=
H
\tilde{\mathbf{X}}_{\mathrm{plane}}
}
$$

with

$$
\boxed{
H
=
K
\begin{bmatrix}
\mathbf{r}_1 &
\mathbf{r}_2 &
\mathbf{t}
\end{bmatrix}.
}
$$

Thus each camera viewpoint is related to the physical plane by
a $3\times3$ projective transformation.

---

## 4.3 Manual Multi-View Point Correspondences

Instead of automatically detecting the object boundary, the same
physical points are manually selected in all four images.

The first four points correspond to the corners of the planar
object and are selected in the fixed order

$$
\text{TL}
\rightarrow
\text{TR}
\rightarrow
\text{BR}
\rightarrow
\text{BL}.
$$

If the physical object has width $W$ and height $H$, these points
are assigned metric plane coordinates

$$
\mathbf{X}_1=(0,0),
$$

$$
\mathbf{X}_2=(W,0),
$$

$$
\mathbf{X}_3=(W,H),
$$

$$
\mathbf{X}_4=(0,H).
$$

Their manually selected image coordinates in the reference view
are

$$
\mathbf{x}^{(1)}_1,
\mathbf{x}^{(1)}_2,
\mathbf{x}^{(1)}_3,
\mathbf{x}^{(1)}_4.
$$

These four correspondences determine an initial metric
plane-to-image homography

$$
\mathbf{x}^{(1)}
\sim
H_1
\mathbf{X}.
$$

Additional interior feature points are also manually selected.
Examples include corners of printed letters, logo intersections,
or other sharply identifiable locations.

For an additional point selected in the reference image,

$$
\mathbf{x}^{(1)}_j,
$$

its physical plane coordinate is obtained using the inverse
reference homography:

$$
\boxed{
\mathbf{X}_j
\sim
H_1^{-1}
\mathbf{x}^{(1)}_j
}
$$

where homogeneous normalization is applied after the
transformation.

The same physical point is then manually selected in the
remaining three viewpoints.

Therefore, for viewpoint $i$ and physical point $j$,

$$
\boxed{
\mathbf{x}^{(i)}_j
\sim
H_i
\mathbf{X}_j
}
$$

where $H_i$ is the plane-to-image homography for viewpoint $i$.

For a general homography

$$
H=
\begin{bmatrix}
h_{11}&h_{12}&h_{13}\\
h_{21}&h_{22}&h_{23}\\
h_{31}&h_{32}&h_{33}
\end{bmatrix},
$$

a physical point

$$
\mathbf{X}_j=
\begin{bmatrix}
X_j\\
Y_j\\
1
\end{bmatrix}
$$

is mapped to image coordinates according to

$$
x_j
=
\frac{
h_{11}X_j+h_{12}Y_j+h_{13}
}{
h_{31}X_j+h_{32}Y_j+h_{33}
}
$$

and

$$
y_j
=
\frac{
h_{21}X_j+h_{22}Y_j+h_{23}
}{
h_{31}X_j+h_{32}Y_j+h_{33}
}.
$$

A homography has eight independent degrees of freedom, so at
least four non-collinear correspondences are required.

When more than four manually selected points are available, the
system becomes overdetermined. In the implementation, the
homography is estimated robustly using RANSAC.

For each candidate homography, a correspondence is considered an
inlier when its reprojection error is sufficiently small:

$$
e_j
=
\left\|
\mathbf{x}_j
-
\pi(H\mathbf{X}_j)
\right\|_2,
$$

where $\pi(\cdot)$ converts a homogeneous coordinate to ordinary
Cartesian image coordinates.

Using multiple manually verified correspondences reduces the
dependence on automatic boundary detection and provides a more
controlled multi-view geometric experiment.

---

## 4.4 Recovering camera rotation and translation

From

$$
H
=
K
\begin{bmatrix}
\mathbf{r}_1 &
\mathbf{r}_2 &
\mathbf{t}
\end{bmatrix},
$$

multiply by $K^{-1}$:

$$
K^{-1}H
=
\begin{bmatrix}
\mathbf{b}_1 &
\mathbf{b}_2 &
\mathbf{b}_3
\end{bmatrix}.
$$

Ideally,

$$
\mathbf{b}_1=\lambda^{-1}\mathbf{r}_1,
$$

$$
\mathbf{b}_2=\lambda^{-1}\mathbf{r}_2,
$$

and

$$
\mathbf{b}_3=\lambda^{-1}\mathbf{t}.
$$

A robust estimate of the scale is

$$
\boxed{
\lambda
=
\frac{2}{
\|\mathbf{b}_1\|
+
\|\mathbf{b}_2\|
}
}.
$$

Therefore

$$
\mathbf{r}_1
=
\lambda\mathbf{b}_1,
$$

$$
\mathbf{r}_2
=
\lambda\mathbf{b}_2,
$$

$$
\mathbf{r}_3
=
\mathbf{r}_1
\times
\mathbf{r}_2,
$$

and

$$
\mathbf{t}
=
\lambda\mathbf{b}_3.
$$

Because measurement noise means the recovered rotation is not
exactly orthonormal, the implementation projects the matrix to
the closest valid rotation using singular-value decomposition:

$$
R_{\mathrm{approx}}
=
U\Sigma V^T,
$$

$$
\boxed{
R=UV^T.
}
$$

---

## 4.5 Camera position

The extrinsic transformation maps world coordinates into the
camera coordinate system:

$$
\mathbf{X}_c
=
R\mathbf{X}_w+\mathbf{t}.
$$

The camera center $\mathbf{C}$ is the world point whose camera
coordinates are zero:

$$
0
=
R\mathbf{C}
+
\mathbf{t}.
$$

Hence

$$
R\mathbf{C}
=
-\mathbf{t},
$$

and therefore

$$
\boxed{
\mathbf{C}
=
-R^T\mathbf{t}.
}
$$

The application reports this vector for all four viewpoints.

If the object dimensions are specified in millimetres and the
camera intrinsic matrix is calibrated correctly, the resulting
translation and camera-center values are also expressed in
millimetres.

---

## 4.6 Reconstruction of the Manually Selected Points

After estimating the homography for each camera viewpoint, every
manually observed point can be mapped back into the common
physical object coordinate system.

For viewpoint $i$,

$$
s\mathbf{x}^{(i)}_j
=
H_i
\mathbf{X}_j.
$$

Therefore,

$$
\boxed{
\hat{\mathbf{X}}^{(i)}_j
\sim
H_i^{-1}
\mathbf{x}^{(i)}_j
}
$$

where $\hat{\mathbf{X}}^{(i)}_j$ is the reconstructed physical
location of point $j$ obtained from viewpoint $i$.

Because the user selects the same physical point in all four
images, the four reconstructed estimates

$$
\hat{\mathbf{X}}^{(1)}_j,
\hat{\mathbf{X}}^{(2)}_j,
\hat{\mathbf{X}}^{(3)}_j,
\hat{\mathbf{X}}^{(4)}_j
$$

should ideally coincide.

Their mean reconstructed location is

$$
\boxed{
\bar{\mathbf{X}}_j
=
\frac{1}{4}
\sum_{i=1}^{4}
\hat{\mathbf{X}}^{(i)}_j
}
$$

and the multi-view point consistency error can be measured using

$$
\boxed{
E_{\mathrm{consistency}}
=
\sqrt{
\frac{1}{4N}
\sum_{i=1}^{4}
\sum_{j=1}^{N}
\left\|
\hat{\mathbf{X}}^{(i)}_j
-
\bar{\mathbf{X}}_j
\right\|_2^2
}
}
$$

where $N$ is the number of manually selected correspondences.

A small consistency error indicates that observations from the
four different viewpoints map to nearly the same physical
locations on the object plane.

---

## 4.7 Boundary Reconstruction from the Four Selected Corners

The first four manually selected correspondences represent the
physical boundary corners

$$
\text{TL},\quad
\text{TR},\quad
\text{BR},\quad
\text{BL}.
$$

Each corner is independently reconstructed from all four
viewpoints and the resulting positions are averaged.

Let

$$
\bar{\mathbf{X}}_{\mathrm{TL}},
\quad
\bar{\mathbf{X}}_{\mathrm{TR}},
\quad
\bar{\mathbf{X}}_{\mathrm{BR}},
\quad
\bar{\mathbf{X}}_{\mathrm{BL}}
$$

denote the mean reconstructed corner locations.

Two independent estimates of the object width are

$$
W_{\mathrm{top}}
=
\left\|
\bar{\mathbf{X}}_{\mathrm{TR}}
-
\bar{\mathbf{X}}_{\mathrm{TL}}
\right\|_2
$$

and

$$
W_{\mathrm{bottom}}
=
\left\|
\bar{\mathbf{X}}_{\mathrm{BR}}
-
\bar{\mathbf{X}}_{\mathrm{BL}}
\right\|_2.
$$

The reconstructed width is taken as their average:

$$
\boxed{
\hat{W}
=
\frac{
W_{\mathrm{top}}
+
W_{\mathrm{bottom}}
}{2}
}
$$

Similarly,

$$
H_{\mathrm{left}}
=
\left\|
\bar{\mathbf{X}}_{\mathrm{BL}}
-
\bar{\mathbf{X}}_{\mathrm{TL}}
\right\|_2
$$

and

$$
H_{\mathrm{right}}
=
\left\|
\bar{\mathbf{X}}_{\mathrm{BR}}
-
\bar{\mathbf{X}}_{\mathrm{TR}}
\right\|_2.
$$

Therefore,

$$
\boxed{
\hat{H}
=
\frac{
H_{\mathrm{left}}
+
H_{\mathrm{right}}
}{2}
}
$$

The percentage reconstruction errors are

$$
\boxed{
E_W
=
\frac{
|\hat{W}-W|
}{
W
}
\times100\%
}
$$

and

$$
\boxed{
E_H
=
\frac{
|\hat{H}-H|
}{
H
}
\times100\%.
}
$$

Unlike the earlier automatic-contour approach, the reconstructed
boundary is now determined directly by the four physical corners
selected by the user. This prevents shadows, the object background,
or adjacent three-dimensional surfaces from being interpreted as
part of the planar boundary.

---

## 4.8 Reprojection error

A reconstructed planar point can be projected back into an
image using

$$
\hat{\mathbf{x}}
\sim
K
\begin{bmatrix}
\mathbf{r}_1 &
\mathbf{r}_2 &
\mathbf{t}
\end{bmatrix}
\mathbf{X}_{\mathrm{plane}}.
$$

For $N$ known points, the reprojection RMSE is

$$
\boxed{
E_{\mathrm{RMSE}}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
\mathbf{x}_i
-
\hat{\mathbf{x}}_i
\right\|_2^2
}.
}
$$

Smaller reprojection error indicates greater consistency among
the estimated camera parameters, pose and observed image
geometry.

---

## 4.9 ORB + RANSAC as independent multi-view evidence

The application additionally detects ORB descriptors inside
the object boundary.

Descriptors are matched between View 1 and each remaining
view. Ambiguous matches are removed using the nearest-neighbor
ratio criterion.

A robust view-to-view homography is then estimated with
RANSAC.

The reported quantities are:

- number of descriptor matches,
- number of RANSAC inliers,
- inlier ratio,
- transferred-boundary RMSE.

A high inlier ratio and low transferred-boundary error provide
independent evidence that the same planar structure is being
observed from different camera viewpoints.

---

## 4.10 Important planar-scene limitation

A completely planar scene is a special case in multi-view
geometry.

General unconstrained 3-D structure-from-motion normally relies
on parallax generated by points with different depths.
A single plane does not contain that depth variation and is
naturally modeled by a homography.

For this reason, this assignment implementation performs a
**planar metric reconstruction** rather than pretending to
recover arbitrary three-dimensional object geometry.

The known physical width and height of the object establish the
metric scale. Accurate camera intrinsics further improve the
accuracy of the recovered camera poses.
""")
