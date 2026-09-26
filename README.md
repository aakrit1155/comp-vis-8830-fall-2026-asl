# comp-vis-8830-fall-2026-asl
Computer Vision code for CSC 8830 under Dr. Ashok Aswin for Fall 2026 at GSU
## Student: Aakrit Sharma Lamsal

# Computer Vision — CSC8830

A coursework repository for **Computer Vision (CSC8830)**. It contains assignment implementations and an interactive **Streamlit** web application that serves as a unified interface for running, demonstrating, and inspecting each assignment.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [Option A — Using `uv` (Recommended, Python 3.12)](#option-a--using-uv-recommended-python-312)
  - [Option B — Using `venv` + `pip`](#option-b--using-venv--pip)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Navigating the Assignments](#navigating-the-assignments)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repository is organized as a single project that grows incrementally as the course progresses. Each assignment lives in its own self-contained folder, while a top-level **Streamlit** application (`app.py`) provides a convenient, browser-based way to launch and evaluate the code for each assignment without needing to run scripts manually.

- **Course:** Computer Vision — CSC8830
- **Primary interface:** Streamlit web app (`app.py`)
- **Assignment folders:** `assignment_01/`, `assignment_02/`, … `assignment_N/`
- **Currently implemented:** `assignment_02`

---

## Repository Structure

```
.
├── app.py                 # Main Streamlit entry point (launch from repo root)
├── requirements.txt       # Python dependencies for the whole project
├── README.md              # This file
├── assignment_01/         # Assignment 1 source code, data, notebooks
│   ├── ...
├── assignment_02/         # Assignment 2 source code, data, notebooks
│   ├── ...
└── assignment_N/          # (Future assignments follow the same layout)
```

Each `assignment_XX/` folder contains all files relevant to that assignment (source code, helper modules, inputs/data, and any assignment-specific README). The Streamlit app in `app.py` discovers and exposes these assignments through the UI, so **you only need to launch a single server** to review everything.

---

## Prerequisites

- **Git** installed and available on your `PATH`
- A terminal / shell (bash, zsh, PowerShell, etc.)
- **Python 3.12**
- Either one of the following tooling options:
  - **[`uv`](https://docs.astral.sh/uv/)** — recommended, for fast, reproducible environment setup, **or**
  - **`python -m venv` + `pip`** — built-in alternative

---

## Setup Instructions

First, clone the repository and move into it:

```bash
git clone https://github.com/aakrit1155/comp-vis-8830-fall-2026-asl.git
cd comp-vis-8830-fall-2026-asl
```


Then choose **one** of the two environment setup options below.

### Option A — Using `uv` (Recommended, Python 3.12)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package/environment manager. If it is not already installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create a virtual environment pinned to **Python 3.12**:

```bash
uv venv --python 3.12
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

Install all dependencies:

```bash
uv pip install -r requirements.txt
```

### Option B — Using `venv` + `pip`

Create the virtual environment with Python 3.12 (make sure `python3.12` is on your `PATH`):

```bash
python3.12 -m venv .venv
```

> On Windows, you may use:
>
> ```bash
> py -3.12 -m venv .venv
> ```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

Upgrade `pip` and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the Streamlit App

All commands below must be run from the **repository root** (the folder containing `app.py`), with the virtual environment **activated**.

```bash
streamlit run app.py
```

By default, Streamlit starts a local server and opens your browser at:

```
http://localhost:8501
```

If the browser does not open automatically, copy the URL printed in the terminal into your browser. To stop the server, press `Ctrl+C` in the terminal.

Once the app is running, use the sidebar / in-app navigation to select the assignment you want to review. The app will load and execute the corresponding code from the matching `assignment_XX/` folder.

---

## Navigating the Assignments

- Each assignment is fully encapsulated in its own top-level folder: `assignment_01/`, `assignment_02/`, … `assignment_N/`.
- The naming convention is strict (`assignment_` + zero-padded two-digit index) so the Streamlit app can enumerate them in order.
- Inside each assignment folder you will typically find:
  - The Python source file(s) implementing the assignment logic.
  - Any helper modules, notebooks, or assets used only by that assignment.
  - A local `README.md` (where applicable) describing the task, methodology, and expected output.
- Assignment folders are added progressively. Only folders that exist in the repository at the time of review will be selectable in the app.

---

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| `streamlit: command not found` | Ensure the virtual environment is activated, and that `streamlit` is listed in `requirements.txt`. |
| `ModuleNotFoundError` when selecting an assignment | Re-run `uv pip install -r requirements.txt` (or `pip install -r requirements.txt`) inside the activated environment. |
| Port `8501` already in use | Run `streamlit run app.py --server.port 8502` (or another free port). |
| `uv venv --python 3.12` fails | Confirm Python 3.12 is installed, or let `uv` fetch it automatically with `uv python install 3.12`. |
| Wrong Python version active | Verify with `python --version`; it should report `Python 3.12.x`. |

---

## Notes for the Grader

1. Clone the repository and create the environment using **either** Option A or Option B.
2. Run `streamlit run app.py` from the repository root.
3. Use the in-app navigation to select respective assignment.
4. Each assignment folder is self-contained; no cross-assignment setup is required.