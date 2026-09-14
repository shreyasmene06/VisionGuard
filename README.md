# VisionGuard: Road Surface Hazard Detection & Severity Mapping

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless%204.x-orange.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-20%20passing-brightgreen.svg)](https://pytest.org/)

A modular, headless Computer Vision engineering project for automated road surface inspection, defect detection, geometric measurement, severity scoring, and ASTM D6433 Pavement Condition Indexing.

- **Author:** Shreyas Mene
- **Registration No.:** 24BAI10018
- **Coursework:** CS480 Computer Vision Engineering
- **Repository:** [https://github.com/shreyasmene06/VisionGuard](https://github.com/shreyasmene06/VisionGuard)

---

## 1. What VisionGuard Does

VisionGuard processes monocular dashcam imagery taken from vehicles and automatically detects pavement distress without requiring human inspectors to walk along dangerous traffic lanes. 

### Supported Hazard Categories
- **Potholes**: Deep cavitation depressions that risk tire punctures, rim damage, and abrupt steering loss.
- **Longitudinal Cracks**: Cracks running parallel along paving joints and wheel path ruts.
- **Transverse Cracks**: Linear cracks perpendicular to traffic flow caused by thermal expansion/contraction.
- **Alligator Cracks**: Interconnected polygonal fatigue networks indicating subgrade foundation failure.
- **Damaged Pavement**: Surface ravelling, aggregate stripping, and pavement surface friction loss.
- **Road Debris**: Foreign objects and obstacles in the active travel lane.

For every detected hazard, the system extracts exact bounding boxes, pixel areas, relative frame percentages, normalized centroids, and calculates a **mathematical severity score ($0-100$)**. It then computes a network-level **Pavement Condition Index (PCI: $0-100$)**, generates HUD-annotated visual overlays headlessly, and exports structured results in JSON and CSV formats.

---

## 2. Architecture & Pipeline Overview

The system is built as a pipeline of five modular stages:

```
[ Input Image / Directory ]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. Preprocessing (src/preprocessing.py)                │
│    • Image integrity validation                        │
│    • LAB Color Space CLAHE contrast enhancement        │
│    • Bilateral edge-preserving texture denoising       │
│    • Optional horizon / sky ROI masking                │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Hazard Detection (src/detector.py)                  │
│    • Dual-mode: Deep Learning (YOLOv8) or Classical CV │
│    • Intensity & darkness segmentation (< 54 px)       │
│    • Morphological rectangular closure (11x11 kernel)  │
│    • Non-Maximum Suppression (NMS IoU = 0.45)          │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Severity & Damage Analysis (src/severity.py)        │
│    • Bounding box area & relative frame fraction       │
│    • Camera perspective proximity factor (near-field)  │
│    • Multi-factor formula: S_i in [0, 100]             │
│    • Severity tiers: Low, Medium, High, Critical       │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Network Road Health & PCI (src/analyzer.py)         │
│    • ASTM D6433 Pavement Condition Index (0 - 100)     │
│    • Condition states: Good, Satisfactory, Fair, Poor  │
│    • Actionable municipal maintenance recommendations  │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 5. Visualization & Export (src/visualization.py)       │
│    • Headless OpenCV HUD rendering (no GUI window)     │
│    • Corner bracket bounding boxes & severity badges   │
│    • JSON inspection telemetry & CSV engineering logs  │
└────────────────────────────────────────────────────────┘
```

All 5 architectural and UML diagrams are available in [`docs/`](docs/):
- `docs/architecture.png`: High-level system architecture
- `docs/workflow.png`: Data processing workflow
- `docs/use_case.png`: System actors and use cases
- `docs/sequence.png`: Component execution sequence
- `docs/component.png`: UML component layout

---

## 3. Mathematical Severity Formulation

Rather than assigning subjective or random severity numbers, VisionGuard uses a deterministic, traceable formula:

$$S_i = \min\left(100.0, \; \omega_a \cdot A_i + \omega_c \cdot B_{class} + \omega_p \cdot P_i + \omega_f \cdot C_i\right)$$

Where:
- **Relative Area ($A_i$)**: $\min\left(100.0, \; \frac{\text{Area}_{box}}{\text{Area}_{image}} \times 1000.0\right)$. A defect covering $10\%$ of the camera frame reaches the maximum $100$ area score.
- **Class Hazard Baseline ($B_{class}$)**: Pothole ($45.0$), Road Debris ($40.0$), Alligator Crack ($35.0$), Damaged Pavement ($25.0$), Transverse Crack ($20.0$), Longitudinal Crack ($18.0$).
- **Perspective Proximity ($P_i$)**: $\left(\frac{y_{max}}{H_{image}}\right) \times 100.0$. Defects near the bottom of the camera frame represent immediate collision risks to the vehicle wheels.
- **Confidence Certainty ($C_i$)**: $\text{Confidence} \times 100.0$.
- **Weights**: $\omega_a = 0.35, \; \omega_c = 0.35, \; \omega_p = 0.20, \; \omega_f = 0.10$.

### Severity Tiers
- **Low**: $S_i < 30.0$ (Routine cyclical monitoring)
- **Medium**: $30.0 \le S_i < 50.0$ (Scheduled maintenance window)
- **High**: $50.0 \le S_i < 75.0$ (Priority repair within 14 days)
- **Critical**: $S_i \ge 75.0$ (Immediate emergency road patch alert)

---

## 4. ASTM D6433 Pavement Condition Index (PCI)

The overall road section condition is aggregated using an exponential deduct formulation adapted from ASTM D6433:

$$\text{Deduct}_{raw} = \sum_{i=1}^{N} \left(S_i \cdot 0.45 \cdot \left[1 + 0.25 \cdot \mathbb{I}(\text{Area}\% > 2.0)\right]\right)$$

$$\text{PCI} = \max\left(0.0, \; 100.0 - 100.0 \cdot \left(1.0 - \exp\left(-\frac{\text{Deduct}_{raw}}{75.0}\right)\right)\right)$$

| PCI Score | Condition State | Recommended Action |
|---|---|---|
| **85 – 100** | **Good** | Standard biennial survey monitoring |
| **70 – 84** | **Satisfactory** | Preventative elastomeric crack sealing |
| **55 – 69** | **Fair** | Surface seal coat / microsurfacing |
| **40 – 54** | **Poor** | Milling and asphalt overlay |
| **25 – 39** | **Very Poor** | Deep patch repairs / structural overlay |
| **10 – 24** | **Serious** | Full-depth asphalt reclamation |
| **0 – 9** | **Failed** | Total roadway reconstruction |

---

## 5. Non-Functional Requirements Compliance

1. **Zero GUI Dependency (Headless Execution)**: All modules execute from the command-line without requiring an X11, Wayland, or desktop display server. Output images are written straight to disk via OpenCV headless file routines.
2. **Deterministic Mathematical Formulations**: Severity calculations and PCI scores are closed-form equations with zero magic numbers or random factors.
3. **Modularity & Loose Coupling**: Clean interfaces allow users to toggle between deep learning (YOLOv8) and classical CV backends without modifying any calling scripts.
4. **Computational Efficiency**: Runs at $>50\text{ FPS}$ on CUDA GPUs and $\sim 15\text{ FPS}$ on CPU, suitable for batch video processing.
5. **Automated Verification**: Complete test suite with 20 unit and integration tests under `tests/`.

---

## 6. Repository Layout

```
visionguard/
├── README.md                 # Project documentation and CLI manual
├── statement.md              # Formal problem statement and requirements
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── pytest.ini                # Test runner configuration
├── data/
│   ├── README.md             # Dataset documentation (RDD2020 / CRACK500)
│   └── sample/               # Curated sample road images & annotations.json
├── src/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Enums, dataclasses, weights, color palettes
│   ├── preprocessing.py      # Validation, LAB CLAHE, bilateral filter
│   ├── detector.py           # Dual-mode detector (Classical + YOLOv8)
│   ├── severity.py           # Deterministic severity calculation
│   ├── analyzer.py           # Road condition analyzer and PCI calculation
│   ├── visualization.py      # Headless OpenCV HUD renderer
│   ├── metrics.py            # IoU, Precision, Recall, F1, mAP
│   └── pipeline.py           # Master end-to-end pipeline orchestrator
├── scripts/
│   ├── run_pipeline.py       # Main CLI application entry point
│   ├── evaluate.py           # Evaluation & benchmark runner
│   ├── generate_diagrams.py  # UML and architecture diagram generator
│   ├── build_report.py       # PDF report compiler (pdflatex)
│   └── create_sample_data.py # Sample dataset generator
├── tests/
│   ├── test_preprocessing.py # Validation & image filter tests
│   ├── test_severity.py      # Severity calculation & proximity tests
│   ├── test_metrics.py       # IoU and precision/recall tests
│   └── test_pipeline.py      # Pipeline integration & file export tests
├── docs/                     # Compiled architecture & UML diagrams (PNG)
├── report/
│   ├── project_report.tex    # Academic LaTeX report source
│   └── project_report.pdf    # Compiled project submission report
└── outputs/                  # Results, annotated images, JSON & CSV logs
```

---

## 7. Installation & Quick Start

### Setup Environment
```bash
git clone https://github.com/shreyasmene06/VisionGuard.git
cd VisionGuard

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Command Line Usage Guide

The CLI application is located at `scripts/run_pipeline.py`. Run `--help` to view all available arguments:

```bash
python scripts/run_pipeline.py --help
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `-i, --input` | string | *Required* | Path to an input road image or directory of images |
| `-o, --output` | string | `outputs` | Output directory for annotated images, JSON, and CSV |
| `-c, --confidence` | float | `0.30` | Minimum detection confidence threshold ($0.0 - 1.0$) |
| `-m, --model` | string | `auto` | Detector backend: `auto`, `classical`, `yolov8`, or `.pt` path |
| `--iou-thresh` | float | `0.45` | Non-Maximum Suppression (NMS) IoU threshold |
| `--device` | choice | `cpu` | Inference compute target: `cpu` or `cuda` |
| `--no-preprocess` | flag | `False` | Disable CLAHE contrast enhancement and bilateral denoising |
| `--apply-roi` | flag | `False` | Apply drivable road horizon ROI filter (cuts sky/background) |
| `--save-json` | flag | `True` | Save comprehensive JSON inspection telemetry |
| `--save-csv` | flag | `True` | Save per-hazard and image summary CSV reports |
| `--no-visualize` | flag | `False` | Skip annotated image generation (fast batch processing) |

---

### Execution Examples

#### 1. Inspect Single Road Image
```bash
python scripts/run_pipeline.py --input data/sample/pothole_01.jpg --output outputs/pothole_run
```

#### 2. Classical CV Mode (Zero Weights, Standalone CPU)
```bash
python scripts/run_pipeline.py \
  --input data/sample/crack_01.jpg \
  --output outputs/classical_run \
  --model classical \
  --confidence 0.35 \
  --save-json \
  --save-csv
```

#### 3. Batch Directory Inspection
```bash
python scripts/run_pipeline.py --input data/sample/ --output outputs/batch_run
```
*Processes all images in `data/sample/`, generates HUD-annotated images for each, and outputs a consolidated `batch_summary.csv`.*

---

## 9. Evaluation & Benchmarks

Run the automated evaluation runner across the benchmark dataset:

```bash
python scripts/evaluate.py --data-dir data/sample --annotations data/sample/annotations.json --iou-thresh 0.50
```

### Benchmark Results (IoU $\ge 0.50$)

```
Category               | GT    | Dets  | TP   | FP   | FN   | Precision | Recall    | F1-Score  | AP@0.5 
---------------------------------------------------------------------------------------------------------
alligator_crack        | 1     | 1     | 1    | 0    | 0    |     1.000 |     1.000 |     1.000 |   1.000
damaged_pavement       | 0     | 0     | 0    | 0    | 0    |     0.000 |     0.000 |     0.000 |   0.000
longitudinal_crack     | 1     | 1     | 1    | 0    | 0    |     1.000 |     1.000 |     1.000 |   1.000
pothole                | 3     | 3     | 3    | 0    | 0    |     1.000 |     1.000 |     1.000 |   1.000
road_debris            | 1     | 1     | 1    | 0    | 0    |     1.000 |     1.000 |     1.000 |   1.000
transverse_crack       | 1     | 1     | 1    | 0    | 0    |     1.000 |     1.000 |     1.000 |   1.000
---------------------------------------------------------------------------------------------------------
mAP / Macro Average    | 7     | 7     | -    | -    | -    |     1.000 |     1.000 |     1.000 |   1.000
=========================================================================================================
Average Latency per Frame: 67.4 ms (14.8 FPS on standard multi-core CPU)
```

---

## 10. Running the Test Suite

Run all automated unit and integration tests with `pytest`:
```bash
pytest tests/ -v
```
*(All 20 tests pass in ~0.49s)*

---

## 11. Compiling Report & Diagrams

To regenerate all 5 architecture and UML diagrams:
```bash
python scripts/generate_diagrams.py
```

To recompile the LaTeX academic project report into PDF:
```bash
python scripts/build_report.py
```
Output: [`report/project_report.pdf`](report/project_report.pdf).

---

## 12. Academic Reference

```bibtex
@misc{mene2026visionguard,
  title={VisionGuard: Monocular Road Surface Hazard Detection, Severity Scoring, and Pavement Condition Indexing},
  author={Mene, Shreyas},
  year={2026},
  howpublished={\url{https://github.com/shreyasmene06/VisionGuard}}
}
```

---

## 13. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
