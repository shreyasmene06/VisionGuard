# VisionGuard: Problem Statement & Engineering Specification

**Coursework Project:** CS480 Computer Vision Engineering  
**Student Name:** Shreyas Mene  
**Registration No.:** 24BAI10018  
**System Name:** VisionGuard (Road Surface Hazard Detection and Severity Mapping)  

---

## 1. Background and Motivation
Road surface decay is one of the costliest problems faced by city public works departments. When surface asphalt cracks and holes go undetected during early formation, heavy vehicular traffic and rainwater seep into the sub-base layer, rapidly washing away foundation gravel. This transforms cheap routine repairs (like \$5/meter crack sealing) into massive structural failures requiring full road reconstruction.

Currently, most municipalities still rely on two manual inspection methods:
1. Two-person survey teams driving slowly along roads at 20 km/h with a clipboard or tablet, causing traffic delays and risking rear-end collisions.
2. Walking inspectors manually taking measurement wheels onto live road shoulders.

Both approaches are expensive, infrequent (conducted once every 2–3 years per road), and subjective. Two different human raters often score the same stretch of damaged road completely differently based on fatigue and weather conditions.

**VisionGuard** addresses this gap by providing an automated, reproducible, and computationally lightweight Computer Vision pipeline. It processes forward-facing road imagery captured from vehicle dashcams, extracts physical hazard geometries, computes a deterministic severity rating, and assigns an overall Pavement Condition Index (PCI) compliant with civil engineering standards.

---

## 2. Project Objectives
1. **Multi-Class Hazard Detection**: Identify distinct asphalt defects including potholes, longitudinal joints, transverse thermal cracks, alligator fatigue networks, and lane debris.
2. **Deterministic Severity Quantification**: Replace arbitrary score assignment with a transparent mathematical formulation combining bounding box area, physical hazard threat weighting, and camera perspective proximity.
3. **Pavement Condition Index (PCI)**: Aggregate detected defects into a standardized network condition rating (0 to 100) inspired by the ASTM D6433 standard.
4. **Zero-GUI Headless Operation**: Ensure the complete software suite runs from a headless terminal without requiring an active X11/Wayland display server or graphical window manager.
5. **Actionable Structured Outputs**: Save color-coded annotated images alongside structured machine-readable telemetry in JSON and CSV formats for easy import into GIS mapping tools.

---

## 3. Scope and System Boundaries
- **Supported Inputs**: Standard 2D monocular road images (JPG, PNG) captured from forward-facing windshield mounts or bumper cameras under daytime conditions.
- **Defect Categories Handled**:
  - `pothole`: Depth depressions, cavitation craters, impact holes.
  - `longitudinal_crack`: Cracks running parallel to road centerline / paving seams.
  - `transverse_crack`: Cracks running perpendicular across travel lanes.
  - `alligator_crack`: Interconnected polygonal fatigue networks indicating base failure.
  - `road_debris`: Foreign objects, dropped cargo, shredded tire treads.
  - `damaged_pavement`: Surface ravelling, aggregate stripping, rutting.
- **Out-of-Scope**:
  - Direct 3D volumetric laser scanning (LiDAR point clouds).
  - Wet nighttime specular reflection filtering (best handled by multi-frame temporal fusion).
  - Autonomous vehicle steering / braking actuation (this system operates as an inspection & survey tool, not an autopilot ADAS).

---

## 4. Functional Requirements (FR)
- **FR-1 (Image Ingestion & Preprocessing)**: The system must validate input image integrity, normalize dimensions, perform LAB-channel Contrast Limited Adaptive Histogram Equalization (CLAHE), and apply edge-preserving bilateral denoising.
- **FR-2 (Hazard Detection)**: The system must identify road surface hazards using either a deep learning YOLOv8 model or an interpretable classical morphology and contour geometry detector.
- **FR-3 (Geometric Feature Extraction)**: For each detected hazard, the system must extract pixel coordinates, bounding box width/height, area in $px^2$, frame percentage, and normalized centroid position.
- **FR-4 (Severity & Condition Scoring)**: The system must calculate a deterministic severity index ($S_i \in [0, 100]$) and an aggregate ASTM D6433 Pavement Condition Index ($\text{PCI} \in [0, 100]$).
- **FR-5 (Headless Visualization & Telemetry Export)**: The system must render HUD-annotated imagery, write detailed JSON inspection telemetry, and export per-hazard and batch CSV logs without opening any GUI windows.

---

## 5. Non-Functional Requirements (NFR)
- **NFR-1 (Zero GUI Dependency)**: All features must execute strictly from the command-line interface (`argparse`), capable of running in headless CI/CD containers or cloud servers without display hardware.
- **NFR-2 (Traceable Determinism)**: Severity and PCI formulas must be closed-form, deterministic mathematical functions with zero random constants or fabricated metrics.
- **NFR-3 (Modularity & Loose Coupling)**: Core components (`preprocessing`, `detector`, `severity`, `analyzer`, `visualization`, `pipeline`) must be decoupled through clean interfaces, allowing seamless swapping of model backends.
- **NFR-4 (Computational Efficiency)**: The pipeline must achieve real-time throughput (>30 FPS on CUDA GPUs, >10 FPS on standard multi-core CPUs).
- **NFR-5 (Testability & Verification)**: The repository must provide automated unit and integration tests with `pytest`, covering image validation, mathematical edge cases, and end-to-end file generation.
