# VisionGuard Datasets & Benchmark Annotation Specification

## 1. Dataset Overview
VisionGuard is designed to benchmark against real-world road inspection imagery derived from leading academic road damage benchmarks:

### A. Road Damage Dataset (RDD2020 / RDD2022)
- **Reference**: Seki et al., *"Global Road Damage Detection Challenge: Multimodal Road Surface Defect Analysis across Japan, India, and the Czech Republic"*, IEEE BigData.
- **Key Categories**:
  - `D00`: Longitudinal Linear Cracks (pavement joint deterioration)
  - `D10`: Transverse Linear Cracks (thermal expansion/contraction cracks)
  - `D20`: Alligator Cracks (mesh fatigue failure of asphalt sub-base)
  - `D40`: Potholes (deep asphalt depressions and cavitation)
- **Capture Modality**: Dashcam / smartphone monocular cameras mounted behind the vehicle windshield.

### B. CRACK500 Dataset
- **Reference**: Yang et al., *"Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection"*, IEEE Transactions on Intelligent Transportation Systems.
- **Focus**: High-resolution close-range pavement surface texture and alligator cracking.

---

## 2. Sample Dataset in `data/sample/`
To enable turnkey offline evaluation and automated testing without requiring multi-gigabyte dataset downloads, VisionGuard includes a curated, realistic test suite in `data/sample/`:

| File Name | Primary Condition | Scene Context | Ground Truth Annotations |
|---|---|---|---|
| `pothole_01.jpg` | Severe Pothole | Asphalt roadway wheel path | 1 Pothole, 1 Longitudinal Crack |
| `pothole_02.jpg` | Multiple Potholes | Urban worn street | 2 Potholes, Damaged Pavement |
| `crack_01.jpg` | Longitudinal Crack | Highway centerline joint | 1 Longitudinal Crack |
| `crack_02.jpg` | Alligator Fatigue Cracking | Weathered suburban pavement | 1 Alligator Crack Mesh |
| `debris_01.jpg` | Road Debris Obstacle | Driving lane tire hazard | 1 Road Debris, 1 Pothole |
| `clean_road_01.jpg` | Pristine Pavement | Newly paved asphalt highway | 0 Defects (Pristine PCI = 100) |

---

## 3. Ground Truth Annotation Schema (`annotations.json`)
The ground-truth annotations are provided in a standardized JSON schema:
```json
{
  "info": {
    "dataset": "VisionGuard Benchmark Dataset",
    "version": "1.0",
    "format": "COCO-compatible pixel bounding boxes [x1, y1, x2, y2]"
  },
  "categories": [
    "pothole",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "damaged_pavement",
    "road_debris"
  ],
  "images": [
    {
      "file_name": "pothole_01.jpg",
      "width": 800,
      "height": 600,
      "annotations": [
        {
          "category": "pothole",
          "bbox": [280, 360, 520, 520],
          "area_px": 38400
        }
      ]
    }
  ]
}
```

---

## 4. Evaluation Protocol
The automated evaluation runner (`scripts/evaluate.py`) assesses:
1. **Intersection-over-Union (IoU)** $\ge 0.50$ (PASCAL VOC / COCO metric).
2. **Category-Specific Precision, Recall, and F1-Scores**.
3. **Mean Average Precision (mAP@0.5)** across all hazard classes.
4. **Latency and FPS throughput** on CPU and CUDA devices.
