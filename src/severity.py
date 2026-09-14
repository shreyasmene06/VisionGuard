"""Deterministic Hazard Severity Scoring and Proximity Quantification.

Author: Shreyas Mene (CS480 Coursework)
Computes a transparent, closed-form severity score (0 to 100) combining:
- Relative damaged surface area (fraction of road camera view)
- Baseline physical hazard danger weight
- Monocular vertical perspective proximity (hazards closer to vehicle wheels)
- Detection certainty weight
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple
import numpy as np

from src.config import (
    BASE_CLASS_WEIGHTS,
    HazardCategory,
    SeverityTier,
    SeverityWeights,
)

logger = logging.getLogger("visionguard.severity")
from src.detector import Detection


@dataclass
class HazardSeverityAssessment:
    """Comprehensive severity and geometric assessment for a single hazard."""
    detection: Detection
    bbox_width: int
    bbox_height: int
    bbox_area_px: int
    image_area_px: int
    relative_area_pct: float         # Percentage of total image area (0.0 to 100.0)
    normalized_centroid: Tuple[float, float]  # (cx, cy) in [0.0, 1.0]
    aspect_ratio: float              # width / height
    base_class_weight: float         # Intrinsic class danger (0 - 100)
    proximity_factor: float          # Near-field perspective factor (0 - 100)
    severity_score: float            # Final composite severity score (0.0 to 100.0)
    severity_tier: SeverityTier      # Categorical tier (low, medium, high, critical)
    formula_breakdown: Dict[str, float]  # Component-wise contribution breakdown

    @property
    def category_name(self) -> str:
        return self.detection.category.value

    @property
    def confidence(self) -> float:
        return self.detection.confidence


class SeverityEngine:
    """Mathematical severity quantification engine.

    Formulation:
        S = min(100.0,
            w_a * AreaComponent +
            w_c * ClassComponent +
            w_p * ProximityComponent +
            w_f * ConfidenceComponent
        )

    Where:
        - AreaComponent: min(100.0, (BboxArea / ImageArea) * 1000.0)
          Standard road hazards typically occupy 0.1% to 10% of the image frame.
          Multiplying by 1000 scales a 10% damage zone to the maximum 100 score.
        - ClassComponent: Base physical danger of the hazard type:
          * Pothole: 45 (severe vehicle suspension damage / tire rupture risk)
          * Alligator Crack: 35 (extensive structural base failure)
          * Road Debris: 40 (collision / acute swerve hazard)
          * Damaged Pavement: 25 (surface ravelling / friction loss)
          * Transverse / Longitudinal Crack: 20 / 18 (early moisture intrusion defect)
        - ProximityComponent: (y_max / H_image) * 100.0
          In forward-facing road cameras, the lower part of the frame (y_max -> H)
          corresponds to the near-field roadway directly ahead of the vehicle wheels,
          representing immediate collision and impact risk.
        - ConfidenceComponent: (Confidence * 100.0)
          Ensures well-defined, prominent hazards with clear visual signatures
          are prioritized over marginal or faint detections.

    Limitations:
        - Assumes standard monocular camera orientation (looking forward and slightly down).
        - Relative area is a 2D projection approximation; actual physical 3D depth and
          pothole crater volume would require stereo vision, LiDAR, or photometric stereo.
    """

    def __init__(self, weights: SeverityWeights = None):
        self.weights = weights or SeverityWeights()

    def assess_hazard(
        self,
        detection: Detection,
        image_shape: Tuple[int, int, int],
    ) -> HazardSeverityAssessment:
        """Compute transparent mathematical severity for a single detection."""
        img_h, img_w = image_shape[:2]
        img_area = max(1, img_h * img_w)

        x1, y1, x2, y2 = detection.box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        bbox_area = bw * bh

        # 1. Relative area
        relative_area_pct = round((bbox_area / img_area) * 100.0, 3)
        # Scaled area component: 10% of frame reaches 100.0
        area_component = min(100.0, (bbox_area / img_area) * 1000.0)

        # 2. Intrinsic class danger
        base_class_weight = BASE_CLASS_WEIGHTS.get(detection.category, 25.0)
        class_component = base_class_weight

        # 3. Proximity factor (vertical position in frame)
        # y2 is bottom edge of hazard; closer to img_h means closer to camera
        norm_y2 = min(1.0, max(0.0, y2 / float(img_h)))
        proximity_factor = norm_y2 * 100.0
        proximity_component = proximity_factor

        # 4. Confidence certainty
        confidence_component = detection.confidence * 100.0

        # Composite score
        raw_severity = (
            self.weights.weight_area * area_component
            + self.weights.weight_class * class_component
            + self.weights.weight_proximity * proximity_component
            + self.weights.weight_confidence * confidence_component
        )
        severity_score = round(float(np.clip(raw_severity, 0.0, 100.0)), 2)

        # Categorical Severity Tier
        if severity_score >= 75.0:
            tier = SeverityTier.CRITICAL
        elif severity_score >= 50.0:
            tier = SeverityTier.HIGH
        elif severity_score >= 30.0:
            tier = SeverityTier.MEDIUM
        else:
            tier = SeverityTier.LOW

        # Centroid & aspect ratio
        cx = round(float((x1 + x2) / 2.0) / img_w, 4)
        cy = round(float((y1 + y2) / 2.0) / img_h, 4)
        aspect_ratio = round(float(bw) / float(bh), 3)

        breakdown = {
            "area_component": round(area_component, 2),
            "area_weighted": round(self.weights.weight_area * area_component, 2),
            "class_component": round(class_component, 2),
            "class_weighted": round(self.weights.weight_class * class_component, 2),
            "proximity_component": round(proximity_component, 2),
            "proximity_weighted": round(self.weights.weight_proximity * proximity_component, 2),
            "confidence_component": round(confidence_component, 2),
            "confidence_weighted": round(self.weights.weight_confidence * confidence_component, 2),
        }

        return HazardSeverityAssessment(
            detection=detection,
            bbox_width=bw,
            bbox_height=bh,
            bbox_area_px=bbox_area,
            image_area_px=img_area,
            relative_area_pct=relative_area_pct,
            normalized_centroid=(cx, cy),
            aspect_ratio=aspect_ratio,
            base_class_weight=base_class_weight,
            proximity_factor=round(proximity_factor, 2),
            severity_score=severity_score,
            severity_tier=tier,
            formula_breakdown=breakdown,
        )

    def assess_all(
        self,
        detections: List[Detection],
        image_shape: Tuple[int, int, int],
    ) -> List[HazardSeverityAssessment]:
        """Assess all detections in an image and sort descending by severity score."""
        assessments = [self.assess_hazard(det, image_shape) for det in detections]
        assessments.sort(key=lambda a: a.severity_score, reverse=True)
        return assessments
