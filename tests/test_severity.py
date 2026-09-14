"""Unit tests for Module 3: Severity and Damage Analysis."""

import pytest
from src.config import HazardCategory, SeverityTier, SeverityWeights
from src.detector import Detection
from src.severity import SeverityEngine


@pytest.fixture
def severity_engine() -> SeverityEngine:
    return SeverityEngine()


def test_assess_hazard_pothole(severity_engine):
    # Pothole near bottom of image (near-field) with 5% relative area
    # Image is 1000x1000, pothole is at y1=700, y2=850, x1=400, x2=600 -> w=200, h=150, area=30,000 (3%)
    det = Detection(
        box=(400, 700, 600, 850),
        category=HazardCategory.POTHOLE,
        confidence=0.90,
        source="classical_cv",
    )
    assessment = severity_engine.assess_hazard(det, image_shape=(1000, 1000, 3))

    assert assessment.bbox_width == 200
    assert assessment.bbox_height == 150
    assert assessment.bbox_area_px == 30000
    assert assessment.relative_area_pct == 3.0
    assert assessment.proximity_factor == 85.0  # y2=850 / 1000 * 100
    assert assessment.base_class_weight == 45.0
    assert 0.0 <= assessment.severity_score <= 100.0
    assert assessment.severity_tier in (SeverityTier.HIGH, SeverityTier.CRITICAL)
    assert "area_component" in assessment.formula_breakdown


def test_proximity_weighting_effect(severity_engine):
    # Far-field crack (top of road, y2=200) vs Near-field crack (bottom of road, y2=950)
    det_far = Detection(box=(200, 100, 400, 200), category=HazardCategory.LONGITUDINAL_CRACK, confidence=0.80)
    det_near = Detection(box=(200, 850, 400, 950), category=HazardCategory.LONGITUDINAL_CRACK, confidence=0.80)

    assess_far = severity_engine.assess_hazard(det_far, image_shape=(1000, 1000, 3))
    assess_near = severity_engine.assess_hazard(det_near, image_shape=(1000, 1000, 3))

    assert assess_near.proximity_factor > assess_far.proximity_factor
    assert assess_near.severity_score > assess_far.severity_score


def test_class_danger_ranking(severity_engine):
    # Identical box and proximity, but Pothole vs Longitudinal Crack
    det_pothole = Detection(box=(200, 500, 400, 600), category=HazardCategory.POTHOLE, confidence=0.80)
    det_crack = Detection(box=(200, 500, 400, 600), category=HazardCategory.LONGITUDINAL_CRACK, confidence=0.80)

    assess_pothole = severity_engine.assess_hazard(det_pothole, image_shape=(1000, 1000, 3))
    assess_crack = severity_engine.assess_hazard(det_crack, image_shape=(1000, 1000, 3))

    assert assess_pothole.base_class_weight > assess_crack.base_class_weight
    assert assess_pothole.severity_score > assess_crack.severity_score


def test_severity_tier_mapping(severity_engine):
    # Extremely minor crack vs catastrophic pothole
    minor_det = Detection(box=(100, 100, 110, 120), category=HazardCategory.LONGITUDINAL_CRACK, confidence=0.35)
    catastrophic_det = Detection(box=(200, 600, 700, 950), category=HazardCategory.POTHOLE, confidence=0.98)

    assess_minor = severity_engine.assess_hazard(minor_det, image_shape=(1000, 1000, 3))
    assess_catastrophic = severity_engine.assess_hazard(catastrophic_det, image_shape=(1000, 1000, 3))

    assert assess_minor.severity_tier == SeverityTier.LOW
    assert assess_catastrophic.severity_tier == SeverityTier.CRITICAL
