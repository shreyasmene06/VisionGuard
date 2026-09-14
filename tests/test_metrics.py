"""Unit tests for Module 5: Metrics and Evaluation Methodology."""

import pytest
from src.config import HazardCategory
from src.detector import Detection
from src.metrics import compute_iou, evaluate_detections


def test_compute_iou_perfect_match():
    box_a = (50, 50, 150, 150)
    box_b = (50, 50, 150, 150)
    assert compute_iou(box_a, box_b) == 1.0


def test_compute_iou_disjoint():
    box_a = (0, 0, 50, 50)
    box_b = (60, 60, 100, 100)
    assert compute_iou(box_a, box_b) == 0.0


def test_compute_iou_partial_overlap():
    # box_a: 100x100 -> area 10000
    box_a = (0, 0, 100, 100)
    # box_b: overlap 50x100 -> area 5000, union 15000 -> IoU = 5000/15000 = 0.333
    box_b = (50, 0, 150, 100)
    iou = compute_iou(box_a, box_b)
    assert round(iou, 3) == round(5000 / 15000, 3)


def test_evaluate_detections_perfect():
    preds = {
        "img1": [
            Detection(box=(10, 10, 50, 50), category=HazardCategory.POTHOLE, confidence=0.95),
            Detection(box=(60, 60, 100, 100), category=HazardCategory.LONGITUDINAL_CRACK, confidence=0.85),
        ]
    }
    gts = {
        "img1": [
            ((10, 10, 50, 50), "pothole"),
            ((60, 60, 100, 100), "longitudinal_crack"),
        ]
    }
    metrics = evaluate_detections(preds, gts, iou_threshold=0.5)

    assert metrics.total_ground_truth == 2
    assert metrics.total_detections == 2
    assert metrics.mean_precision == 1.0
    assert metrics.mean_recall == 1.0
    assert metrics.mean_f1 == 1.0
    assert metrics.map50 == 1.0


def test_evaluate_detections_mismatch_and_miss():
    preds = {
        "img1": [
            # False positive: wrong class
            Detection(box=(10, 10, 50, 50), category=HazardCategory.ROAD_DEBRIS, confidence=0.90),
        ]
    }
    gts = {
        "img1": [
            # Ground truth is pothole
            ((10, 10, 50, 50), "pothole"),
        ]
    }
    metrics = evaluate_detections(preds, gts, iou_threshold=0.5)

    assert metrics.class_metrics["road_debris"].false_positives == 1
    assert metrics.class_metrics["pothole"].false_negatives == 1
    assert metrics.mean_precision == 0.0
    assert metrics.mean_recall == 0.0
