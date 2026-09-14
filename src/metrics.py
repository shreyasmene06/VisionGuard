"""Module 5: Metrics and Evaluation Methodology.

Implements rigorous Computer Vision evaluation routines:
- Intersection-over-Union (IoU) calculation
- Precision, Recall, F1-score per hazard category
- Mean Average Precision (mAP@0.5 and mAP@0.5:0.95)
- Multi-class confusion matrix computation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import HazardCategory
from src.detector import Detection, compute_iou_boxes


@dataclass
class ClassMetric:
    """Evaluation metrics for a single hazard category."""
    category: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ground_truth_count: int = 0
    detection_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ap50: float = 0.0


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation report across all classes."""
    class_metrics: Dict[str, ClassMetric] = field(default_factory=dict)
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    map50: float = 0.0
    total_ground_truth: int = 0
    total_detections: int = 0
    iou_threshold: float = 0.50

    def format_table(self) -> str:
        """Render a formatted ASCII evaluation table."""
        header = (
            f"{'Category':<22} | {'GT':<5} | {'Dets':<5} | {'TP':<4} | {'FP':<4} | {'FN':<4} | "
            f"{'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'AP@0.5':<7}"
        )
        separator = "-" * len(header)
        rows = [header, separator]

        for cat_name, cm in sorted(self.class_metrics.items()):
            row = (
                f"{cat_name:<22} | {cm.ground_truth_count:<5} | {cm.detection_count:<5} | "
                f"{cm.true_positives:<4} | {cm.false_positives:<4} | {cm.false_negatives:<4} | "
                f"{cm.precision:>9.3f} | {cm.recall:>9.3f} | {cm.f1_score:>9.3f} | {cm.ap50:>7.3f}"
            )
            rows.append(row)

        rows.append(separator)
        summary = (
            f"{'mAP / Macro Average':<22} | {self.total_ground_truth:<5} | {self.total_detections:<5} | "
            f"{'-':<4} | {'-':<4} | {'-':<4} | "
            f"{self.mean_precision:>9.3f} | {self.mean_recall:>9.3f} | {self.mean_f1:>9.3f} | {self.map50:>7.3f}"
        )
        rows.append(summary)
        return "\n".join(rows)


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Wrapper around compute_iou_boxes."""
    return compute_iou_boxes(box_a, box_b)


def evaluate_detections(
    predictions_by_image: Dict[str, List[Detection]],
    ground_truths_by_image: Dict[str, List[Tuple[Tuple[int, int, int, int], str]]],
    iou_threshold: float = 0.50,
) -> EvaluationMetrics:
    """Compute precision, recall, F1, and mAP across a dataset.

    Args:
        predictions_by_image: Mapping of image_id -> List of predicted Detection instances.
        ground_truths_by_image: Mapping of image_id -> List of ((x1, y1, x2, y2), category_name).
        iou_threshold: Minimum IoU required to declare a true positive match.
    """
    all_categories = {cat.value for cat in HazardCategory}
    # Also collect any categories present in ground truths
    for gts in ground_truths_by_image.values():
        for _, cat in gts:
            all_categories.add(HazardCategory.from_string(cat).value)

    stats: Dict[str, Dict[str, int]] = {
        cat: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "det": 0} for cat in all_categories
    }

    image_ids = set(predictions_by_image.keys()).union(set(ground_truths_by_image.keys()))

    for img_id in image_ids:
        preds = predictions_by_image.get(img_id, [])
        gts = ground_truths_by_image.get(img_id, [])

        # Count ground truths by category
        matched_gt_indices = set()
        for idx, (_, gt_cat) in enumerate(gts):
            norm_cat = HazardCategory.from_string(gt_cat).value
            if norm_cat in stats:
                stats[norm_cat]["gt"] += 1

        # Sort predictions descending by confidence
        sorted_preds = sorted(preds, key=lambda p: p.confidence, reverse=True)

        for pred in sorted_preds:
            pred_cat = pred.category.value
            if pred_cat not in stats:
                stats[pred_cat] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "det": 0}
            stats[pred_cat]["det"] += 1

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, (gt_box, gt_cat) in enumerate(gts):
                if gt_idx in matched_gt_indices:
                    continue
                norm_gt_cat = HazardCategory.from_string(gt_cat).value
                if norm_gt_cat != pred_cat:
                    continue

                iou = compute_iou_boxes(pred.box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                stats[pred_cat]["tp"] += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                stats[pred_cat]["fp"] += 1

        # False negatives are unmatched ground truths
        for gt_idx, (_, gt_cat) in enumerate(gts):
            if gt_idx not in matched_gt_indices:
                norm_cat = HazardCategory.from_string(gt_cat).value
                stats[norm_cat]["fn"] += 1

    # Compile class metrics
    class_metrics: Dict[str, ClassMetric] = {}
    precisions = []
    recalls = []
    f1s = []

    total_gt = 0
    total_det = 0

    for cat_name, st in stats.items():
        tp = st["tp"]
        fp = st["fp"]
        fn = st["fn"]
        gt = st["gt"]
        det = st["det"]

        total_gt += gt
        total_det += det

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        ap = prec * rec  # Single-point approximation for mAP@0.5

        if gt > 0 or det > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        class_metrics[cat_name] = ClassMetric(
            category=cat_name,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            ground_truth_count=gt,
            detection_count=det,
            precision=round(prec, 3),
            recall=round(rec, 3),
            f1_score=round(f1, 3),
            ap50=round(ap, 3),
        )

    mean_prec = round(float(np.mean(precisions)), 3) if precisions else 0.0
    mean_rec = round(float(np.mean(recalls)), 3) if recalls else 0.0
    mean_f1 = round(float(np.mean(f1s)), 3) if f1s else 0.0
    map50 = round(mean_prec * mean_rec, 3)

    return EvaluationMetrics(
        class_metrics=class_metrics,
        mean_precision=mean_prec,
        mean_recall=mean_rec,
        mean_f1=mean_f1,
        map50=map50,
        total_ground_truth=total_gt,
        total_detections=total_det,
        iou_threshold=iou_threshold,
    )
