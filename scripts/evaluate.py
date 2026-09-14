#!/usr/bin/env python3
"""VisionGuard: Automated Evaluation and Benchmark Runner.

Evaluates detector precision, recall, F1-score, IoU, and mAP against
ground-truth road damage annotations.
"""

import argparse
import json
from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import PipelineConfig
from src.metrics import evaluate_detections
from src.pipeline import VisionGuardPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visionguard-evaluate",
        description="Benchmark and evaluate VisionGuard hazard detection against ground-truth labels.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/sample",
        type=str,
        help="Directory containing test images and annotations.json (default: data/sample).",
    )
    parser.add_argument(
        "--annotations",
        default="data/sample/annotations.json",
        type=str,
        help="Path to ground truth JSON annotations file (default: data/sample/annotations.json).",
    )
    parser.add_argument(
        "--iou-thresh",
        default=0.50,
        type=float,
        help="IoU threshold for true positive match (default: 0.50).",
    )
    parser.add_argument(
        "--confidence",
        default=0.30,
        type=float,
        help="Detection confidence threshold (default: 0.30).",
    )
    parser.add_argument(
        "--model",
        default="auto",
        type=str,
        help="Detector backend: 'auto', 'classical', or 'yolov8' (default: auto).",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation_report.json",
        type=str,
        help="Path to save evaluation results JSON (default: outputs/evaluation_report.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    annot_path = Path(args.annotations)
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  VISIONGUARD: Hazard Detection Benchmark & Evaluation Runner")
    print("=" * 78)
    print(f"[*] Annotations: {annot_path.resolve()}")
    print(f"[*] IoU Threshold: {args.iou_thresh}")
    print(f"[*] Confidence Threshold: {args.confidence}")
    print(f"[*] Detector Backend: {args.model}")

    if not annot_path.is_file():
        print(f"[ERROR] Annotations file not found: {annot_path}", file=sys.stderr)
        return 1

    with open(annot_path, "r", encoding="utf-8") as f:
        ground_truth_raw = json.load(f)

    config = PipelineConfig(
        confidence_threshold=args.confidence,
        nms_iou_threshold=0.45,
        detection_backend=args.model,
    )
    pipeline = VisionGuardPipeline(config=config)

    predictions_by_image = {}
    ground_truths_by_image = {}
    total_latency_ms = 0.0
    image_count = 0

    print(f"\n[*] Running inference across dataset...")

    for item in ground_truth_raw.get("images", []):
        img_filename = item["file_name"]
        img_path = data_dir / img_filename
        if not img_path.is_file():
            print(f"  [WARN] Skipping missing image: {img_path}")
            continue

        start_t = time.perf_counter()
        res = pipeline.process_image(
            img_path,
            output_dir=out_file.parent / "eval_vis",
            save_visual=True,
            save_json=False,
            save_csv=False,
        )
        latency = (time.perf_counter() - start_t) * 1000.0
        total_latency_ms += latency
        image_count += 1

        predictions_by_image[img_filename] = [a.detection for a in res.assessments]

        # Extract GT boxes
        gt_boxes = []
        for ann in item.get("annotations", []):
            bbox = ann["bbox"]  # [x1, y1, x2, y2]
            cat = ann["category"]
            gt_boxes.append(((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), cat))
        ground_truths_by_image[img_filename] = gt_boxes

    # Run evaluation
    metrics = evaluate_detections(
        predictions_by_image=predictions_by_image,
        ground_truths_by_image=ground_truths_by_image,
        iou_threshold=args.iou_thresh,
    )

    avg_latency = total_latency_ms / max(1, image_count)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("\n" + "=" * 78)
    print("  EVALUATION RESULTS & BENCHMARK MATRIX")
    print("=" * 78)
    print(metrics.format_table())
    print("-" * 78)
    print(f" Average Latency per Frame: {avg_latency:.2f} ms ({fps:.1f} FPS)")
    print(f" Evaluated Images: {image_count}")
    print(f" Total Ground Truth Annotations: {metrics.total_ground_truth}")
    print(f" Total Pipeline Detections: {metrics.total_detections}")
    print("=" * 78)

    # Save to JSON
    report_dict = {
        "benchmark_metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluated_images": image_count,
            "average_latency_ms": round(avg_latency, 2),
            "throughput_fps": round(fps, 1),
            "iou_threshold": args.iou_thresh,
            "confidence_threshold": args.confidence,
            "model_name": args.model,
        },
        "summary": {
            "mean_precision": metrics.mean_precision,
            "mean_recall": metrics.mean_recall,
            "mean_f1": metrics.mean_f1,
            "map50": metrics.map50,
            "total_ground_truth": metrics.total_ground_truth,
            "total_detections": metrics.total_detections,
        },
        "per_class": {
            cat: {
                "tp": cm.true_positives,
                "fp": cm.false_positives,
                "fn": cm.false_negatives,
                "ground_truth_count": cm.ground_truth_count,
                "detection_count": cm.detection_count,
                "precision": cm.precision,
                "recall": cm.recall,
                "f1_score": cm.f1_score,
                "ap50": cm.ap50,
            }
            for cat, cm in metrics.class_metrics.items()
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\n[Output] Evaluation metrics saved to: {out_file.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
