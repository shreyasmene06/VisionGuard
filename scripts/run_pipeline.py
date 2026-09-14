#!/usr/bin/env python3
"""VisionGuard: Computer Vision Based Road Surface Hazard Detection and Severity Mapping.

Primary Command Line Interface (CLI).
Supports single image processing, batch directory scanning, configurable thresholds,
and headless structured exports (JSON, CSV, annotated images).
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import PipelineConfig
from src.pipeline import VisionGuardPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VisionGuard execution."""
    parser = argparse.ArgumentParser(
        prog="visionguard",
        description="VisionGuard: Computer Vision Based Road Surface Hazard Detection and Severity Mapping",
        epilog=(
            "Example usages:\n"
            "  python scripts/run_pipeline.py --input data/sample/pothole_01.jpg --output outputs/test\n"
            "  python scripts/run_pipeline.py --input data/sample/ --output outputs/batch --confidence 0.35\n"
            "  python scripts/run_pipeline.py --input data/sample/crack_01.jpg --model classical --save-json --save-csv\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core I/O
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to an input road image (JPG/PNG) or a directory containing multiple images.",
    )
    parser.add_argument(
        "-o", "--output",
        default="outputs",
        type=str,
        help="Destination directory where annotated images, JSON, and CSV reports are saved (default: outputs/).",
    )

    # Detection parameters
    parser.add_argument(
        "-c", "--confidence",
        default=0.30,
        type=float,
        help="Minimum detection confidence threshold between 0.0 and 1.0 (default: 0.30).",
    )
    parser.add_argument(
        "-m", "--model",
        default="auto",
        type=str,
        help="Hazard detector engine or weights path: 'auto', 'classical', 'yolov8', or path to .pt weights (default: auto).",
    )
    parser.add_argument(
        "--iou-thresh",
        default=0.45,
        type=float,
        help="Non-Maximum Suppression (NMS) Intersection-over-Union threshold (default: 0.45).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device for deep learning inference: 'cpu' or 'cuda' (default: cpu).",
    )

    # Preprocessing toggles
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable CLAHE contrast enhancement and bilateral denoising stages.",
    )
    parser.add_argument(
        "--apply-roi",
        action="store_true",
        help="Apply road region-of-interest horizon masking to filter out sky/background.",
    )

    # Export & Visualization toggles
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save comprehensive JSON result telemetry (default: True).",
    )
    parser.add_argument(
        "--no-json",
        dest="save_json",
        action="store_false",
        help="Disable saving JSON telemetry.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=True,
        help="Save per-hazard and image summary CSV logs (default: True).",
    )
    parser.add_argument(
        "--no-csv",
        dest="save_csv",
        action="store_false",
        help="Disable saving CSV logs.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip generating annotated output images (useful for headless headless batch throughput).",
    )

    return parser.parse_args()


def print_banner() -> None:
    """Print clean terminal header banner."""
    print("=" * 78)
    print("  VISIONGUARD: Road Surface Hazard Detection & Severity Mapping Engine")
    print("  Autonomous Monocular Computer Vision Assessment | Pavement Condition Index")
    print("=" * 78)


def print_result_summary(result) -> None:
    """Format and print road inspection result summary to stdout."""
    res_dict = result.to_dict()
    pci_info = res_dict["pavement_condition_index"]
    hazards = res_dict["hazard_summary"]

    print("\n" + "-" * 78)
    print(f" Image: {result.image_name}")
    print(f" Resolution: {result.preprocess_result.width}x{result.preprocess_result.height}")
    print(f" Processing Time: {result.total_processing_time_ms:.1f} ms ({result.detection_result.model_name})")
    print(f" Pavement Condition Index (PCI): {pci_info['pci_score']:.1f}/100.0 --> [{pci_info['condition_state']}]")
    print(f" Total Hazards Detected: {hazards['total_detected']}")
    print(f" Severity Distribution: {hazards['by_severity_tier']}")
    print(f" Category Breakdown: {hazards['by_category']}")

    if result.assessments:
        print("\n Detected Hazards Detail:")
        header = f"   {'#':<3} | {'Category':<20} | {'Conf':<6} | {'Area %':<8} | {'Score':<6} | {'Tier':<8} | {'Centroid':<14}"
        print(header)
        print("   " + "-" * (len(header) - 3))
        for idx, a in enumerate(result.assessments):
            cx, cy = a.normalized_centroid
            print(
                f"   {idx+1:<3} | {a.detection.category.value:<20} | {a.detection.confidence:>5.2f} | "
                f"{a.relative_area_pct:>7.3f}% | {a.severity_score:>5.1f} | {a.severity_tier.value.upper():<8} | "
                f"({cx:.2f}, {cy:.2f})"
            )

    print("\n Engineering Recommendations:")
    for rec in res_dict["recommendations"]:
        print(f"   * {rec}")

    if result.annotated_image_path:
        print(f"\n [Output] Annotated image: {result.annotated_image_path}")
    if result.json_result_path:
        print(f" [Output] JSON telemetry: {result.json_result_path}")
    print("-" * 78)


def main() -> int:
    args = parse_args()
    print_banner()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assemble pipeline configuration
    backend = args.model.lower()
    weights_path = args.model if backend not in ("auto", "classical", "yolov8") else "weights/yolov8n_road.pt"

    config = PipelineConfig(
        apply_clahe=not args.no_preprocess,
        apply_bilateral_denoising=not args.no_preprocess,
        apply_road_roi=args.apply_roi,
        confidence_threshold=args.confidence,
        nms_iou_threshold=args.iou_thresh,
        detection_backend=backend if backend in ("auto", "classical", "yolov8") else "yolov8",
        model_weights_path=weights_path,
        device=args.device,
        generate_annotated_image=not args.no_visualize,
        save_json=args.save_json,
        save_csv=args.save_csv,
        default_output_dir=output_dir,
    )

    print(f"[*] Initializing VisionGuard Pipeline...")
    print(f"    - Detection Engine: {config.detection_backend}")
    print(f"    - Confidence Threshold: {config.confidence_threshold}")
    print(f"    - Preprocessing: {'Enabled (CLAHE + Bilateral)' if not args.no_preprocess else 'Disabled'}")
    print(f"    - Output Directory: {output_dir.resolve()}")

    pipeline = VisionGuardPipeline(config=config)

    if input_path.is_file():
        print(f"\n[*] Processing single image: {input_path}")
        result = pipeline.process_image(
            image_input=input_path,
            output_dir=output_dir,
            save_visual=not args.no_visualize,
            save_json=args.save_json,
            save_csv=args.save_csv,
        )
        print_result_summary(result)
        print("\n[SUCCESS] Inspection completed successfully.")
        return 0

    elif input_path.is_dir():
        print(f"\n[*] Scanning batch directory: {input_path}")
        results = pipeline.process_batch(
            input_directory=input_path,
            output_dir=output_dir,
        )
        print(f"\nProcessed {len(results)} image(s) from {input_path}")
        for res in results:
            print_result_summary(res)
        print(f"\n[Output] Master batch summary CSV: {output_dir / 'batch_summary.csv'}")
        print("\n[SUCCESS] Batch inspection completed successfully.")
        return 0

    else:
        print(f"[ERROR] Specified input path does not exist: {input_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
