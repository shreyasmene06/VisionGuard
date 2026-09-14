"""VisionGuard End-to-End Pipeline Orchestrator.

Author: Shreyas Mene (CS480 Coursework)
Coordinates the sequential flow from image ingestion to preprocessing,
hazard detection, severity calculation, road condition analysis,
and structured JSON/CSV telemetry exports.
"""

import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.analyzer import RoadAnalyzer, RoadConditionAnalysis
from src.config import PipelineConfig
from src.detector import BaseHazardDetector, DetectionResult, HybridHazardDetector
from src.preprocessing import ImagePreprocessor, PreprocessResult
from src.severity import HazardSeverityAssessment, SeverityEngine
from src.visualization import RoadVisualizer

logger = logging.getLogger("visionguard.pipeline")


@dataclass
class PipelineResult:
    """Encapsulates all generated data and file paths from processing a road image."""
    image_path: str
    image_name: str
    preprocess_result: PreprocessResult
    detection_result: DetectionResult
    assessments: List[HazardSeverityAssessment]
    analysis: RoadConditionAnalysis
    annotated_image_path: Optional[str] = None
    json_result_path: Optional[str] = None
    total_processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a structured, serializable dictionary."""
        return {
            "metadata": {
                "system": "VisionGuard Road Surface Hazard Detection",
                "version": "1.0.0",
                "image_path": self.image_path,
                "image_name": self.image_name,
                "resolution": {
                    "width": self.preprocess_result.width,
                    "height": self.preprocess_result.height,
                    "channels": self.preprocess_result.channels,
                },
                "total_processing_time_ms": self.total_processing_time_ms,
                "detection_inference_time_ms": self.detection_result.inference_time_ms,
                "detector_model": self.detection_result.model_name,
            },
            "pavement_condition_index": {
                "pci_score": self.analysis.pci_score,
                "condition_state": self.analysis.condition_state.value,
                "total_deduct_points": self.analysis.total_deduct_points,
                "max_severity_score": self.analysis.max_severity_score,
                "mean_severity_score": self.analysis.mean_severity_score,
            },
            "hazard_summary": {
                "total_detected": self.analysis.total_hazards,
                "by_category": self.analysis.category_counts,
                "by_severity_tier": self.analysis.severity_distribution,
            },
            "recommendations": self.analysis.recommendations,
            "hazards": [
                {
                    "hazard_id": idx + 1,
                    "category": a.detection.category.value,
                    "confidence": a.detection.confidence,
                    "source": a.detection.source,
                    "bounding_box": {
                        "x1": a.detection.x1,
                        "y1": a.detection.y1,
                        "x2": a.detection.x2,
                        "y2": a.detection.y2,
                        "width": a.bbox_width,
                        "height": a.bbox_height,
                        "pixel_area": a.bbox_area_px,
                    },
                    "relative_area_pct": a.relative_area_pct,
                    "normalized_centroid": {
                        "cx": a.normalized_centroid[0],
                        "cy": a.normalized_centroid[1],
                    },
                    "aspect_ratio": a.aspect_ratio,
                    "proximity_factor": a.proximity_factor,
                    "severity_score": a.severity_score,
                    "severity_tier": a.severity_tier.value,
                    "formula_breakdown": a.formula_breakdown,
                }
                for idx, a in enumerate(self.assessments)
            ],
        }


class VisionGuardPipeline:
    """Master pipeline for road hazard detection, severity mapping, and reporting."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        detector: Optional[BaseHazardDetector] = None,
    ):
        self.config = config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(
            apply_clahe=self.config.apply_clahe,
            clahe_clip_limit=self.config.clahe_clip_limit,
            clahe_tile_grid_size=self.config.clahe_tile_grid_size,
            apply_denoising=self.config.apply_bilateral_denoising,
            bilateral_d=self.config.bilateral_d,
            bilateral_sigma_color=self.config.bilateral_sigma_color,
            bilateral_sigma_space=self.config.bilateral_sigma_space,
            apply_roi=self.config.apply_road_roi,
            horizon_fraction=self.config.road_horizon_fraction,
        )
        self.detector = detector or HybridHazardDetector(self.config)
        self.severity_engine = SeverityEngine(weights=self.config.severity_weights)
        self.analyzer = RoadAnalyzer(deduct_scaling=self.config.pci_deduct_scaling)
        self.visualizer = RoadVisualizer(config=self.config)

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_dir: Optional[Union[str, Path]] = None,
        save_visual: bool = True,
        save_json: bool = True,
        save_csv: bool = True,
    ) -> PipelineResult:
        """Run the complete VisionGuard pipeline on a single image."""
        start_time = time.perf_counter()

        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else "memory_array"
        image_name = Path(image_path_str).stem if isinstance(image_input, (str, Path)) else "frame"

        out_path = Path(output_dir) if output_dir else self.config.default_output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # Stage 1: Preprocessing
        preprocess_res = self.preprocessor.preprocess(image_input)

        # Stage 2: Hazard Detection
        det_res = self.detector.detect(preprocess_res.processed_image)

        # Stage 3: Severity Assessment
        assessments = self.severity_engine.assess_all(
            det_res.detections,
            image_shape=(preprocess_res.height, preprocess_res.width, preprocess_res.channels),
        )

        # Stage 4: Network-level Road Analysis & PCI
        analysis = self.analyzer.analyze(assessments)

        total_elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Stage 5: Visualization
        annotated_path = None
        if save_visual and self.config.generate_annotated_image:
            annotated_filename = out_path / f"{image_name}_annotated.jpg"
            self.visualizer.save_annotated(
                image=preprocess_res.original_image,
                analysis=analysis,
                output_path=annotated_filename,
                inference_time_ms=det_res.inference_time_ms,
            )
            annotated_path = str(annotated_filename)

        # Build pipeline result
        result = PipelineResult(
            image_path=image_path_str,
            image_name=image_name,
            preprocess_result=preprocess_res,
            detection_result=det_res,
            assessments=assessments,
            analysis=analysis,
            annotated_image_path=annotated_path,
            total_processing_time_ms=total_elapsed_ms,
        )

        # Stage 6: Structured Export
        if save_json and self.config.save_json:
            json_file = out_path / f"{image_name}_result.json"
            with open(json_file, "w", encoding="utf-8") as jf:
                json.dump(result.to_dict(), jf, indent=2)
            result.json_result_path = str(json_file)

        if save_csv and self.config.save_csv:
            self._export_csv(result, out_path)

        return result

    def _export_csv(self, result: PipelineResult, output_dir: Path) -> None:
        """Export detections CSV and summary CSV."""
        # 1. Per-hazard detections CSV
        detections_csv = output_dir / f"{result.image_name}_detections.csv"
        with open(detections_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "hazard_id",
                "category",
                "confidence",
                "severity_score",
                "severity_tier",
                "x1",
                "y1",
                "x2",
                "y2",
                "bbox_area_px",
                "relative_area_pct",
                "centroid_x",
                "centroid_y",
                "aspect_ratio",
                "proximity_factor",
            ])
            for idx, a in enumerate(result.assessments):
                writer.writerow([
                    idx + 1,
                    a.detection.category.value,
                    a.detection.confidence,
                    a.severity_score,
                    a.severity_tier.value,
                    a.detection.x1,
                    a.detection.y1,
                    a.detection.x2,
                    a.detection.y2,
                    a.bbox_area_px,
                    a.relative_area_pct,
                    a.normalized_centroid[0],
                    a.normalized_centroid[1],
                    a.aspect_ratio,
                    a.proximity_factor,
                ])

        # 2. Overall image summary CSV
        summary_csv = output_dir / f"{result.image_name}_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_name",
                "pci_score",
                "condition_state",
                "total_hazards",
                "critical_count",
                "high_count",
                "medium_count",
                "low_count",
                "max_severity",
                "mean_severity",
                "latency_ms",
            ])
            writer.writerow([
                result.image_name,
                result.analysis.pci_score,
                result.analysis.condition_state.value,
                result.analysis.total_hazards,
                result.analysis.severity_distribution.get("critical", 0),
                result.analysis.severity_distribution.get("high", 0),
                result.analysis.severity_distribution.get("medium", 0),
                result.analysis.severity_distribution.get("low", 0),
                result.analysis.max_severity_score,
                result.analysis.mean_severity_score,
                result.total_processing_time_ms,
            ])

    def process_batch(
        self,
        input_directory: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> List[PipelineResult]:
        """Process an entire directory of road images."""
        in_dir = Path(input_directory)
        if not in_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {in_dir.resolve()}")

        out_path = Path(output_dir) if output_dir else self.config.default_output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in extensions])
        results: List[PipelineResult] = []

        # Master batch summary CSV
        batch_summary_file = out_path / "batch_summary.csv"
        with open(batch_summary_file, "w", newline="", encoding="utf-8") as bf:
            writer = csv.writer(bf)
            writer.writerow([
                "image_file",
                "pci_score",
                "condition_state",
                "total_hazards",
                "critical_hazards",
                "potholes",
                "cracks",
                "debris",
                "processing_ms",
            ])

            for img_path in image_files:
                res = self.process_image(
                    img_path,
                    output_dir=out_path,
                    save_visual=self.config.generate_annotated_image,
                    save_json=self.config.save_json,
                    save_csv=self.config.save_csv,
                )
                results.append(res)

                potholes = res.analysis.category_counts.get("pothole", 0)
                cracks = (
                    res.analysis.category_counts.get("longitudinal_crack", 0)
                    + res.analysis.category_counts.get("transverse_crack", 0)
                    + res.analysis.category_counts.get("alligator_crack", 0)
                )
                debris = res.analysis.category_counts.get("road_debris", 0)

                writer.writerow([
                    img_path.name,
                    res.analysis.pci_score,
                    res.analysis.condition_state.value,
                    res.analysis.total_hazards,
                    res.analysis.severity_distribution.get("critical", 0),
                    potholes,
                    cracks,
                    debris,
                    res.total_processing_time_ms,
                ])

        return results
