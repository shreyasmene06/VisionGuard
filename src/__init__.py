"""VisionGuard: Computer Vision Based Road Surface Hazard Detection and Severity Mapping.

A modular, headless Computer Vision framework for automated road surface inspection,
hazard detection, geometric measurement, severity quantification, and pavement condition indexing.
"""

__version__ = "1.0.0"
__author__ = "Shreyas Mene"
__license__ = "MIT"

from src.config import (
    HazardCategory,
    SeverityTier,
    RoadConditionState,
    PipelineConfig,
)
from src.preprocessing import ImagePreprocessor, PreprocessResult
from src.detector import (
    Detection,
    DetectionResult,
    BaseHazardDetector,
    ClassicalHazardDetector,
    YOLOv8HazardDetector,
    HybridHazardDetector,
)
from src.severity import SeverityEngine, HazardSeverityAssessment
from src.analyzer import RoadAnalyzer, RoadConditionAnalysis
from src.visualization import RoadVisualizer
from src.metrics import EvaluationMetrics, compute_iou, evaluate_detections
from src.pipeline import VisionGuardPipeline, PipelineResult

__all__ = [
    "HazardCategory",
    "SeverityTier",
    "RoadConditionState",
    "PipelineConfig",
    "ImagePreprocessor",
    "PreprocessResult",
    "Detection",
    "DetectionResult",
    "BaseHazardDetector",
    "ClassicalHazardDetector",
    "YOLOv8HazardDetector",
    "HybridHazardDetector",
    "SeverityEngine",
    "HazardSeverityAssessment",
    "RoadAnalyzer",
    "RoadConditionAnalysis",
    "RoadVisualizer",
    "EvaluationMetrics",
    "compute_iou",
    "evaluate_detections",
    "VisionGuardPipeline",
    "PipelineResult",
]
