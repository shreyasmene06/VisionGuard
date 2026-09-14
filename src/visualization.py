"""Headless HUD Visualization and Telemetry Rendering.

Author: Shreyas Mene (CS480 Coursework)
Renders annotated visual inspection overlays directly to image files using OpenCV
headless operations (cv2.imwrite) without requiring a GUI window or display server.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.analyzer import RoadConditionAnalysis
from src.config import (
    CLASS_COLORS_BGR,
    SEVERITY_COLORS_BGR,
    HazardCategory,
    PipelineConfig,
    RoadConditionState,
    SeverityTier,
)
from src.severity import HazardSeverityAssessment

logger = logging.getLogger("visionguard.visualization")


class RoadVisualizer:
    """Headless canvas renderer for road inspection overlays and telemetry HUD."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def _draw_corner_brackets(
        self,
        canvas: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        length_ratio: float = 0.20,
    ) -> None:
        """Draw high-tech corner brackets around detection bounding box."""
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        line_len = max(8, int(min(bw, bh) * length_ratio))

        # Top-left
        cv2.line(canvas, (x1, y1), (x1 + line_len, y1), color, thickness)
        cv2.line(canvas, (x1, y1), (x1, y1 + line_len), color, thickness)
        # Top-right
        cv2.line(canvas, (x2, y1), (x2 - line_len, y1), color, thickness)
        cv2.line(canvas, (x2, y1), (x2, y1 + line_len), color, thickness)
        # Bottom-left
        cv2.line(canvas, (x1, y2), (x1 + line_len, y2), color, thickness)
        cv2.line(canvas, (x1, y2), (x1, y2 - line_len), color, thickness)
        # Bottom-right
        cv2.line(canvas, (x2, y2), (x2 - line_len, y2), color, thickness)
        cv2.line(canvas, (x2, y2), (x2, y2 - line_len), color, thickness)

    def draw_hud_header(
        self,
        image: np.ndarray,
        analysis: RoadConditionAnalysis,
        inference_time_ms: float = 0.0,
    ) -> np.ndarray:
        """Render a translucent top HUD banner summarizing road health and telemetry."""
        h, w = image.shape[:2]
        hud_height = max(70, int(h * 0.11))
        overlay = image.copy()

        # Dark gradient top bar
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (18, 22, 28), -1)
        # Accent line at bottom of HUD
        accent_color = (0, 200, 255)  # Gold/amber default
        if analysis.pci_score >= 85:
            accent_color = (76, 175, 80)    # Green
        elif analysis.pci_score < 40:
            accent_color = (34, 34, 220)    # Crimson
        cv2.line(overlay, (0, hud_height - 2), (w, hud_height - 2), accent_color, 2)

        # Alpha blend for translucency
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)

        # Title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_text = "VISIONGUARD | Road Surface Condition Telemetry"
        cv2.putText(image, title_text, (20, 28), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # PCI Score indicator block
        pci_str = f"PCI: {analysis.pci_score:.1f}/100 [{analysis.condition_state.value.upper()}]"
        pci_color = accent_color
        cv2.putText(image, pci_str, (20, hud_height - 18), font, 0.60, pci_color, 2, cv2.LINE_AA)

        # Stats on the right
        crit_count = analysis.severity_distribution.get(SeverityTier.CRITICAL.value, 0)
        high_count = analysis.severity_distribution.get(SeverityTier.HIGH.value, 0)
        med_count = analysis.severity_distribution.get(SeverityTier.MEDIUM.value, 0)
        low_count = analysis.severity_distribution.get(SeverityTier.LOW.value, 0)

        stats_text = (
            f"Hazards: {analysis.total_hazards}  |  "
            f"Crit: {crit_count}  High: {high_count}  Med: {med_count}  Low: {low_count}  |  "
            f"Latency: {inference_time_ms:.1f}ms"
        )
        (tw, th), _ = cv2.getTextSize(stats_text, font, 0.50, 1)
        cv2.putText(
            image,
            stats_text,
            (max(20, w - tw - 25), 32),
            font,
            0.50,
            (210, 220, 230),
            1,
            cv2.LINE_AA,
        )

        return image

    def annotate(
        self,
        image: np.ndarray,
        analysis: RoadConditionAnalysis,
        inference_time_ms: float = 0.0,
    ) -> np.ndarray:
        """Render complete annotation on the input image."""
        canvas = image.copy()
        h, w = canvas.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw detections and severity badges
        for assess in analysis.hazard_assessments:
            det = assess.detection
            x1, y1, x2, y2 = det.box

            # Class color & Severity color
            class_color = CLASS_COLORS_BGR.get(det.category, (0, 255, 255))
            severity_color = SEVERITY_COLORS_BGR.get(assess.severity_tier, (0, 255, 255))

            # Main bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), class_color, self.config.box_line_thickness)
            # Corner accents
            self._draw_corner_brackets(canvas, (x1, y1, x2, y2), severity_color, thickness=2)

            # Centroid crosshair
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.drawMarker(canvas, (cx, cy), severity_color, cv2.MARKER_CROSS, 8, 1)

            # Label pill text: e.g. "pothole | 85% | S:78.4 [CRITICAL]"
            label_text = (
                f"{det.category.value} {int(det.confidence * 100)}% | "
                f"S:{assess.severity_score:.1f} [{assess.severity_tier.value.upper()}]"
            )
            font_scale = 0.45
            (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

            # Position label above bounding box (or below if near top)
            label_y1 = max(0, y1 - txt_h - 8)
            label_y2 = label_y1 + txt_h + 8
            label_x2 = min(w, x1 + txt_w + 12)

            # Background pill
            cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), (24, 28, 36), -1)
            cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), severity_color, 1)

            # Text
            cv2.putText(
                canvas,
                label_text,
                (x1 + 6, label_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw top HUD header
        if self.config.draw_hud_banner:
            canvas = self.draw_hud_header(canvas, analysis, inference_time_ms)

        return canvas

    def save_annotated(
        self,
        image: np.ndarray,
        analysis: RoadConditionAnalysis,
        output_path: Union[str, Path],
        inference_time_ms: float = 0.0,
    ) -> Path:
        """Render annotations and save image directly to disk headlessly."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        annotated = self.annotate(image, analysis, inference_time_ms)
        success = cv2.imwrite(str(out_file), annotated)
        if not success:
            raise IOError(f"Failed to write annotated image to: {out_file}")

        return out_file
