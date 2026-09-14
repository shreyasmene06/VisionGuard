"""Road Condition Aggregation and ASTM D6433 Pavement Condition Indexing.

Author: Shreyas Mene (CS480 Coursework)
Aggregates detected hazard severity assessments into a standard 0-100 PCI score
using an exponential deduct model and synthesizes municipal maintenance actions.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List
import numpy as np

from src.config import HazardCategory, RoadConditionState, SeverityTier
from src.severity import HazardSeverityAssessment

logger = logging.getLogger("visionguard.analyzer")


@dataclass
class RoadConditionAnalysis:
    """Comprehensive road condition assessment result for an image or road segment."""
    total_hazards: int
    category_counts: Dict[str, int]
    severity_distribution: Dict[str, int]
    max_severity_score: float
    mean_severity_score: float
    total_deduct_points: float
    pci_score: float                  # Pavement Condition Index (0.0 to 100.0)
    condition_state: RoadConditionState
    hazard_assessments: List[HazardSeverityAssessment]
    recommendations: List[str]


class RoadAnalyzer:
    """Aggregates hazard severity assessments and computes ASTM D6433-inspired condition metrics."""

    def __init__(self, deduct_scaling: float = 0.45):
        self.deduct_scaling = deduct_scaling

    def analyze(
        self,
        assessments: List[HazardSeverityAssessment],
    ) -> RoadConditionAnalysis:
        """Compute aggregated condition metrics, PCI score, and maintenance recommendations."""
        total_hazards = len(assessments)

        # Initialize category and severity counters
        category_counts: Dict[str, int] = {cat.value: 0 for cat in HazardCategory}
        severity_counts: Dict[str, int] = {tier.value: 0 for tier in SeverityTier}

        for assess in assessments:
            cat_name = assess.detection.category.value
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

            tier_name = assess.severity_tier.value
            severity_counts[tier_name] = severity_counts.get(tier_name, 0) + 1

        if total_hazards == 0:
            # Pristine road surface
            return RoadConditionAnalysis(
                total_hazards=0,
                category_counts=category_counts,
                severity_distribution=severity_counts,
                max_severity_score=0.0,
                mean_severity_score=0.0,
                total_deduct_points=0.0,
                pci_score=100.0,
                condition_state=RoadConditionState.GOOD,
                hazard_assessments=[],
                recommendations=[
                    "Pavement surface is in excellent condition with no visible structural hazards.",
                    "Continue standard routine biennial surveillance.",
                ],
            )

        scores = [a.severity_score for a in assessments]
        max_severity = round(float(np.max(scores)), 2)
        mean_severity = round(float(np.mean(scores)), 2)

        # ASTM D6433 Deduct Value Aggregation Formulation
        # Each defect contributes deduct points based on severity and area extent
        raw_deduct = 0.0
        for assess in assessments:
            # Severity contributes fundamentally to deduct value
            hazard_deduct = assess.severity_score * self.deduct_scaling
            # Additional penalty for high relative area
            if assess.relative_area_pct > 2.0:
                hazard_deduct *= 1.25
            raw_deduct += hazard_deduct

        # Diminishing returns formula to avoid negative PCI while ensuring severe roads drop rapidly
        # PCI = 100 - TotalDeduct (bounded between 0 and 100)
        corrected_deduct = 100.0 * (1.0 - np.exp(-raw_deduct / 75.0))
        pci_score = round(float(np.clip(100.0 - corrected_deduct, 0.0, 100.0)), 2)
        condition_state = RoadConditionState.from_pci(pci_score)

        # Generate rule-based municipal maintenance recommendations
        recommendations: List[str] = []
        crit_count = severity_counts.get(SeverityTier.CRITICAL.value, 0)
        high_count = severity_counts.get(SeverityTier.HIGH.value, 0)
        pothole_count = category_counts.get(HazardCategory.POTHOLE.value, 0)
        alligator_count = category_counts.get(HazardCategory.ALLIGATOR_CRACK.value, 0)
        debris_count = category_counts.get(HazardCategory.ROAD_DEBRIS.value, 0)

        if crit_count > 0:
            recommendations.append(
                f"PRIORITY ALERT: {crit_count} critical hazard(s) detected. Immediate dispatch for emergency road patching required to prevent vehicle rim and tire damage."
            )
        if debris_count > 0:
            recommendations.append(
                f"OBSTACLE HAZARD: {debris_count} road debris item(s) detected in traffic lane. Highway maintenance clearance crew dispatched."
            )
        if pothole_count > 0 and crit_count == 0:
            recommendations.append(
                f"POTHOLE REPAIR: {pothole_count} pothole depression(s) identified. Schedule cold-mix / hot-mix asphalt patching within 7 days."
            )
        if alligator_count > 0:
            recommendations.append(
                f"STRUCTURAL DEFECT: {alligator_count} alligator fatigue cracking region(s) observed, indicating sub-base deterioration. Full-depth reclamation or overlay recommended."
            )
        crack_count = category_counts.get(HazardCategory.LONGITUDINAL_CRACK.value, 0) + category_counts.get(HazardCategory.TRANSVERSE_CRACK.value, 0)
        if crack_count > 0:
            recommendations.append(
                f"PREVENTATIVE SEALING: {crack_count} longitudinal/transverse crack(s) identified. Hot-pour elastomeric sealant application advised before rainy season to stop water ingress."
            )

        if not recommendations:
            recommendations.append(
                f"Road condition rated {condition_state.value} (PCI: {pci_score}/100). Schedule standard cyclical maintenance."
            )

        return RoadConditionAnalysis(
            total_hazards=total_hazards,
            category_counts=category_counts,
            severity_distribution=severity_counts,
            max_severity_score=max_severity,
            mean_severity_score=mean_severity,
            total_deduct_points=round(float(corrected_deduct), 2),
            pci_score=pci_score,
            condition_state=condition_state,
            hazard_assessments=assessments,
            recommendations=recommendations,
        )
