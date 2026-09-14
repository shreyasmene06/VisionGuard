"""Configuration definitions, enumerations, and default parameters for VisionGuard."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple


class HazardCategory(str, Enum):
    """Supported road surface hazard categories."""
    POTHOLE = "pothole"
    LONGITUDINAL_CRACK = "longitudinal_crack"
    TRANSVERSE_CRACK = "transverse_crack"
    ALLIGATOR_CRACK = "alligator_crack"
    DAMAGED_PAVEMENT = "damaged_pavement"
    ROAD_DEBRIS = "road_debris"

    @classmethod
    def from_string(cls, name: str) -> "HazardCategory":
        cleaned = name.strip().lower().replace(" ", "_").replace("-", "_")
        for member in cls:
            if member.value == cleaned:
                return member
        # Fallback heuristics
        if "pot" in cleaned or "hole" in cleaned:
            return cls.POTHOLE
        if "alligator" in cleaned or "mesh" in cleaned:
            return cls.ALLIGATOR_CRACK
        if "long" in cleaned:
            return cls.LONGITUDINAL_CRACK
        if "trans" in cleaned:
            return cls.TRANSVERSE_CRACK
        if "crack" in cleaned:
            return cls.LONGITUDINAL_CRACK
        if "debris" in cleaned or "obstacle" in cleaned:
            return cls.ROAD_DEBRIS
        return cls.DAMAGED_PAVEMENT


class SeverityTier(str, Enum):
    """Categorical severity tiers."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoadConditionState(str, Enum):
    """Pavement Condition Index (PCI) rating scale based on ASTM D6433."""
    GOOD = "Good"                # 85 - 100
    SATISFACTORY = "Satisfactory" # 70 - 84
    FAIR = "Fair"                # 55 - 69
    POOR = "Poor"                # 40 - 54
    VERY_POOR = "Very Poor"      # 25 - 39
    SERIOUS = "Serious"          # 10 - 24
    FAILED = "Failed"            # 0 - 9

    @classmethod
    def from_pci(cls, pci: float) -> "RoadConditionState":
        if pci >= 85.0:
            return cls.GOOD
        elif pci >= 70.0:
            return cls.SATISFACTORY
        elif pci >= 55.0:
            return cls.FAIR
        elif pci >= 40.0:
            return cls.POOR
        elif pci >= 25.0:
            return cls.VERY_POOR
        elif pci >= 10.0:
            return cls.SERIOUS
        else:
            return cls.FAILED


# Base hazard impact score (0 - 100) reflecting intrinsic physical danger
BASE_CLASS_WEIGHTS: Dict[HazardCategory, float] = {
    HazardCategory.POTHOLE: 45.0,           # High vehicular impact, blowout hazard
    HazardCategory.ALLIGATOR_CRACK: 35.0,   # Severe structural subgrade fatigue
    HazardCategory.ROAD_DEBRIS: 40.0,       # Sudden obstacle, swerve hazard
    HazardCategory.DAMAGED_PAVEMENT: 25.0,  # Surface ravelling, rutting
    HazardCategory.TRANSVERSE_CRACK: 20.0,  # Thermal expansion defect
    HazardCategory.LONGITUDINAL_CRACK: 18.0,# Pavement joint defect
}


# Visual color mapping (BGR format for OpenCV)
CLASS_COLORS_BGR: Dict[HazardCategory, Tuple[int, int, int]] = {
    HazardCategory.POTHOLE: (34, 34, 220),          # Bright Crimson
    HazardCategory.LONGITUDINAL_CRACK: (0, 200, 255),# Yellow-Orange
    HazardCategory.TRANSVERSE_CRACK: (0, 140, 255),  # Deep Orange
    HazardCategory.ALLIGATOR_CRACK: (180, 50, 240), # Magenta / Violet
    HazardCategory.DAMAGED_PAVEMENT: (0, 215, 255), # Amber
    HazardCategory.ROAD_DEBRIS: (255, 120, 0),      # Blue-Cyan
}

SEVERITY_COLORS_BGR: Dict[SeverityTier, Tuple[int, int, int]] = {
    SeverityTier.LOW: (76, 175, 80),        # Soft Green
    SeverityTier.MEDIUM: (0, 215, 255),     # Amber Gold
    SeverityTier.HIGH: (0, 120, 255),       # Deep Orange
    SeverityTier.CRITICAL: (34, 34, 220),   # Crimson Red
}


@dataclass
class SeverityWeights:
    """Mathematical weights for multi-factor severity scoring."""
    weight_area: float = 0.35         # Influence of relative bounding box area
    weight_class: float = 0.35        # Influence of intrinsic hazard class danger
    weight_proximity: float = 0.20    # Influence of vertical camera proximity (near vs far)
    weight_confidence: float = 0.10   # Influence of detection confidence certainty


@dataclass
class PipelineConfig:
    """Master configuration parameters for the VisionGuard pipeline."""
    # Preprocessing
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    apply_bilateral_denoising: bool = True
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    apply_road_roi: bool = False
    road_horizon_fraction: float = 0.25

    # Detection
    confidence_threshold: float = 0.30
    nms_iou_threshold: float = 0.45
    detection_backend: str = "auto"  # 'auto', 'yolov8', 'classical'
    model_weights_path: str = "weights/yolov8n_road.pt"
    device: str = "cpu"  # 'cpu' or 'cuda'

    # Severity & Analysis
    severity_weights: SeverityWeights = field(default_factory=SeverityWeights)
    pci_deduct_scaling: float = 0.45

    # Visualization
    generate_annotated_image: bool = True
    draw_hud_banner: bool = True
    draw_severity_badge: bool = True
    box_line_thickness: int = 2
    font_scale: float = 0.55

    # Storage
    save_json: bool = True
    save_csv: bool = True
    default_output_dir: Path = Path("outputs")
