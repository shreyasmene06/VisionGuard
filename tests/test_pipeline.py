"""Integration tests for VisionGuard end-to-end pipeline and exports."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.config import PipelineConfig, RoadConditionState
from src.pipeline import VisionGuardPipeline


@pytest.fixture
def temp_road_image(tmp_path: Path) -> Path:
    """Create a realistic temporary road image with a dark defect patch."""
    img = np.full((400, 400, 3), 90, dtype=np.uint8)
    # Add dark pothole crater
    cv2.circle(img, (200, 260), 35, (25, 25, 25), -1)
    file_path = tmp_path / "test_pothole.jpg"
    cv2.imwrite(str(file_path), img)
    return file_path


@pytest.fixture
def pristine_road_image(tmp_path: Path) -> Path:
    """Create a completely uniform, pristine pavement image."""
    img = np.full((400, 400, 3), 100, dtype=np.uint8)
    file_path = tmp_path / "test_pristine.jpg"
    cv2.imwrite(str(file_path), img)
    return file_path


def test_pipeline_single_image(temp_road_image: Path, tmp_path: Path):
    out_dir = tmp_path / "outputs"
    config = PipelineConfig(
        detection_backend="classical",
        save_json=True,
        save_csv=True,
        generate_annotated_image=True,
    )
    pipeline = VisionGuardPipeline(config=config)

    res = pipeline.process_image(temp_road_image, output_dir=out_dir)

    assert res.image_name == "test_pothole"
    assert res.preprocess_result.height == 400
    assert res.preprocess_result.width == 400
    assert 0.0 <= res.analysis.pci_score <= 100.0
    assert res.total_processing_time_ms > 0.0

    # Verify files created
    assert res.annotated_image_path is not None
    assert Path(res.annotated_image_path).is_file()

    assert res.json_result_path is not None
    assert Path(res.json_result_path).is_file()

    # Verify JSON structure
    with open(res.json_result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "pavement_condition_index" in data
    assert "hazard_summary" in data
    assert "hazards" in data

    # Verify CSV files
    det_csv = out_dir / "test_pothole_detections.csv"
    sum_csv = out_dir / "test_pothole_summary.csv"
    assert det_csv.is_file()
    assert sum_csv.is_file()


def test_pipeline_pristine_road(pristine_road_image: Path, tmp_path: Path):
    out_dir = tmp_path / "outputs_pristine"
    config = PipelineConfig(detection_backend="classical")
    pipeline = VisionGuardPipeline(config=config)

    res = pipeline.process_image(pristine_road_image, output_dir=out_dir)

    assert res.analysis.total_hazards == 0
    assert res.analysis.pci_score == 100.0
    assert res.analysis.condition_state == RoadConditionState.GOOD


def test_pipeline_batch_processing(tmp_path: Path):
    input_dir = tmp_path / "batch_in"
    input_dir.mkdir()
    out_dir = tmp_path / "batch_out"

    # Create 2 test images
    for i in range(2):
        img = np.full((200, 200, 3), 80 + i * 20, dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"road_{i}.jpg"), img)

    config = PipelineConfig(detection_backend="classical")
    pipeline = VisionGuardPipeline(config=config)

    results = pipeline.process_batch(input_dir, output_dir=out_dir)
    assert len(results) == 2
    assert (out_dir / "batch_summary.csv").is_file()
