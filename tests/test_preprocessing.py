"""Unit tests for Module 1: Road Image Preprocessing."""

import numpy as np
import pytest
from pathlib import Path

from src.preprocessing import ImagePreprocessor, PreprocessResult


@pytest.fixture
def sample_road_image() -> np.ndarray:
    """Create a synthetic 100x100 3-channel test image."""
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    # Add a dark patch representing a defect
    img[40:60, 40:60] = [30, 30, 30]
    return img


def test_validate_image_with_array(sample_road_image):
    preprocessor = ImagePreprocessor()
    validated = preprocessor.validate_image(sample_road_image)
    assert validated.shape == (100, 100, 3)
    assert validated.dtype == np.uint8


def test_validate_image_with_grayscale():
    preprocessor = ImagePreprocessor()
    gray = np.full((80, 80), 100, dtype=np.uint8)
    validated = preprocessor.validate_image(gray)
    assert validated.shape == (80, 80, 3)


def test_validate_image_file_not_found():
    preprocessor = ImagePreprocessor()
    with pytest.raises(FileNotFoundError):
        preprocessor.validate_image("non_existent_image_xyz.jpg")


def test_validate_image_too_small():
    preprocessor = ImagePreprocessor()
    tiny = np.full((16, 16, 3), 100, dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        preprocessor.validate_image(tiny)


def test_enhance_contrast_clahe(sample_road_image):
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.enhance_contrast_clahe(sample_road_image)
    assert enhanced.shape == sample_road_image.shape
    assert enhanced.dtype == np.uint8


def test_denoise_edge_preserving(sample_road_image):
    preprocessor = ImagePreprocessor()
    denoised = preprocessor.denoise_edge_preserving(sample_road_image)
    assert denoised.shape == sample_road_image.shape


def test_extract_road_roi(sample_road_image):
    preprocessor = ImagePreprocessor(apply_roi=True, horizon_fraction=0.30)
    roi, bbox = preprocessor.extract_road_roi(sample_road_image)
    assert roi.shape[0] == 70  # 100 - (100 * 0.3)
    assert roi.shape[1] == 100
    assert bbox == (0, 30, 100, 100)


def test_preprocess_pipeline(sample_road_image):
    preprocessor = ImagePreprocessor(apply_clahe=True, apply_denoising=True)
    res = preprocessor.preprocess(sample_road_image)

    assert isinstance(res, PreprocessResult)
    assert res.height == 100
    assert res.width == 100
    assert res.channels == 3
    assert res.clahe_applied is True
    assert res.denoising_applied is True
    assert res.processed_image.shape == (100, 100, 3)
