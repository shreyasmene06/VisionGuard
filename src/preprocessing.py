"""Road Image Preprocessing Pipeline for VisionGuard.

Author: Shreyas Mene (CS480 Coursework)
Handles raw dashcam image ingestion, validation against corrupt frames,
LAB-space CLAHE illumination compensation, and bilateral texture filtering.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger("visionguard.preprocessing")


@dataclass
class PreprocessResult:
    """Preprocessed road frame with transformation audit metadata."""
    original_image: np.ndarray
    processed_image: np.ndarray
    height: int
    width: int
    channels: int
    clahe_applied: bool
    denoising_applied: bool
    roi_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)


class ImagePreprocessor:
    """Multi-stage preprocessor designed specifically for outdoor asphalt imagery."""

    def __init__(
        self,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        apply_denoising: bool = True,
        bilateral_d: int = 7,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        apply_roi: bool = False,
        horizon_fraction: float = 0.25,
    ):
        # CLAHE parameters: 2.5 clip limit balances shadow enhancement without blowing out sunlit asphalt
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        # Bilateral filter parameters: keeps sharp crack edges while smoothing coarse gravel texture
        self.apply_denoising = apply_denoising
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.apply_roi = apply_roi
        self.horizon_fraction = horizon_fraction

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )

    def validate_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Validate input path or array, ensuring non-empty dimensions and valid format.

        Raises:
            FileNotFoundError: If image file does not exist.
            ValueError: If file cannot be decoded or image array is empty/invalid.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path.resolve()}")

            # Read using OpenCV
            image = cv2.imread(str(path))
            if image is None or image.size == 0:
                raise ValueError(f"Failed to decode image from path: {path}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
            if image.size == 0:
                raise ValueError("Provided image numpy array is empty (size 0).")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if len(image.shape) == 2:
            # Grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # BGRA to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image dimensions: {image.shape}")

        h, w = image.shape[:2]
        if h < 32 or w < 32:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small for processing (minimum 32x32).")

        return image

    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """Adaptive histogram equalization in LAB color space to reveal subtle cracks."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def denoise_edge_preserving(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to smooth asphalt texture grain while retaining sharp edges."""
        return cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )

    def extract_road_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Mask out the sky/horizon region above the drivable roadway."""
        h, w = image.shape[:2]
        y_start = int(h * self.horizon_fraction)
        roi = image[y_start:h, 0:w]
        bbox = (0, y_start, w, h)
        return roi, bbox

    def preprocess(self, image_input: Union[str, Path, np.ndarray]) -> PreprocessResult:
        """Execute full preprocessing pipeline on the input image."""
        raw_image = self.validate_image(image_input)
        h, w, c = raw_image.shape

        processed = raw_image.copy()

        # Step 1: Contrast enhancement
        clahe_done = False
        if self.apply_clahe:
            processed = self.enhance_contrast_clahe(processed)
            clahe_done = True

        # Step 2: Denoising
        denoise_done = False
        if self.apply_denoising:
            processed = self.denoise_edge_preserving(processed)
            denoise_done = True

        # Step 3: Optional ROI
        roi_bbox = None
        if self.apply_roi:
            _, roi_bbox = self.extract_road_roi(processed)

        return PreprocessResult(
            original_image=raw_image,
            processed_image=processed,
            height=h,
            width=w,
            channels=c,
            clahe_applied=clahe_done,
            denoising_applied=denoise_done,
            roi_bbox=roi_bbox,
        )
