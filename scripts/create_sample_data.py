#!/usr/bin/env python3
"""VisionGuard: Sample Data Generator.

Generates realistic road surface test images and corresponding ground-truth
annotations for automated evaluation and turnkey testing.
"""

import json
from pathlib import Path
import cv2
import numpy as np


def generate_asphalt_base(width: int = 800, height: int = 600) -> np.ndarray:
    """Generate a realistic asphalt pavement background with perspective and texture."""
    # Base asphalt color (dark charcoal gray)
    base_color = np.array([72, 74, 78], dtype=np.float32)
    img = np.tile(base_color, (height, width, 1))

    # Add high-frequency asphalt gravel noise
    noise = np.random.normal(0, 14, (height, width, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Road perspective lighting (darker in distance / top, slightly brighter near vehicle / bottom)
    gradient = np.linspace(0.82, 1.05, height).reshape(height, 1, 1)
    img = np.clip(img * gradient, 0, 255).astype(np.uint8)

    # Add faint road lane marking (dashed white/yellow line)
    lane_x = int(width * 0.15)
    for y in range(0, height, 80):
        cv2.line(img, (lane_x, y), (lane_x + 10, y + 45), (200, 200, 200), 4)

    return img


def add_pothole(img: np.ndarray, center: tuple, axes: tuple) -> tuple:
    """Render a realistic dark crater depression with textured rim and depth."""
    cx, cy = center
    ax, ay = axes

    # Inner dark cavity
    cv2.ellipse(img, (cx, cy), (ax, ay), 15, 0, 360, (28, 30, 34), -1)
    # Inner shadow depth
    cv2.ellipse(img, (cx + 4, cy + 4), (int(ax * 0.7), int(ay * 0.7)), 15, 0, 360, (18, 20, 22), -1)
    # Distressed broken asphalt perimeter
    cv2.ellipse(img, (cx, cy), (ax + 3, ay + 3), 15, 0, 360, (55, 58, 62), 3)

    # Calculate bounding box (x1, y1, x2, y2)
    x1 = max(0, cx - ax - 5)
    y1 = max(0, cy - ay - 5)
    x2 = min(img.shape[1] - 1, cx + ax + 5)
    y2 = min(img.shape[0] - 1, cy + ay + 5)
    return (x1, y1, x2, y2)


def add_longitudinal_crack(img: np.ndarray, start_pt: tuple, length: int) -> tuple:
    """Render a vertical meandering crack along the pavement."""
    x, y = start_pt
    pts = [(x, y)]
    curr_x, curr_y = x, y

    min_x, max_x = curr_x, curr_x
    min_y = curr_y

    for _ in range(length // 15):
        curr_y += 15
        curr_x += np.random.randint(-4, 5)
        pts.append((curr_x, curr_y))
        min_x = min(min_x, curr_x)
        max_x = max(max_x, curr_x)

    max_y = curr_y

    pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_arr], isClosed=False, color=(22, 24, 28), thickness=3)

    return (max(0, min_x - 6), max(0, min_y - 4), min(img.shape[1] - 1, max_x + 6), min(img.shape[0] - 1, max_y + 4))


def add_transverse_crack(img: np.ndarray, start_pt: tuple, length: int) -> tuple:
    """Render a horizontal meandering crack across pavement."""
    x, y = start_pt
    pts = [(x, y)]
    curr_x, curr_y = x, y

    min_x = curr_x
    min_y, max_y = curr_y, curr_y

    for _ in range(length // 15):
        curr_x += 15
        curr_y += np.random.randint(-3, 4)
        pts.append((curr_x, curr_y))
        min_y = min(min_y, curr_y)
        max_y = max(max_y, curr_y)

    max_x = curr_x

    pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_arr], isClosed=False, color=(22, 24, 28), thickness=3)

    return (max(0, min_x - 4), max(0, min_y - 6), min(img.shape[1] - 1, max_x + 4), min(img.shape[0] - 1, max_y + 6))


def add_alligator_crack(img: np.ndarray, center: tuple, size: int) -> tuple:
    """Render interconnected mesh cracking network."""
    cx, cy = center
    radius = size // 2
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(img.shape[1] - 1, cx + radius), min(img.shape[0] - 1, cy + radius)

    # Draw interlocking polygon crack cells
    step = 25
    for gx in range(x1 + 8, x2 - 8, step):
        for gy in range(y1 + 8, y2 - 8, step):
            pts = [
                (gx + np.random.randint(-3, 4), gy + np.random.randint(-3, 4)),
                (gx + step + np.random.randint(-3, 4), gy + np.random.randint(-3, 4)),
                (gx + step + np.random.randint(-3, 4), gy + step + np.random.randint(-3, 4)),
                (gx + np.random.randint(-3, 4), gy + step + np.random.randint(-3, 4)),
            ]
            cv2.polylines(img, [np.array(pts, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(20, 22, 26), thickness=3)

    return (x1, y1, x2, y2)


def add_road_debris(img: np.ndarray, center: tuple, size: int) -> tuple:
    """Render high-contrast road debris / dropped tire tread or wood block."""
    cx, cy = center
    w = size
    h = int(size * 0.5)
    x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
    x2, y2 = min(img.shape[1] - 1, cx + w // 2), min(img.shape[0] - 1, cy + h // 2)

    # Cast shadow
    cv2.rectangle(img, (x1 + 4, y1 + 4), (x2 + 6, y2 + 6), (30, 32, 35), -1)
    # Bright/contrasting debris object
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 170, 160), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (120, 110, 100), 2)

    return (x1, y1, x2, y2)


def main():
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    annotations_data = {
        "info": {
            "dataset": "VisionGuard Benchmark Dataset",
            "version": "1.0",
            "format": "Pixel coordinates [x1, y1, x2, y2]",
        },
        "categories": [
            "pothole",
            "longitudinal_crack",
            "transverse_crack",
            "alligator_crack",
            "damaged_pavement",
            "road_debris",
        ],
        "images": [],
    }

    # Image 1: pothole_01.jpg (Large pothole in near-field wheel path + longitudinal crack)
    img1 = generate_asphalt_base(800, 600)
    box1_1 = add_pothole(img1, center=(460, 420), axes=(65, 45))
    box1_2 = add_longitudinal_crack(img1, start_pt=(240, 200), length=160)
    cv2.imwrite(str(sample_dir / "pothole_01.jpg"), img1)
    annotations_data["images"].append({
        "file_name": "pothole_01.jpg",
        "width": 800,
        "height": 600,
        "annotations": [
            {"category": "pothole", "bbox": list(box1_1), "area_px": (box1_1[2]-box1_1[0])*(box1_1[3]-box1_1[1])},
            {"category": "longitudinal_crack", "bbox": list(box1_2), "area_px": (box1_2[2]-box1_2[0])*(box1_2[3]-box1_2[1])},
        ],
    })

    # Image 2: pothole_02.jpg (Double pothole in urban lane)
    img2 = generate_asphalt_base(800, 600)
    box2_1 = add_pothole(img2, center=(320, 360), axes=(50, 35))
    box2_2 = add_pothole(img2, center=(540, 460), axes=(55, 40))
    cv2.imwrite(str(sample_dir / "pothole_02.jpg"), img2)
    annotations_data["images"].append({
        "file_name": "pothole_02.jpg",
        "width": 800,
        "height": 600,
        "annotations": [
            {"category": "pothole", "bbox": list(box2_1), "area_px": (box2_1[2]-box2_1[0])*(box2_1[3]-box2_1[1])},
            {"category": "pothole", "bbox": list(box2_2), "area_px": (box2_2[2]-box2_2[0])*(box2_2[3]-box2_2[1])},
        ],
    })

    # Image 3: crack_01.jpg (Transverse thermal crack across lane)
    img3 = generate_asphalt_base(800, 600)
    box3_1 = add_transverse_crack(img3, start_pt=(260, 380), length=280)
    cv2.imwrite(str(sample_dir / "crack_01.jpg"), img3)
    annotations_data["images"].append({
        "file_name": "crack_01.jpg",
        "width": 800,
        "height": 600,
        "annotations": [
            {"category": "transverse_crack", "bbox": list(box3_1), "area_px": (box3_1[2]-box3_1[0])*(box3_1[3]-box3_1[1])},
        ],
    })

    # Image 4: crack_02.jpg (Extensive alligator mesh fatigue)
    img4 = generate_asphalt_base(800, 600)
    box4_1 = add_alligator_crack(img4, center=(420, 390), size=140)
    cv2.imwrite(str(sample_dir / "crack_02.jpg"), img4)
    annotations_data["images"].append({
        "file_name": "crack_02.jpg",
        "width": 800,
        "height": 600,
        "annotations": [
            {"category": "alligator_crack", "bbox": list(box4_1), "area_px": (box4_1[2]-box4_1[0])*(box4_1[3]-box4_1[1])},
        ],
    })

    # Image 5: debris_01.jpg (Road debris obstacle in driving line)
    img5 = generate_asphalt_base(800, 600)
    box5_1 = add_road_debris(img5, center=(480, 410), size=60)
    cv2.imwrite(str(sample_dir / "debris_01.jpg"), img5)
    annotations_data["images"].append({
        "file_name": "debris_01.jpg",
        "width": 800,
        "height": 600,
        "annotations": [
            {"category": "road_debris", "bbox": list(box5_1), "area_px": (box5_1[2]-box5_1[0])*(box5_1[3]-box5_1[1])},
        ],
    })

    # Image 6: clean_road_01.jpg (Pristine asphalt roadway)
    img6 = generate_asphalt_base(800, 600)
    cv2.imwrite(str(sample_dir / "clean_road_01.jpg"), img6)
    annotations_data["images"].append({
        "file_name": "clean_road_01.jpg",
        "width": 800,
        "height": 600,
        "annotations": [],
    })

    # Save annotations.json
    annot_file = sample_dir / "annotations.json"
    with open(annot_file, "w", encoding="utf-8") as f:
        json.dump(annotations_data, f, indent=2)

    print(f"[OK] Generated 6 realistic test road images in: {sample_dir}")
    print(f"[OK] Generated benchmark annotations in: {annot_file}")


if __name__ == "__main__":
    main()
