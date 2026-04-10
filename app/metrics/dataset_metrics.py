from pathlib import Path
import cv2
import numpy as np
from app.metrics.segmentation_metrics import dice_coefficient, jaccard_index


def evaluate_segmentation_consistency(
    segments_dir: Path,
    refined_dir: Path
):
    """
    Compara segmentações base vs refinadas
    (ex: Canny vs morfologia / modelo futuro)
    """

    records = []

    for class_dir in segments_dir.iterdir():
        refined_class_dir = refined_dir / class_dir.name
        if not refined_class_dir.exists():
            continue

        for seg_path in class_dir.glob("*.jpg"):
            ref_path = refined_class_dir / seg_path.name
            if not ref_path.exists():
                continue

            seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

            seg = (seg > 0).astype(np.uint8)
            ref = (ref > 0).astype(np.uint8)

            dice = dice_coefficient(seg, ref)
            jaccard = jaccard_index(seg, ref)

            records.append({
                "class": class_dir.name,
                "filename": seg_path.name,
                "dice": dice,
                "jaccard": jaccard
            })

    return records