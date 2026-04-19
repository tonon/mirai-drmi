import numpy as np
import cv2


def lesion_area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def lesion_perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return float(sum(cv2.arcLength(c, True) for c in contours))


def area_ratio(mask: np.ndarray, retina_mask: np.ndarray = None) -> float:
    if retina_mask is None:
        return mask.sum() / mask.size
    return mask.sum() / retina_mask.sum()
