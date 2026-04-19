import numpy as np


def centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    return float(xs.mean()), float(ys.mean())


def distance_to_center(mask_shape, centroid):
    h, w = mask_shape
    cx, cy = w / 2, h / 2
    return np.sqrt((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2)


def lesion_location(distance, radius_ratio=0.15):
    if distance < radius_ratio:
        return "central"
    elif distance < radius_ratio * 2:
        return "paracentral"
    return "periferica"
