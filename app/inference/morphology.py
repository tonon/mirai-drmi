import numpy as np


def circularity(area: float, perimeter: float) -> float:
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def count_components(mask: np.ndarray) -> int:
    from cv2 import connectedComponents
    _, labels = connectedComponents(mask.astype("uint8"))
    return labels.max()
