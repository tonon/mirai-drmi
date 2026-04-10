import numpy as np


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Dice = 2 |A ∩ B| / (|A| + |B|)
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    size_a = mask_a.sum()
    size_b = mask_b.sum()

    if size_a + size_b == 0:
        return 1.0

    return 2.0 * intersection / (size_a + size_b)


def jaccard_index(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Jaccard = |A ∩ B| / |A ∪ B|
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 1.0

    return intersection / union