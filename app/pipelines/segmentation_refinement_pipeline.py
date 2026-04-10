from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from app.core.config import settings


SEGMENTS_DIR = Path(settings.data_dir) / "processed" / "segments"
REFINED_DIR = Path(settings.data_dir) / "processed" / "segments_refined"


def refine_segmentation(mask: np.ndarray) -> np.ndarray:
    """
    Refinamento morfológico simples e robusto
    """
    kernel = np.ones((5, 5), np.uint8)

    # Fechamento para unir regiões
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Abertura para remover ruído pequeno
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Preenchimento
    _, filled = cv2.threshold(opened, 1, 255, cv2.THRESH_BINARY)

    return filled


def run_refinement_pipeline():
    REFINED_DIR.mkdir(parents=True, exist_ok=True)

    for class_dir in SEGMENTS_DIR.iterdir():
        refined_class_dir = REFINED_DIR / class_dir.name
        refined_class_dir.mkdir(exist_ok=True)

        for seg_path in tqdm(
            list(class_dir.glob("*.jpg")),
            desc=f"Refining {class_dir.name}"
        ):
            mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            refined = refine_segmentation(mask)
            cv2.imwrite(
                str(refined_class_dir / seg_path.name),
                refined
            )

    print("✅ Segmentações refinadas geradas com sucesso")


if __name__ == "__main__":
    run_refinement_pipeline()