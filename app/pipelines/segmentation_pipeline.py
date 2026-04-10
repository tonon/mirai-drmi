from pathlib import Path
import cv2
from tqdm import tqdm

from app.core.config import settings
from app.preprocessing.edge_detector import EdgeDetector


IMAGES_DIR = Path(settings.data_dir) / "processed" / "images"
SEGMENTS_DIR = Path(settings.data_dir) / "processed" / "segments"


def run_segmentation_pipeline():
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    edge_detector = EdgeDetector()

    image_paths = list(IMAGES_DIR.rglob("*.jpg"))

    for img_path in tqdm(image_paths, desc="Generating segmentation assets"):
        class_name = img_path.parent.name

        out_dir = SEGMENTS_DIR / class_name
        out_dir.mkdir(exist_ok=True)

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        edges = edge_detector.detect(image)
        cv2.imwrite(str(out_dir / img_path.name), edges)

    print("✅ Ativos de segmentação gerados com sucesso")


if __name__ == "__main__":
    run_segmentation_pipeline()
