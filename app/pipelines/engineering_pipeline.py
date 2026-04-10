from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

from app.core.config import settings
from app.preprocessing.image_engineering import FundusPreprocessor
from app.preprocessing.edge_detector import EdgeDetector


RAW_DIR = Path(settings.data_dir) / "raw" / "kaggle"
PROCESSED_DIR = Path(settings.data_dir) / "processed"
IMAGES_OUT = PROCESSED_DIR / "images"
EDGES_OUT = PROCESSED_DIR / "edges"


def ensure_dirs():
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    EDGES_OUT.mkdir(parents=True, exist_ok=True)


def run_engineering_pipeline():
    ensure_dirs()

    preprocessor = FundusPreprocessor(
        image_size=settings.image_size
    )

    edge_detector = EdgeDetector()

    records = []

    # Recursively find all images due to nested Kaggle extraction structure
    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        all_images.extend(RAW_DIR.rglob(ext))
        all_images.extend(RAW_DIR.rglob(ext.upper()))

    for img_path in tqdm(all_images, desc="Processing Images"):
        class_name = img_path.parent.name

        (IMAGES_OUT / class_name).mkdir(parents=True, exist_ok=True)
        (EDGES_OUT / class_name).mkdir(parents=True, exist_ok=True)

        # --- Load image ---
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # --- Engineering ---
        processed = preprocessor.process(image)
        edges = edge_detector.detect(processed)

        # --- Persist ---
        image_out = IMAGES_OUT / class_name / img_path.name
        edge_out = EDGES_OUT / class_name / img_path.name

        cv2.imwrite(str(image_out), processed)
        cv2.imwrite(str(edge_out), edges)

        # --- Metadata ---
        records.append({
            "filename": img_path.name,
            "class": class_name,
            "height": processed.shape[0],
            "width": processed.shape[1]
        })

    # --- Save metadata ---
    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)

    print("✅ Engenharia de imagens finalizada com sucesso")


if __name__ == "__main__":
    run_engineering_pipeline()
