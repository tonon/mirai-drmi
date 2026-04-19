from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

from app.core.config import settings
from app.inference.report import analyze_lesion


def main():
    # Diretórios
    masks_dir = Path(settings.data_dir) / "processed" / "segments_ensemble"
    output_dir = Path(settings.data_dir) / "processed" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # Percorre classes (Normal / AMD)
    for class_dir in masks_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for mask_path in tqdm(
            list(class_dir.glob("*.png")),
            desc=f"Processing {class_name}"
        ):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            mask = (mask > 0).astype("uint8")

            report = analyze_lesion(mask)

            record = {
                "image_id": mask_path.stem,
                "class": class_name,
                **report
            }

            records.append(record)

    df = pd.DataFrame(records)
    csv_path = output_dir / "lesion_inference.csv"
    df.to_csv(csv_path, index=False)

    print("✅ Pipeline de inferência concluído")
    print(f"CSV salvo em: {csv_path}")
    print(df.describe())


if __name__ == "__main__":
    main()