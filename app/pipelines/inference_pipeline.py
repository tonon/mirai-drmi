from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

from app.core.config import settings


def classify_clinical_type(area, circularity):
    """
    Classificação clínica baseada em descritores morfológicos.
    Ajustável conforme validação clínica.
    """
    if area > 20000:
        return "atrofia_geografica"
    if area < 2000 and circularity > 0.80:
        return "drusa_dura"
    return "drusa_mole"



# ============================================================
# Função de análise por componente (drusa individual)
# ============================================================


def analyze_by_component(binary_mask, min_area=20):
    
    if binary_mask.max() > 1:
        binary_mask = (binary_mask > 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(binary_mask)
    drusas = []

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        area = int(component.sum())

        if area < min_area:
            continue

        # morfologia por tamanho
        if area < 2000:
            morphology_class = "small_druse"
        elif area < 20000:
            morphology_class = "medium_druse"
        else:
            morphology_class = "large_region"

        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = contours[0].squeeze().tolist()
        perimeter = float(cv2.arcLength(contours[0], True))
        circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)

        ys, xs = np.where(component > 0)
        cx, cy = float(xs.mean()), float(ys.mean())

        clinical_type = classify_clinical_type(area, circularity)

        drusas.append({
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "centroid_x": cx,
            "centroid_y": cy,
            "morphology_class": morphology_class,  
            "clinical_type": clinical_type,         
            "contour": contour  
        })

    return drusas

# ============================================================
# 2️⃣ Pipeline principal de inferência
# ============================================================

def main():
    from app.segmentation.dataset_seg import FundusSegmentationDataset
    
    images_dir = Path(settings.data_dir) / "processed" / "images"
    masks_dir = Path(settings.data_dir) / "processed" / "segments_ensemble"

    output_dir = Path(settings.data_dir) / "processed" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FundusSegmentationDataset(images_dir, masks_dir)

    records = []

    for idx in tqdm(range(len(dataset)), desc="Inferência por drusa"):
        img, gt_mask = dataset[idx]
        img_path = dataset.image_paths[idx]
        rel_path = img_path.relative_to(images_dir)

        # máscara do ensemble já binária
        pred = masks_dir / rel_path.with_suffix(".png")
        pred_mask = cv2.imread(str(pred), cv2.IMREAD_GRAYSCALE)

        if pred_mask is None:
            continue

        binary_mask = (pred_mask > 0).astype(np.uint8)

        # ====================================================
        # 🔹 Métricas globais
        # ====================================================

        area_total = int(binary_mask.sum())
        area_ratio = area_total / binary_mask.size

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        perimeter_total = float(
            sum(cv2.arcLength(c, True) for c in contours)
        )

        circularity_total = (
            4 * np.pi * area_total / (perimeter_total**2 + 1e-6)
            if area_total > 0 else 0.0
        )

        # ====================================================
        # 🔹 Análise POR DRUSA (novo passo)
        # ====================================================

        drusas = analyze_by_component(binary_mask)
        num_drusas = len(drusas)

        areas = [d["area"] for d in drusas]
        perimeters = [d["perimeter"] for d in drusas]
        circularities = [d["circularity"] for d in drusas]

        record = {
            "image_id": img_path.stem,
            "area_total": area_total,
            "area_ratio": area_ratio,
            "perimeter_total": perimeter_total,
            "circularity_total": circularity_total,
            "num_drusas": num_drusas,
            "mean_area_per_drusa": float(np.mean(areas)) if areas else 0.0,
            "min_area_per_drusa": int(np.min(areas)) if areas else 0,
            "max_area_per_drusa": int(np.max(areas)) if areas else 0,
            # armazenamos lista completa em JSON-like
            "drusas": drusas,
        }

        records.append(record)

    # ========================================================
    # 3️⃣ Salvar CSV com resultados estruturados
    # ========================================================

    df = pd.DataFrame(records)

    csv_path = output_dir / "lesion_inference_by_drusa.csv"
    df.to_csv(csv_path, index=False)

    print("✅ Inferência por drusa concluída")
    print(f"CSV gerado em: {csv_path}")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()