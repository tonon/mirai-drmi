from pathlib import Path
import pandas as pd
from app.core.config import settings
from app.metrics.dataset_metrics import evaluate_segmentation_consistency


SEGMENTS_DIR = Path(settings.data_dir) / "processed" / "segments"
REFINED_DIR = Path(settings.data_dir) / "processed" / "segments_refined"
REPORTS_DIR = Path(settings.data_dir) / "reports"


def run_segmentation_stats():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    records = evaluate_segmentation_consistency(
        segments_dir=SEGMENTS_DIR,
        refined_dir=REFINED_DIR
    )

    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️ Nenhuma segmentação refinada encontrada. Execute a etapa de refinamento primeiro.")
        return

    df.to_csv(REPORTS_DIR / "segmentation_metrics.csv", index=False)

    print("✅ Estatísticas de segmentação calculadas")
    print(df.describe())


if __name__ == "__main__":
    run_segmentation_stats()