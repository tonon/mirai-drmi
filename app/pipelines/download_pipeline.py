from pathlib import Path
from app.ingestion.kaggle_downloader import KaggleDatasetDownloader
from app.core.config import settings


def run_download():
    downloader = KaggleDatasetDownloader(
        dataset="orvile/macular-degeneration-disease-dataset",
        output_dir=Path(settings.data_dir) / "raw" / "kaggle"
    )
    downloader.download()


if __name__ == "__main__":
    run_download()