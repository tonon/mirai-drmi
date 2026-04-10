import subprocess
from pathlib import Path


class KaggleDatasetDownloader:
    def __init__(
        self,
        dataset: str,
        output_dir: Path
    ):
        self.dataset = dataset
        self.output_dir = output_dir

    def download(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            self.dataset,
            "-p",
            str(self.output_dir),
            "--unzip"
        ]

        subprocess.run(command, check=True)
        return self.output_dir

