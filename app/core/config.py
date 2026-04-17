import os
from pathlib import Path

class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "/workspace/data"))
    image_size: int = int(os.getenv("IMAGE_SIZE", 256))
    batch_size: int = int(os.getenv("BATCH_SIZE", 4))
    learning_rate: float = float(os.getenv("LEARNING_RATE", 1e-4))
    epochs: int = int(os.getenv("EPOCHS", 20))
    tensorboard_dir: str = os.getenv("TENSORBOARD_DIR", "/workspace/runs")

settings = Settings()
