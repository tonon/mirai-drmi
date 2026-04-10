import os
from pathlib import Path

class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "/workspace/data"))
    image_size: int = int(os.getenv("IMAGE_SIZE", 256))

settings = Settings()
