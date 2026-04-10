from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict


class HybridFundusDataset(Dataset):
    """
    Dataset híbrido para fundoscopia:
    - RGB (3 canais)
    - Edge map (1 canal)
    """

    def __init__(
        self,
        images_dir: Path,
        edges_dir: Path,
        class_map: Dict[str, int],
        transform=None
    ):
        self.images_dir = images_dir
        self.edges_dir = edges_dir
        self.class_map = class_map
        self.transform = transform

        self.samples = self._index_dataset()

    def _index_dataset(self):
        samples = []

        for class_name, label in self.class_map.items():
            image_class_dir = self.images_dir / class_name
            edge_class_dir = self.edges_dir / class_name

            for img_path in image_class_dir.glob("*.jpg"):
                edge_path = edge_class_dir / img_path.name
                if edge_path.exists():
                    samples.append((img_path, edge_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, edge_path, label = self.samples[idx]

        # --- Load image ---
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)

        # --- Normalize ---
        image = image.astype("float32") / 255.0
        edge = edge.astype("float32") / 255.0

        # --- To tensor ---
        image = torch.from_numpy(image).permute(2, 0, 1)   # [3, H, W]
        edge = torch.from_numpy(edge).unsqueeze(0)         # [1, H, W]

        x = torch.cat([image, edge], dim=0)                # [4, H, W]
        y = torch.tensor(label).long()

        if self.transform:
            x = self.transform(x)

        return x, y