from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset


from app.core.config import settings

class HybridClassificationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, class_map: dict):
        self.samples = []
        self.class_map = class_map

        for class_name, label in class_map.items():
            img_class_dir = images_dir / class_name
            mask_class_dir = masks_dir / class_name

            for img_path in img_class_dir.glob("*.jpg"):
                mask_path = mask_class_dir / img_path.with_suffix('.png').name
                if mask_path.exists():
                    self.samples.append((img_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (settings.image_size, settings.image_size))
        image = image.astype("float32") / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (settings.image_size, settings.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype("float32")

        image = torch.from_numpy(image).permute(2, 0, 1)   # [3,H,W]
        mask = torch.from_numpy(mask).unsqueeze(0)         # [1,H,W]

        x = torch.cat([image, mask], dim=0)                # [4,H,W]
        y = torch.tensor(label).long()

        return x, y