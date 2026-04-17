import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class FundusSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        self.image_paths = sorted(list(self.images_dir.rglob("*.jpg")) + list(self.images_dir.rglob("*.png")))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        rel_path = img_path.relative_to(self.images_dir)
        mask_path = self.masks_dir / rel_path
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Erro ao ler imagem: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not mask_path.exists():
            mask_path = mask_path.with_suffix('.png')
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Erro ao ler mascara: {mask_path}")
            
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        image = image.astype("float32") / 255.0
        mask = mask.astype("float32") / 255.0
        
        mask = (mask > 0.5).astype("float32")
        
        image = torch.from_numpy(image).permute(2, 0, 1) # [3, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)       # [1, H, W]
        
        if self.transform:
            image = self.transform(image)
            
        return image, mask
