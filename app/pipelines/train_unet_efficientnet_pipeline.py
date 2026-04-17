from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from app.core.config import settings
from app.segmentation.dataset_seg import FundusSegmentationDataset
from app.segmentation.models.unet_efficientnet import UNetEfficientNetB3
from app.segmentation.trainer_unet import UNetTrainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    images_dir = Path(settings.data_dir) / "processed" / "images"
    masks_dir = Path(settings.data_dir) / "processed" / "segments_refined"

    dataset = FundusSegmentationDataset(images_dir, masks_dir)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = UNetEfficientNetB3(pretrained=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate
    )

    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=settings.tensorboard_dir + "/unet_efficientnet_b3"
    )

    best_dice = 0.0
    ckpt_dir = Path("models")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(settings.epochs):
        trainer.train_epoch(epoch)
        val_dice = trainer.val_epoch(epoch)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                model.state_dict(),
                ckpt_dir / "unet_efficientnet_b3_best.pth"
            )
            print(f"✅ New best Dice: {best_dice:.4f}")

    print("✅ Treinamento UNet + EfficientNet concluído")
    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()