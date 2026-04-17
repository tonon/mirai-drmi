from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from app.core.config import settings
from app.classification.dataset_cls import HybridClassificationDataset
from app.classification.model_cnn import LesionAwareCNN
from app.classification.trainer_cls import ClassifierTrainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    images_dir = Path(settings.data_dir) / "processed" / "images"
    masks_dir = Path(settings.data_dir) / "processed" / "segments_ensemble"

    dataset = HybridClassificationDataset(
        images_dir,
        masks_dir,
        class_map={"normal": 0, "amd": 1}
    )

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
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=settings.batch_size,
        shuffle=False
    )

    model = LesionAwareCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    trainer = ClassifierTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=settings.tensorboard_dir + "/classifier_cnn"
    )

    best_acc = 0.0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(settings.epochs):
        trainer.train_epoch(epoch)
        acc = trainer.val_epoch(epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/classifier_best.pth")
            print(f"✅ New best accuracy: {best_acc:.4f}")

    print("✅ Treinamento do classificador concluído")


if __name__ == "__main__":
    main()