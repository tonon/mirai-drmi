from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm

from app.core.config import settings
from app.segmentation.dataset_seg import FundusSegmentationDataset
from app.segmentation.models.unet_resnet import UNetResNet34
from app.segmentation.models.unet_efficientnet import UNetEfficientNetB3
from app.metrics.segmentation_metrics import dice_coefficient, jaccard_index


def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    images_dir = Path(settings.data_dir) / "processed" / "images"
    masks_dir = Path(settings.data_dir) / "processed" / "segments_refined"

    output_dir = Path(settings.data_dir) / "processed" / "segments_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FundusSegmentationDataset(images_dir, masks_dir)

    model_resnet = load_model(
        UNetResNet34(pretrained=False),
        Path("models/unet_resnet34_best.pth"),
        device
    )

    model_effnet = load_model(
        UNetEfficientNetB3(pretrained=False),
        Path("models/unet_efficientnet_b3_best.pth"),
        device
    )

    dices = []
    jaccards = []

    for idx in tqdm(range(len(dataset)), desc="Running ensemble"):
        img, gt_mask = dataset[idx]
        img_path = dataset.image_paths[idx]
        rel_path = img_path.relative_to(images_dir)

        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_resnet = model_resnet(img)
            pred_effnet = model_effnet(img)

            ensemble_prob = (pred_resnet + pred_effnet) / 2.0
            ensemble_mask = (ensemble_prob > 0.5).float()

        gt = gt_mask.squeeze().cpu().numpy()
        pred = ensemble_mask.squeeze().cpu().numpy()

        dice = dice_coefficient(pred, gt)
        jaccard = jaccard_index(pred, gt)

        dices.append(dice)
        jaccards.append(jaccard)

        out_path = output_dir / rel_path
        out_path = out_path.with_suffix('.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), (pred * 255).astype(np.uint8))

    mean_dice = np.mean(dices)
    mean_jaccard = np.mean(jaccards)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=settings.tensorboard_dir + "/ensemble")
    writer.add_scalar("Test/Dice", mean_dice, 0)
    writer.add_scalar("Test/Jaccard", mean_jaccard, 0)
    writer.close()

    print("✅ Ensemble concluído")
    print(f"Dice médio: {mean_dice:.4f}")
    print(f"Jaccard médio: {mean_jaccard:.4f}")


if __name__ == "__main__":
    main()