import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def forward(self, logits, targets):
        """
        logits: saída do modelo (N, 1, H, W)
        targets: máscara GT (N, 1, H, W)
        """

        probs = torch.sigmoid(logits)

        # gradiente horizontal / vertical (Sobel simples)
        grad_pred_x = probs[:, :, :, 1:] - probs[:, :, :, :-1]
        grad_pred_y = probs[:, :, 1:, :] - probs[:, :, :-1, :]

        grad_gt_x = targets[:, :, :, 1:] - targets[:, :, :, :-1]
        grad_gt_y = targets[:, :, 1:, :] - targets[:, :, :-1, :]

        loss = F.l1_loss(grad_pred_x, grad_gt_x) + F.l1_loss(grad_pred_y, grad_gt_y)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: saída bruta do modelo (N, 1, H, W)
        targets: máscara binária (N, 1, H, W)
        """
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_weight * torch.log(pt + 1e-8)
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(smooth)
        self.focal_loss = FocalLoss(alpha, gamma)
        self.boundary_loss = BoundaryLoss()

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        loss = dice + 0.5 * focal + 0.1 * boundary
        return loss
