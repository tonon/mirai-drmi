import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from app.metrics.segmentation_metrics import dice_coefficient, jaccard_index

class UNetTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader, device, log_dir):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCELoss()
        
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_dice = 0.0
        train_jaccard = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()

            preds = (outputs > 0.5).float()
            batch_dice = 0.0
            batch_jaccard = 0.0
            for pred, mask in zip(preds, masks):
                batch_dice += dice_coefficient(pred.cpu().detach().numpy(), mask.cpu().detach().numpy())
                batch_jaccard += jaccard_index(pred.cpu().detach().numpy(), mask.cpu().detach().numpy())
            batch_dice /= len(images)
            batch_jaccard /= len(images)
            
            train_dice += batch_dice
            train_jaccard += batch_jaccard

            pbar.set_postfix({"Loss": loss.item()})
            
        avg_loss = running_loss / len(self.train_loader)
        avg_dice = train_dice / len(self.train_loader)
        avg_jaccard = train_jaccard / len(self.train_loader)
        
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        self.writer.add_scalar("Dice/Train", avg_dice, epoch)
        self.writer.add_scalar("Jaccard/Train", avg_jaccard, epoch)
        print(f"Epoch {epoch} Train Avg Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, Jaccard: {avg_jaccard:.4f}")
        return avg_loss

    def val_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        val_dice = 0.0
        val_jaccard = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                batch_dice = 0.0
                batch_jaccard = 0.0
                for pred, mask in zip(preds, masks):
                    batch_dice += dice_coefficient(pred.cpu().numpy(), mask.cpu().numpy())
                    batch_jaccard += jaccard_index(pred.cpu().numpy(), mask.cpu().numpy())
                batch_dice /= len(images)
                batch_jaccard /= len(images)
                
                val_dice += batch_dice
                val_jaccard += batch_jaccard
                
                pbar.set_postfix({"Loss": loss.item(), "Dice": batch_dice})
                
        avg_loss = running_loss / len(self.val_loader)
        avg_dice = val_dice / len(self.val_loader)
        avg_jaccard = val_jaccard / len(self.val_loader)
        
        self.writer.add_scalar("Loss/Val", avg_loss, epoch)
        self.writer.add_scalar("Dice/Val", avg_dice, epoch)
        self.writer.add_scalar("Jaccard/Val", avg_jaccard, epoch)
        print(f"Epoch {epoch} Val Avg Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, Jaccard: {avg_jaccard:.4f}")
        return avg_dice
