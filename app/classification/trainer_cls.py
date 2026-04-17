import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import recall_score, roc_auc_score

class ClassifierTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader, device, log_dir):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_epoch(self, epoch):
        self.model.train()
        correct, total, loss_sum = 0, 0, 0

        for x, y in tqdm(self.train_loader, desc=f"Train {epoch}"):
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item()

        acc = correct / total
        avg_loss = loss_sum / len(self.train_loader)
        
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/Train", acc, epoch)

    def val_epoch(self, epoch):
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0
        
        all_labels = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc=f"Val {epoch}"):
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                probs = torch.softmax(logits, dim=1)[:, 1] # assuming class 1 is positive
                preds = logits.argmax(dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
                loss_sum += loss.item()
                
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        acc = correct / total
        avg_loss = loss_sum / len(self.val_loader)
        
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5 # Default if only one class is present in val set

        self.writer.add_scalar("Loss/Val", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/Val", acc, epoch)
        self.writer.add_scalar("Recall/Val", recall, epoch)
        self.writer.add_scalar("AUC/Val", auc, epoch)

        return acc