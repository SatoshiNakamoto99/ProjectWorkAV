from sklearn.base import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

class Validator:
    def __init__(self, model, val_loader,criterion,  device):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() # per ora sommiamo le 5 loss

                all_labels.extend(labels.cpu().numpy())     # ??? 
                all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())  # ??? 

        val_loss = total_loss / len(self.val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_precision = precision_score(all_labels, all_predictions, average='macro')
        val_recall = recall_score(all_labels, all_predictions, average='macro')

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        return val_loss


# Example usage:
# validator = Validator(model, val_loader, device)
# validator.validate()
