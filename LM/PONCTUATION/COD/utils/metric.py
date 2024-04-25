import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

class Metrics:
    def __init__(self, device='cpu'):
        self.device = device
        self.punct_correct = 0
        self.capit_correct = 0
        self.total_tokens = 0

    def reset(self):
        self.punct_correct = 0
        self.capit_correct = 0
        self.total_tokens = 0

    def update(self, punct_preds, capit_preds, punct_labels, capit_labels):
        punct_preds = punct_preds.to(self.device)
        capit_preds = capit_preds.to(self.device)
        punct_labels = punct_labels.to(self.device)
        capit_labels = capit_labels.to(self.device)
        
        self.punct_correct += (punct_preds == punct_labels).sum().item()
        self.capit_correct += (capit_preds == capit_labels).sum().item()
        self.total_tokens += punct_labels.numel()  # Number of elements in the tensor

    def compute(self):
        punct_accuracy = self.punct_correct / self.total_tokens
        capit_accuracy = self.capit_correct / self.total_tokens
        return punct_accuracy, capit_accuracy

    def __call__(self, punct_logits, capit_logits, punct_labels, capit_labels):
        punct_preds = torch.argmax(punct_logits, axis=-1)
        capit_preds = torch.argmax(capit_logits, axis=-1)
        self.update(punct_preds, capit_preds, punct_labels, capit_labels)
        return self.compute()


class Metric:
    def __init__(self, device='cpu'):
        self.device = device
        self.punct_preds = []
        self.capit_preds = []
        self.punct_labels = []
        self.capit_labels = []

    def reset(self):
        self.punct_preds = []
        self.capit_preds = []
        self.punct_labels = []
        self.capit_labels = []

    def update(self, punct_preds, capit_preds, punct_labels, capit_labels):
        punct_labels = punct_labels.to(self.device)
        capit_labels = capit_labels.to(self.device)
        
        # Convert logits to class predictions
        punct_preds = torch.argmax(punct_preds, axis=-1).cpu().numpy()
        capit_preds = torch.argmax(capit_preds, axis=-1).cpu().numpy()

        self.punct_preds.append(punct_preds)
        self.capit_preds.append(capit_preds)
        self.punct_labels.append(punct_labels.cpu().numpy())
        self.capit_labels.append(capit_labels.cpu().numpy())

    def compute(self):
        punct_orig = np.concatenate([np.concatenate(arr) for arr in self.punct_labels])
        punct_predicted = np.concatenate([np.concatenate(arr) for arr in self.punct_preds])
        capit_orig = np.concatenate([np.concatenate(arr) for arr in self.capit_labels])
        capit_predicted = np.concatenate([np.concatenate(arr) for arr in self.capit_preds])
        
        punct_pad_indices = np.where(punct_orig == -100)[0]
        capit_pad_indices = np.where(capit_orig == -100)[0]
        
        
        punct_orig = np.delete(punct_orig, punct_pad_indices)
        punct_predicted = np.delete(punct_predicted, punct_pad_indices)
        capit_orig = np.delete(capit_orig, capit_pad_indices)
        capit_predicted = np.delete(capit_predicted, capit_pad_indices)
        
        punct_accuracy = accuracy_score(punct_orig, punct_predicted)
        capit_accuracy = accuracy_score(capit_orig, capit_predicted)
        punct_precision = precision_score(punct_orig, punct_predicted, average='macro')
        capit_precision = precision_score(capit_orig, capit_predicted, average='macro')
        punct_recall = recall_score(punct_orig, punct_predicted, average='macro')
        capit_recall = recall_score(capit_orig, capit_predicted, average='macro')
        
        return punct_accuracy, capit_accuracy, punct_precision, capit_precision, punct_recall, capit_recall

    def __call__(self, punct_logits, capit_logits, punct_labels, capit_labels):
        self.update(punct_logits, capit_logits, punct_labels, capit_labels)
        return self.compute()
