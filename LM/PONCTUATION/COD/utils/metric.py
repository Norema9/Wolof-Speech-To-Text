import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


class Metric2:
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
        # Convert logits to class predictions on the GPU
        punct_preds = torch.argmax(punct_preds, axis=-1)
        capit_preds = torch.argmax(capit_preds, axis=-1)

        # Transfer data to CPU
        self.punct_preds.append(punct_preds.cpu().numpy())
        self.capit_preds.append(capit_preds.cpu().numpy())
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


class Metric:
    def __init__(self, device='cpu'):
        self.device = device
        self.punct_correct = 0
        self.capit_correct = 0
        self.total_tokens = 0
        self.punct_preds = 0
        self.capit_preds = 0
        self.punct_labels = 0
        self.capit_labels = 0

    def reset(self):
        self.punct_correct = 0
        self.capit_correct = 0
        self.total_tokens = 0
        self.punct_preds = 0
        self.capit_preds = 0
        self.punct_labels = 0
        self.capit_labels = 0

    def update(self, punct_preds, capit_preds, punct_labels, capit_labels):
        self.punct_preds = punct_preds.to(self.device)
        self.capit_preds = capit_preds.to(self.device)
        self.punct_labels = punct_labels.to(self.device)
        self.capit_labels = capit_labels.to(self.device)
        
        self.punct_correct = (self.punct_preds == self.punct_labels).sum().item()
        self.capit_correct = (self.capit_preds == self.capit_labels).sum().item()
        self.total_tokens = self.punct_labels.numel()  # Number of elements in the tensor

    def compute(self):
        punct_accuracy = self.punct_correct / self.total_tokens
        capit_accuracy = self.capit_correct / self.total_tokens
        
        # Detect masked labels
        mask_punct = self.punct_labels != -100
        mask_capit = self.capit_labels != -100

        # Filter out masked labels
        punct_preds_filtered = self.punct_preds[mask_punct]
        capit_preds_filtered = self.capit_preds[mask_capit]
        punct_labels_filtered = self.punct_labels[mask_punct]
        capit_labels_filtered = self.capit_labels[mask_capit]
        
        # Calculate weighted precision, recall, and F1 scores
        punct_precision = precision_score(punct_labels_filtered, punct_preds_filtered, average='weighted')
        punct_recall = recall_score(punct_labels_filtered, punct_preds_filtered, average='weighted')
        punct_f1 = f1_score(punct_labels_filtered, punct_preds_filtered, average='weighted')
        
        capit_precision = precision_score(capit_labels_filtered, capit_preds_filtered, average='weighted')
        capit_recall = recall_score(capit_labels_filtered, capit_preds_filtered, average='weighted')
        capit_f1 = f1_score(capit_labels_filtered, capit_preds_filtered, average='weighted')
        
        return punct_accuracy, capit_accuracy, punct_precision, capit_precision, punct_recall, capit_recall, punct_f1, capit_f1

    def __call__(self, punct_logits, capit_logits, punct_labels, capit_labels):
        punct_preds = torch.argmax(punct_logits, axis=-1)
        capit_preds = torch.argmax(capit_logits, axis=-1)
        
        print(f"punct_preds : {punct_preds.shape}")
        print(f"capit_preds : {capit_preds.shape}")
        print(f"punct_labels : {punct_labels.shape}")
        print(f"capit_labels : {capit_labels.shape}")
        print("-"*40)
        
        self.update(punct_preds, capit_preds, punct_labels, capit_labels)
        return self.compute()
