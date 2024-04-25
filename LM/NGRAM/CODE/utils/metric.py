import torch

class Metric:
    def __init__(self):
        self.punct_correct = 0
        self.capit_correct = 0
        self.total_tokens = 0

    def reset(self):
        self.punct_correct = 0
        self.capit_correct = 0
        self.total_tokens = 0

    def update(self, punct_preds, capit_preds, punct_labels, capit_labels):
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
