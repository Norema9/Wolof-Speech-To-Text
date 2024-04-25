import torch
import torch.nn as nn

class AggregatorLoss(nn.Module):
    def __init__(self, num_losses = 2):
        super().__init__()
        self.num_losses = num_losses
        self.weights = nn.Parameter(torch.ones(num_losses))

    def forward(self, *losses):
        weighted_losses = [weight * loss for weight, loss in zip(self.weights, losses)]
        total_loss = sum(weighted_losses)
        return total_loss