import torch.nn as nn 
import logging
import torch

class compute_crossentropyloss_manual(nn.Module):
    """
    y0 is the vector with shape (batch_size,C)
    x shape is the same (batch_size), whose entries are integers from 0 to C-1
    taken from this link: https://stackoverflow.com/questions/70202761/manually-computing-cross-entropy-loss-in-pytorch
    """
    def __init__(self, ignore_index=-100) -> None:
        super(compute_crossentropyloss_manual, self).__init__()
        self.ignore_index=ignore_index
    
    def forward(self, y0, x):
        loss = 0.
        #n_batch, n_class = y0.shape
        # print(n_class)
        cnt = 0             # <----- I added this
        for y1, x1 in zip(y0, x):
            class_index = int(x1.item())
            if class_index == self.ignore_index:
                continue
            loss = loss + torch.log(torch.exp(y1[class_index])/(torch.exp(y1).sum()))
            cnt += 1        # <----- I added this
        loss = - loss/cnt   # <---- I changed this from nbatch to 'cnt'
        return loss


class CrossEntropyLoss(compute_crossentropyloss_manual):
    """
    CrossEntropyLoss
    """

    def __init__(self,  weight=None, reduction='mean', ignore_index=-100):
        """
        Args:
            logits_ndim (int): number of dimensions (or rank) of the logits tensor
            weight (list): list of rescaling weight given to each class
            reduction (str): type of the reduction over the batch
        """
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.FloatTensor(weight)
            logging.info(f"Weighted Cross Entropy loss with weight {weight}")
        super(CrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        

    def forward(self, logits, labels, loss_mask=None):
        """
        Args:
            logits (float): output of the classifier
            labels (long): ground truth labels
            loss_mask (bool/float/int): tensor to specify the masking
        """
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            return super().forward(logits, torch.argmax(logits, dim=-1))
        
        loss = super().forward(logits_flatten, labels_flatten)
        return loss
    
    
    
