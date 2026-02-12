import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-class classification
    Parameters:
        alpha (float or Tensor): per-class weight (e.g., entropy-inspired weights), must be shape [num_classes]
        gamma (float): focusing parameter for modulating factor (1-p)
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes] (logits)
        targets: [batch_size] (int64 labels)
        """
        # Move alpha to device if set
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
        logpt = F.log_softmax(inputs, dim=1)  # [batch, num_classes]
        pt = torch.exp(logpt)                 # Probabilities

        # Select the log probability and pt for the target class only
        targets = targets.view(-1, 1)
        logpt = logpt.gather(1, targets).view(-1)
        pt = pt.gather(1, targets).view(-1)

        # Alpha weighting
        if self.alpha is not None:
            at = self.alpha[targets.squeeze()]
            loss = -at * (1 - pt) ** self.gamma * logpt
        else:
            loss = -(1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
