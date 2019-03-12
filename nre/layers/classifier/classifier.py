"""
    Used for relation classification
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Calculate the final loss using cross entropy loss function

        Args:
            logits: scores or probabilities corresponding to each relation
            labels: actual label for each bag
        Return:
            loss: batch loass
            acc: accuracy
            output: predictions
        """

        loss = self.loss_fn(logits, labels)
        _, output = torch.max(logits, dim=1)
        
        return loss, output.data
