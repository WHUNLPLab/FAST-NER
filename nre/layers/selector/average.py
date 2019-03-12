"""
    Average attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Average(nn.Module):
    def __init__(self, opt):
        super(Average, self).__init__()

        self.relation_matrix = nn.Embedding(opt.num_classes, opt.hidden_size)
        self.bias = nn.Parameter(torch.Tensor(opt.num_classes))
        self.dropout = nn.Dropout(opt.dropout_keep)

        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.normal_(self.bias)

    def forward(self, x, input_scope, is_train, label_for_select=None):
        """
        Calculate average attention for each bag

        Args:
            x: encoded hidden states of each sentence
            input_scope: ranges of each bag in this batch
            is_train: for train or evaluation
        Return:
            bag features of each bag
        """

        tower_repre = []
        for i in range(len(input_scope) - 1):
            sen_matrix = x[input_scope[i] : input_scope[i+1]]
            final_repre = torch.mean(sen_matrix, 0)
            tower_repre.append(final_repre)

        stack_repre = torch.stack(tower_repre)
        logits = torch.matmul(stack_repre, torch.transpose(self.relation_matrix.weight, 0, 1)) + self.bias

        if is_train:
            return logits
        else:
            score = F.softmax(logits, 1)
            return score.detach()
