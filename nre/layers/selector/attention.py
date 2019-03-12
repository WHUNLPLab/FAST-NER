"""
    Attention mechanisim proposed by Lin et al.(2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, relation_dim, opt):
        super(Attention, self).__init__()

        self.relation_matrix = nn.Embedding(opt.num_classes, relation_dim)
        self.bias = nn.Parameter(torch.Tensor(opt.num_classes))

        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.normal_(self.bias)

    def _logits(self, x):
        """
        Calculate logits of x
        """
        logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1)) + self.bias
        return logits

    def _train_logits(self, x, query):
        relation_query = self.relation_matrix(query)
        attention_logits = torch.sum(x * relation_query, 1, True)

        return attention_logits

    def _test_logits(self, x):
        attention_logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1))

        return attention_logits

    def forward(self, x, input_scope, is_train, query=None):
        """
        Calculate bag features for each bag

        Args:
            x: encoded hidden states of all sentences
            input_scope: scopes of all bags
            is_train: train or test
            query: label for select, only used in training
        Return:
            logits
        """
        if is_train:
            attention_logits = self._train_logits(x, query)

            tower_repre = []
            for i in range(len(input_scope) - 1):
                sen_matrix = x[input_scope[i] : input_scope[i+1]]
                attention_score = F.softmax(torch.transpose(attention_logits[input_scope[i] : input_scope[i+1]], 0, 1), 1)
                final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
                tower_repre.append(final_repre)

            stack_repre = torch.stack(tower_repre)
            logits = self._logits(stack_repre)
            return logits

        else:
            attention_logits = self._test_logits(x)

            tower_output = []
            for i in range(len(input_scope) - 1):
                sen_matrix = x[input_scope[i] : input_scope[i+1]]
                attention_score = F.softmax(torch.transpose(attention_logits[input_scope[i] : input_scope[i+1]], 0, 1), 1)
                final_repre = torch.matmul(attention_score, sen_matrix)
                logits = self._logits(final_repre) 
                tower_output.append(torch.diag(F.softmax(logits, 1)))

            stack_output = torch.stack(tower_output)
            return stack_output.data
