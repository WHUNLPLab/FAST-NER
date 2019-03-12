"""
    New attention mechanisim combining selective attention and entity type constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeAttention(nn.Module):
    def __init__(self, relation_dim, use_gpu, opt):
        """
        Relation matrx represents for the combination of type labels, relation labels and their embeddings.
        """

        super(TypeAttention, self).__init__()

        self.total_type_num = opt.type_num * opt.type_num
        self.use_gpu = use_gpu
        self.alpha = opt.alpha
        self.beta = opt.beta

        self.relation_matrix = nn.Embedding(opt.num_classes, relation_dim)
        self.type_matrix = nn.Embedding(self.total_type_num, relation_dim)
        self.bias = nn.Parameter(torch.Tensor(opt.num_classes))

        # Project layer
        self.type_liner = nn.Linear(relation_dim, self.total_type_num)

        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.xavier_uniform_(self.type_matrix.weight.data)
        nn.init.normal_(self.bias)

    def soft_type(self, x, type_label):
        """
        Args:
            x: hidden states of sentences
            type_label: type labels in text
        Return
            soft labels, same shape of type_label
        """

        type_logits = F.softmax(self.type_liner(x), dim=1)

        if self.use_gpu:
            one_hot = torch.zeros(len(type_label), self.total_type_num).cuda().scatter_(1, type_label.view(-1, 1), 1)
        else:
            one_hot = torch.zeros(len(type_label), self.total_type_num).scatter_(1, type_label.view(-1, 1), 1)

        if self.beta == 0.0:
            return type_label, type_logits
        else:
            max_scores, _ = torch.max(type_logits, dim=1)
            nscore = type_logits + self.alpha * max_scores.view(-1, 1) * one_hot
            _, soft_label = torch.max(nscore, dim=1)

            return soft_label, type_logits

    def _logits(self, x):
        """
        Calculate logits of x
        """

        logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1)) + self.bias
        return logits

    def _train_logits(self, x, relation_label, type_label):
        """
        Args:
            x: [total, relation_dim]
            relation_label: [total, 1]
            type_label: [total, 1]
        Return:
            [total, 1]
        """
        
        type_query = self.type_matrix(type_label)
        x = x + type_query
        relation_query = self.relation_matrix(relation_label)
        attention_logits = torch.sum(x * relation_query, 1, True)

        return attention_logits

    def _test_logits(self, x, type_label):
        """
        Args:
            x: [total, relation_dim]
            type_label: [total, 1]
        Return:
            [total, num_classes]
        """

        type_query = self.type_matrix(type_label)
        x = x + type_query
        attention_logits = torch.matmul(x, self.relation_matrix.weight.t())

        return attention_logits

    def forward(self, x, input_scope, is_train, type_label, query=None):
        """
        Calculate bag features for each bag

        Args:
            x: encoded hidden states of all sentences
            input_scope: scopes of all bags
            is_train: train or test
            type_label: label of entity type pair
            query: label for select, only used in training
        Return:
            logits
        """

        type_label, type_logits = self.soft_type(x, type_label)

        if is_train:
            attention_logits = self._train_logits(x, query, type_label)

            tower_repre = []
            for i in range(len(input_scope) - 1):
                sen_matrix = x[input_scope[i] : input_scope[i+1]] # [num_sentences, relation_dim]
                attention_score = F.softmax(torch.transpose(attention_logits[input_scope[i] : input_scope[i+1]], 0, 1), 1) # [num_sentences]
                final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix)) # [relation_dim]
                tower_repre.append(final_repre)

            stack_repre = torch.stack(tower_repre)
            logits = self._logits(stack_repre)
            return [logits, type_logits]

        else:
            attention_logits = self._test_logits(x, type_label)

            tower_output = []
            for i in range(len(input_scope) - 1):
                sen_matrix = x[input_scope[i] : input_scope[i+1]] # [total, relation_dim]
                attention_score = F.softmax(torch.transpose(attention_logits[input_scope[i] : input_scope[i+1]], 0, 1), 1) # [num_classes, total]
                final_repre = torch.matmul(attention_score, sen_matrix) # [num_classes, relation_dim]
                logits = self._logits(final_repre) # [num_classes, num_classes]
                tower_output.append(torch.diag(F.softmax(logits, 1)))

            stack_output = torch.stack(tower_output)
            return stack_output.detach()
