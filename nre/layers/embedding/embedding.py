"""
    Map sentences into embeddings, includes word embeddings, position embeddings, and type embeddings
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, pretrained_word_embedding, opt):
        super(Embedding, self).__init__()

        self.word_embedding = nn.Embedding(opt.vocabulary_size+2, opt.word_vec_size) 
        self.pos1_embedding = nn.Embedding(opt.position_num+1, opt.position_size, padding_idx=opt.position_num)
        self.pos2_embedding = nn.Embedding(opt.position_num+1, opt.position_size, padding_idx=opt.position_num)

        self.init_embeddings(pretrained_word_embedding)

    def init_embeddings(self, pretrained_word_embedding):
        """
        Initialize word embeddings and position embeddings
        """

        # Initialize word embeddings
        self.word_embedding.weight.data.copy_(pretrained_word_embedding)

        # Initialize position embeddings
        nn.init.xavier_uniform_(self.pos1_embedding.weight.data)
        if self.pos1_embedding.padding_idx is not None:
            self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)

        nn.init.xavier_uniform_(self.pos2_embedding.weight.data)
        if self.pos2_embedding.padding_idx is not None:
            self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)

    def forward(self, input_word, input_pos1, input_pos2):
        """
        Look up embeddings for inputs

        Args:
            input_word: input word for model
            input_pos1: input pos1 for model
            input_pos2: input pos2 for model
        Return:
            Concatenated embedding including word embeddings and position embeddings
        """

        word = self.word_embedding(input_word)
        pos1 = self.pos1_embedding(input_pos1)
        pos2 = self.pos2_embedding(input_pos2)
        embedding = torch.cat((word, pos1, pos2), dim=2)

        return embedding
