"""
    Use CNN for instances encoder
"""

import torch
import torch.nn as nn


class CnnEncoder(nn.Module):
    def __init__(self, opt):
        super(CnnEncoder, self).__init__()

        self.opt = opt

        self.cnn = nn.Conv2d(
            in_channels=1,
            out_channels=opt.hidden_size,
            kernel_size=(opt.cnn_window_size, opt.word_vec_size + 2*opt.position_size),
            stride=(1, 1),
            padding=(1, 0)
        )

        self.activation = nn.ReLU()

    def forward(self, embeddings):
        """
        Encode embeddings, including convolution and max-pooling operations

        Args:
            embeddings: [batch_size, num_step, embedding_size]
        Return:
            hidden state of each sentence: [batch_size, hidden_size]
        """

        embeddings = torch.unsqueeze(embeddings, dim=1)
        embeddings = self.cnn(embeddings)

        # Max-pooling
        x, _ = torch.max(embeddings, dim=2)
        x = x.view(-1, self.opt.hidden_size)

        return self.activation(x) 
