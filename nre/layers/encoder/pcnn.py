"""
    Piecewise convolutional networks as encoder
"""

import torch
import torch.nn as nn


class PcnnEncoder(nn.Module):
    def __init__(self, opt):
        super(PcnnEncoder, self).__init__()

        self.opt = opt

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(opt.dropout_keep)

        self.cnn = nn.Conv2d(
            in_channels=1,
            out_channels = opt.hidden_size,
            kernel_size=(opt.cnn_window_size, opt.word_vec_size + 2*opt.position_size),
            stride=(1,1),
            padding=(1, 0)
        )

    def forward(self, embeddings, masks):
        """
        Encode embeddings, using piece-wise convolutional networks.
        Here, we suppose the total number of sentences in bags is batch_size.

        Args:
            embeddings: [batch_size, num_step, embedding_size]
            masks: [batch_size, num_step]
        Return:
            hidden state of each sentence: [batch_size, hidden_size]
        """

        # embeddings = torch.unsqueeze(embeddings, dim=1)
        embeddings.unsqueeze_(dim=1)
        x = self.cnn(embeddings) # [batch_size, out_channel, num_step, 1]

        masks.unsqueeze_(dim=1) # [batch_size, 1, num_step, 3]
        x, _ = torch.max(masks + x, dim=2)
        x = x - 100
        x = x.view(-1, self.opt.hidden_size * 3)

        return self.dropout(self.activation(x))
