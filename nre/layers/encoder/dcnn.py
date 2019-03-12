"""
    Iterated Convolutional networks as encoder
"""

import torch
import torch.nn as nn


class DcnnEncoder(nn.Module):
    def __init__(self, opt):
        super(DcnnEncoder, self).__init__()

        self.opt = opt

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(opt.dropout_keep)

        # Base convolutional layer
        self.cnn = nn.Conv2d(
            in_channels=1,
            out_channels = opt.hidden_size,
            kernel_size=(opt.cnn_window_size, opt.word_vec_size + 2*opt.position_size),
            stride=(1,1),
            padding=(1, 0)
        )

        # Dilated convolutional layers
        self.dilated_cnn = nn.Conv2d(
            in_channels=opt.hidden_size,
            out_channels=opt.hidden_size,
            kernel_size=(opt.cnn_window_size, 1),
            stride=(1, 1),
            padding=(1, 0),
            dilation=2
        )

    def forward(self, embeddings):
        """
        Encode embeddings, using dilated convolutional networks.
        Here, we suppose the total number of sentences in bags is batch_size.

        Args:
            embeddings: [batch_size, num_step, embedding_size]
        Return:
            hidden state of each sentence: [batch_size, hidden_size]
        """

        # embeddings = torch.unsqueeze(embeddings, dim=1)
        embeddings.unsqueeze_(dim=1)

        dilated_input = self.cnn(embeddings)
        h, _ = torch.max(dilated_input, dim=2)

        block_out = self.dilated_cnn(dilated_input)
        block_out, _ = torch.max(block_out, dim=2)
        h += block_out

        x, _ = torch.max(h, dim=2)
        x = x.view(-1, self.opt.hidden_size)
        return self.dropout(self.activation(x))
