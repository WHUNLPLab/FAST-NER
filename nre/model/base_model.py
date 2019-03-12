"""
    Overall model architecture for relation extraction
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model for all models. Each model has layers of embedding, encoder, selector and classifier.

    Args:
        pretrained_word_embeddings: for initializing word embeddings.
        gpu_utils: to convert a module into a DataParallel.
        opt: options.
    """

    def __init__(self, pretrained_word_embeddings, gpu_utils, opt):
        super(BaseModel, self).__init__()

        self.opt = opt

        self.embedding_layer = None
        self.encoder_layer = None
        self.selector_layer = None
        self.classifier_layer = None

    def forward(self, is_train, input_word, input_pos1, input_pos2,
        input_mask, input_type, input_scope, type_label, input_label=None, label_for_select=None):

        """
        Each time, batch_size bags are sent into network

        Args:
            is_train: for training or evaluation
            input_word: [total_number, num_steps]
            input_pos1: [total_number, num_steps]
            input_pos2: [total_number, num_steps]
            input_mask: [total_number, num_steps, 3]
            input_type: [total_number, num_steps]
            input_scope: [batch_size+1]
            type_label: [total_numer], entity type pair label
            input_label: [batch_size], lables of each bag, only for training
            label_for_select: [total_number], labels of each instance, only for training
        Return:
            when training, return loss and accuracy.detach()
            when evaluating,  return logits.detach()
        """

        raise NotImplementedError

    def model_output(self, is_train, logits, input_label):
        """
        Called by the forward function.

        Args:
            is_train: train or evaluate.
            logits: calculated by the selector.
            input_label: for calculating loss.
        """

        # For training, we need to calculate loss
        if is_train:
            loss, output = self.classifier_layer(logits, input_label)
            correct_predictions = torch.eq(input_label, output).to(torch.float)
            accuracy = torch.mean(correct_predictions)
            return loss, accuracy.detach()
        # For testing, we don't need to classify
        else:
            return logits.detach()
