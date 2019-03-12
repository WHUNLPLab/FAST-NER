"""
    Overall model architecture for relation extraction
"""

import torch
import torch.nn as nn

from nre.model.base_model import BaseModel
from nre.layers.embedding.embedding import Embedding
from nre.layers.encoder.pcnn import PcnnEncoder
from nre.layers.selector.type_attention import TypeAttention
from nre.layers.classifier.classifier import Classifier


class PCNNTATT(BaseModel):
    def __init__(self, pretrained_word_embeddings, gpu_utils, opt):
        super(PCNNTATT, self).__init__(pretrained_word_embeddings, gpu_utils, opt)

        self.beta = opt.beta

        self.embedding_layer = Embedding(pretrained_word_embeddings, opt)
        self.encoder_layer = gpu_utils.module_to_parallel(PcnnEncoder(opt))
        self.selector_layer = TypeAttention(opt.hidden_size*3, gpu_utils.use_gpu, opt)
        self.classifier_layer = Classifier(opt)

    def forward(self, is_train, input_word, input_pos1, input_pos2,
        input_mask, input_type, input_scope, type_label, input_label=None, label_for_select=None):

        embeddings = self.embedding_layer(input_word, input_pos1, input_pos2)
        encoder_out = self.encoder_layer(embeddings, input_mask)
        logits = self.selector_layer(encoder_out, input_scope, is_train, type_label, label_for_select)

        if is_train:
            relation_loss, relation_acc = self.model_output(is_train, logits[0], input_label)
            type_loss, _ = self.model_output(is_train, logits[1], type_label)

            return relation_loss + self.beta * type_loss, relation_acc
        else:
            return self.model_output(is_train, logits, input_label)
