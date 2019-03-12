"""
    Help building model according to options
"""

import torch.optim as optim

from nre.model.models.cnn_att import CNNATT
from nre.model.models.dcnn_att import DCNNATT
from nre.model.models.pdcnn_att import PDCNNATT
from nre.model.models.pdcnn_tatt import PDCNNTATT
from nre.model.models.dcnn_tatt import DCNNTATT
from nre.model.models.pcnn_att import PCNNATT
from nre.model.models.cnn_tatt import CNNTATT
from nre.model.models.pcnn_tatt import PCNNTATT

class ModelHelper(object):
    def __init__(self, opt, gpu_utils):
        self.opt = opt
        self.gpu_utils = gpu_utils

        self.model_types = {
            'CNN+ATT': CNNATT,
            'DCNN+ATT': DCNNATT,
            'PDCNN+ATT': PDCNNATT,
            'PDCNN+TATT': PDCNNTATT,
            'DCNN+TATT': DCNNTATT,
            'PCNN+ATT': PCNNATT,
            'CNN+TATT': CNNTATT,
            'PCNN+TATT': PCNNTATT,
        }

    def create_model(self, pretrained_word_embeddings, opt):
        """
        Create a corresponding model according to opt

        Args:
            pretrained_word_embeddings: word2vec
            opt: options

        Return:
            model: a model instance
        """

        model_class = self.model_types[opt.model_type]
        model = model_class(pretrained_word_embeddings, self.gpu_utils, opt)

        return model
