"""
    Wrapper API for GPU useage
"""

import torch
import torch.nn as nn

from nre.utils.logging import logger


class GpuUtils(object):
    def __init__(self, opt):
        self._check_gpu_setting(opt)

        self.gpu_num = opt.gpu_num
        self.device_ids = opt.gpu_ranks[:self.gpu_num]

        self.use_gpu = True
        if not torch.cuda.is_available() or self.gpu_num == 0:
            self.use_gpu = False

        if opt.gpu_master != None and opt.gpu_master in self.device_ids:
            self.gpu_master = opt.gpu_master
        elif opt.gpu_master == None and len(self.device_ids) >= 1:
            self.gpu_master = self.device_ids[0]
        elif self.use_gpu:
            raise AssertionError('Invalid gpu_master value')
            

        if self.use_gpu:
            logger.info('Use GPU of {}'.format(str(self.device_ids)))
        else:
            logger.info('Use CPU only')

    def _check_gpu_setting(self, opt):
        """
        Check the gpu settings right or not

        Args:
            opt: options
        Return:
            if wrong, raise error
        """

        if opt.gpu_num < 0:
            raise AssertionError('The GPU number must be no less than 0')

        if (opt.gpu_num != 0) and (opt.batch_size % opt.gpu_num != 0):
            raise AssertionError('The GPU number must be divisible by batch size')

    def to_cuda(self, x, requires_grad=False):
        """
        Convert numpy data or Tensor to cuda Variable

        Args:
            x: the data to be converted
            required_grad: autograd or not
        Return:
            the resulted variable
        """

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
            x.requires_grad_(requires_grad=requires_grad)

        if not self.use_gpu:
            return x
        else:
            return x.cuda(self.gpu_master)

    def model_to_cuda(self, model):
        """
        Run model on GPU or CPU

        Args:
            model: nn.Module, the model accepts inputs
        """

        if not self.use_gpu:
            # Use CPU
            return model
        else:
            # multi-GPU
            # cuda_model = nn.DataParallel(model, device_ids=self.device_ids)
            return model.cuda(self.gpu_master)

    def module_to_parallel(self, module):
        """
        Convert a nn.Module to a DataParallel

        Args:
            module: instance of nn.Module
        Return:
            Wrapper of DataParallel
        """

        if not isinstance(module, nn.Module):
            raise AssertionError('Can only convert nn.Module to DataParallel')

        if self.use_gpu:
            return nn.DataParallel(module, device_ids=self.device_ids)
        else:
            return module

    def get_map_location(self):
        """
        Get a map location when load a model
        """

        if not self.use_gpu:
            return torch.device('cpu')
        else:
            return None
