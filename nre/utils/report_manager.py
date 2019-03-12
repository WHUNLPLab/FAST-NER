"""
    Printer
"""

from __future__ import print_function

from nre.utils.logging import logger

class ReportManager(object):
    def __init__(self):
        pass

    @classmethod
    def print_args(self, opt):
        """
        Print all arguments

        Args:
            opt: options
        """

        logger.info('Following are all arguments')

        args = vars(opt)
        for k in args:
           msg = '{} = {}'.format(k, args[k]) 
           print(msg)

    @classmethod
    def print_model(self, model_parameters):
        """
        Print all parameters in model

        Args:
            model_parameters:
        """

        logger.info('Following are all model parameters')

        total_params = sum(p.numel() for p in model_parameters)
        total_trainable_params = sum(p.numel() for p in model_parameters if p.required_grad)
        logger.info('Total number of parameters is {}'.format(total_params))
        logger.info('Total number of trainable parameters is {}'.format(total_trainable_params))
