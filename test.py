"""
    Entry point of our models
"""

import argparse

import nre.options as opts
from nre.engine import Engine
from nre.utils.logging import init_logger
from nre.utils.report_manager import ReportManager


def main(opt):
    """
    Create an engine for training and testing.

    Args:
        opt: training options
    """

    # Initialize logger
    init_logger(opt.log_file)

    ReportManager.print_args(opt)

    if opt.pretrain_model == None or opt.pretrain_model == '':
        raise AssertionError('Must specify pretrained model for evaluation')

    model_engine = Engine(opt)
    model_engine.launch(opt)

    model_engine.eval(True, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='main.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    opts.model_opts(parser)
    opts.shared_opts(parser)

    opt = parser.parse_args()
    main(opt)
