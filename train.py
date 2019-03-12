"""
    Entry point of our models
"""

import argparse
import signal
import sys

import nre.options as opts
from nre.engine import Engine
from nre.utils.logging import init_logger, logger
from nre.utils.report_manager import ReportManager


class SignalListener(object):
    """
    Process Ctrl+C and Ctrl+Z
    """

    def __init__(self, engine, opt):
        self.model_engine = engine
        self.opt = opt

    def register_handler(self, signum):
        signal.signal(signum, self.signal_hander)

    def signal_hander(self, signum, frame):
        """
        Process the signal
        """

        logger.info('Recieve signal of {}'.format(signum))
        self.model_engine.stats.final_save(self.opt.model_type, self.opt.res_dir)
        sys.exit(0)


def main(opt):
    """
    Create an engine for training and testing.

    Args:
        opt: training options
    """

    # Initialize logger
    init_logger(opt.log_file)

    ReportManager.print_args(opt)

    model_engine = Engine(opt)
    model_engine.launch(opt)

    # Handle Ctrl+C
    signal_listener = SignalListener(model_engine, opt)
    signal_listener.register_handler(signal.SIGINT)

    model_engine.train(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='main.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    opts.model_opts(parser)
    opts.shared_opts(parser)

    opt = parser.parse_args()
    main(opt)

