import numpy as np
import os
import codecs
import argparse
import matplotlib.pyplot as plt

import nre.options as opts
from nre.utils.file_helper import check_file


def draw_pr(data_dir, models_to_plot, xlim, ylim):
    """
    Draw precision and recall curves.
    When finished, a pdf will be produced in data_dir.

    Args:
        data_dir: where the pr data are stored.
        models_to_plot: a list of model names
        xlim: recall (0.0~xlim)
        ylim: precision (ylim~1.0)
    """

    if len(models_to_plot) > 10:
        raise AssertionError('Only specified 10 colors')

    plt.clf()

    color = ['cornflowerblue', 'turquoise', 'darkorange', 'red', 'teal', 'blueviolet', 'black', 'green', 'slategray', 'brown']
    markers = ['<', 's', 'o', '*', 'x', '^', 'd', 'v', 'p', 'h', '1', '2', '3', '4', '>']

    for i in range(len(models_to_plot)):
        precision_file = os.path.join(data_dir, models_to_plot[i] + '_precision.npy')
        recall_file = os.path.join(data_dir, models_to_plot[i] + '_recall.npy')

        precision = np.load(precision_file)
        recall  = np.load(recall_file)
        plt.plot(recall, precision, color = color[i], marker=markers[i], markevery=120, lw=1, label = models_to_plot[i])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([ylim, 1.0])
    plt.xlim([0.0, xlim])
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, 'pr.pdf'))


def read_filenames(pr_data_dir):
    """
    Read 'curves.meta' file.
    In this file, each line contains a name of model to draw.

    Args:
        pr_data_dir: meta file' directory
    Return:
        A list of model names
    """

    meta_file = os.path.join(pr_data_dir, 'curves.meta')
    if not check_file(meta_file):
        raise AssertionError('Meta file is not existed')
    else:
        with codecs.open(meta_file, 'r', encoding='utf-8') as f:
            contents = f.readlines()
            file_names = [content.strip() for content in contents]

            return file_names


def main(opt):
    """
    Draw precision and recall curves

    Args:
        opt: plot options
    """

    models_to_draw = read_filenames(opt.pr_data_dir)
    draw_pr(opt.pr_data_dir, models_to_draw, opt.xlim, opt.ylim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='main.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    opts.plot_opts(parser)

    opt = parser.parse_args()
    main(opt)
