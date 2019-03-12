"""
    Calculate accuracy, loss, precision, auc...

    For training and evaluation
"""

from __future__ import print_function
import sklearn.metrics
import tableprint as tp
import numpy as np
import os

from nre.utils.file_helper import ensure_folder
from nre.utils.logging import logger


class Statistics(object):
    def __init__(self):
        # Global best
        self.best_auc = 0
        self.f1_score = 0
        self.save_precision = None
        self.save_recall = None

        # How many times haven't the best updated
        self.stop_times = 0

        # Current model values
        self.current_auc = 0
        self.precision_100 = 0
        self.precision_200 = 0
        self.precision_300 = 0
        self.current_f1 = 0

    def _update_auc(self, auc_value, f1, precision, recall):
        """
        Update the best auc

        Args:
            auc_value: current value of auc
            f1: f1 score
            precision: a list, current precision
            recall: a list, current recall
        Return:
            if the current auc is the best
        """
        if auc_value > self.best_auc:
            self.best_auc = auc_value
            self.f1_score = f1
            self.save_precision = precision
            self.save_recall = recall

            return True
        else:

            return False

    def calculate_test_result(self, test_result, total_recall):
        """
        Calculate result for current model

        Args:
            test_result: a list containing predictions: [(entity1, entity2, rel), flag, pred[rel]]
                for memory consideration, we change it into [flag, pre[rel]]
            total_recall: the number of right predictions
        Return:
            if best_auc has been updated
        """

        # Sorted by pred[rel]
        # logger.info('Sort the test result, it may take several minutes...')
        sorted_test_result = sorted(test_result, key=lambda x: x[2])
        # logger.info('Sort the test result over')

        # Reference url: https://blog.csdn.net/zk_j1994/article/details/78478502
        pr_result_x = [] # recall
        pr_result_y = [] # precision
        correct = 0
        for i, item in enumerate(sorted_test_result[::-1]):
            if item[1] == 1: # flag == 1
                correct += 1
            pr_result_y.append(float(correct) / (i+1))
            pr_result_x.append(float(correct) / total_recall)

        pr_result_x = np.array(pr_result_x)
        pr_result_y = np.array(pr_result_y)
        auc = sklearn.metrics.auc(x=pr_result_x, y=pr_result_y)
        f1 = (2 * pr_result_x * pr_result_y / (pr_result_x + pr_result_y + 1e-20)).max()

        self.current_auc = auc
        self.current_f1 = f1
        self.precision_100 = pr_result_y[100]
        self.precision_200 = pr_result_y[200]
        self.precision_300 = pr_result_y[300]

        if_updated = self._update_auc(auc, f1, pr_result_y, pr_result_x)
        if if_updated:
            self.stop_times = 0
        else:
            self.stop_times += 1

        self._report_result()

        return if_updated

    def _report_result(self):
        """
        Report auc value, precisions
        """

        mean = (self.precision_100 + self.precision_200 + self.precision_300) / 3
        data = [[self.precision_100, self.precision_200, self.precision_300, mean, self.current_auc, self.current_f1, self.best_auc]]
        headers = ['P@100', 'P@200', 'P@300', 'Mean', 'AUC', 'Max F1', 'Best-AUC']

        tp.table(data, headers)

    def final_save(self, model_name, save_dir):
        """
        Print and save the best results

        Args:
            model_name:
            save_dir: directory for saving results
        """

        if (self.save_precision is not None) and (self.save_recall is not None):
            tp.banner('This is the best results!')
            mean = (self.save_precision[100] + self.save_precision[200] + self.save_precision[300]) / 3
            data = [[self.save_precision[100], self.save_precision[200], self.save_precision[300], mean, self.best_auc, self.f1_score]]
            headers = ['P@100', 'P@200', 'P@300', 'Mean', 'AUC', 'Max F1']
            tp.table(data, headers)

            ensure_folder(save_dir)
            np.save(os.path.join(save_dir, '{}_recall.npy'.format(model_name)), self.save_recall[:2000])
            np.save(os.path.join(save_dir, '{}_precision.npy'.format(model_name)), self.save_precision[:2000])
        else:
            logger.error('No model result to save')

    def stop_training_or_not(self, stop_after_n_eval):
        """
        Stop training or not

        Args:
            stop_after_n_eval: The number of evaluation times to stop training
        Return:
            True or False
        """

        return self.stop_times >= stop_after_n_eval
