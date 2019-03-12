"""
    Data loader for training set
"""

import torch
import os
import numpy as np

from nre.dataset.dataset import DataSet
from nre.dataset.dataset import Batch

class TrainningSet(DataSet):
    def __init__(self, data_dir):
        DataSet.__init__(self, data_dir)

        # Load training data from peocessed files
        self.train_instance_triple = self.load_data('train_instance_triple')
        self.train_instance_scope = self.load_data('train_instance_scope')
        self.train_len = self.load_data('train_len')
        self.train_word = self.load_data('train_word')
        self.train_pos1 = self.load_data('train_pos1')
        self.train_pos2 = self.load_data('train_pos2')
        self.train_type = self.load_data('train_type')
        self.train_mask = self.load_data('train_mask')
        self.train_label = self.load_data('train_label')
        self.train_type_label = self.load_data('train_type_label')

    def __getitem__(self, index):
        """
        Return a bag with 'index' of index
        """

        instances_scope = self.train_instance_scope[index] 

        # A list containing all index of instance in the bag
        instances_index = list(range(instances_scope[0], instances_scope[1] + 1))

        bag_label = self.train_label[instances_scope[0]]
        # bag_id = (entity1, entity2, relation)
        bag_id = self.train_instance_triple[index]

        sen_word = self.train_word[instances_index, :]
        sen_pos1 = self.train_pos1[instances_index, :]
        sen_pos2 = self.train_pos2[instances_index, :]
        sen_type = self.train_type[instances_index, :]
        sen_mask = self.train_mask[instances_index, :]
        sen_lens = self.train_len[instances_index]
        label_for_select = self.train_label[instances_index]
        sen_type_label = self.train_type_label[instances_index]

        single_batch = Batch(
            word = sen_word,
            pos1 = sen_pos1,
            pos2 = sen_pos2,
            type = sen_type,
            mask = sen_mask,
            lens = sen_lens,
            label = bag_label,
            label_for_select = label_for_select,
            bag_id = bag_id,
            type_label = sen_type_label
        )

        return single_batch

    def __len__(self):
        return len(self.train_instance_scope)
