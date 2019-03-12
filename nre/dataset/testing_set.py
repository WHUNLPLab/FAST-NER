"""
    Data loader for testing set
"""

import torch
import os
import numpy as np

from nre.dataset.dataset import DataSet
from nre.dataset.dataset import Batch

class TestningSet(DataSet):
    def __init__(self, data_dir):
        DataSet.__init__(self, data_dir)

        # Load testing data from processed files
        self.test_instance_entity = self.load_data('test_instance_entity')
        self.test_instance_entity_no_bag = self.load_data('test_instance_entity_no_bag')
        instance_triple = self.load_data('test_instance_triple')
        self.test_instance_triple = set()
        for item in instance_triple:
            tup = (item[0], item[1], int(item[2]))
            self.test_instance_triple.add(tup)
        self.test_instance_scope = self.load_data('test_instance_scope')
        self.test_len = self.load_data('test_len')
        self.test_word = self.load_data('test_word')
        self.test_pos1 = self.load_data('test_pos1')
        self.test_pos2 = self.load_data('test_pos2')
        self.test_type = self.load_data('test_type')
        self.test_mask = self.load_data('test_mask')
        self.test_label = self.load_data('test_label')
        self.test_type_label = self.load_data('test_type_label')

    def __getitem__(self, index):
        """
        Return a bag with 'index' of index
        """

        instances_scope = self.test_instance_scope[index] 

        # A list containing all index of instance in the bag
        instances_index = list(range(instances_scope[0], instances_scope[1] + 1))

        bag_label = self.test_label[instances_scope[0]]
        # bag_id = (entity1, entity2)
        bag_id = self.test_instance_entity[index]

        sen_word = self.test_word[instances_index, :]
        sen_pos1 = self.test_pos1[instances_index, :]
        sen_pos2 = self.test_pos2[instances_index, :]
        sen_type = self.test_type[instances_index, :]
        sen_mask = self.test_mask[instances_index, :]
        sen_lens = self.test_len[instances_index]
        label_for_select = self.test_label[instances_index]
        sen_type_label = self.test_type_label[instances_index]

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
        return len(self.test_instance_scope)

    def exists_triple(self, triple):
        """
        If triple is contained in instance triple

        Args:
            triple: (entity1, entity2, relation)
        """

        return triple in self.test_instance_triple

    def find_entity_pair(self, index):
        """
        Find an entity pair according to index
        """

        return self.test_instance_entity[index]
