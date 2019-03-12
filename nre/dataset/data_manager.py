"""
    Get DataLoader for training set and testing set
"""

import os
import torch
from torch.utils.data import DataLoader

from nre.dataset.training_set import TrainningSet
from nre.dataset.testing_set import TestningSet
from nre.dataset.dataset import Batch


class DataManager(object):
    def __init__(self):
        self.training_set = None
        self.testing_set = None

    def self_collate_fn(self, batch):
        """
        Merges a list of bags to form a mini-batch

        Args:
            batch: A list containing bags    
        Return:
            new_batch: merged result
        """ 

        word_batch = torch.cat([torch.tensor(bag.word, dtype=torch.long) for bag in batch], dim=0)
        pos1_batch = torch.cat([torch.tensor(bag.pos1, dtype=torch.long) for bag in batch], dim=0)
        pos2_batch = torch.cat([torch.tensor(bag.pos2, dtype=torch.long) for bag in batch], dim=0)
        type_batch = torch.cat([torch.tensor(bag.type, dtype=torch.long) for bag in batch], dim=0)
        mask_batch = torch.cat([torch.tensor(bag.mask, dtype=torch.float) for bag in batch], dim=0)
        lens_batch = torch.cat([torch.tensor(bag.lens, dtype=torch.int) for bag in batch], dim=0)
        label = torch.tensor([bag.label for bag in batch], dtype=torch.long)
        label_for_select = torch.cat([torch.tensor(bag.label_for_select, dtype=torch.long) for bag in batch], dim=0)
        type_label_batch = torch.cat([torch.tensor(bag.type_label, dtype=torch.long) for bag in batch], dim=0)
        bag_ids = [bag.bag_id for bag in batch]

        scope_batch = [0]
        total_instance = 0
        for bag in batch:
            total_instance += len(bag.word)
            scope_batch.append(total_instance)
        scope_batch = torch.tensor(scope_batch, dtype=torch.int)

        new_batch = Batch(
            word = word_batch,
            pos1 = pos1_batch,
            pos2 = pos2_batch,
            type = type_batch,
            mask = mask_batch,
            lens = lens_batch,
            label = label,
            label_for_select = label_for_select,
            scope = scope_batch,
            bag_ids = bag_ids,
            type_label = type_label_batch
        )

        return new_batch


    def get_train_data_loader(self, data_dir, batch_size):
        """
        Get a iteratable data loader for training set

        Args:
            data_dir: directory saving training data
            batch_size: batch size
        Return:
            torch DataLoader
        """

        self.training_set = TrainningSet(data_dir)
        data_loader = DataLoader(dataset=self.training_set, batch_size=batch_size, shuffle=True, collate_fn=self.self_collate_fn, drop_last=True)

        return data_loader


    def get_test_data_loader(self, data_dir, batch_size):
        """
        Get a iteratable data loader for testing set

        Args:
            data_dir: directory saving training data
            batch_size: batch size
        Return:
            torch DataLoader
        """

        self.testing_set = TestningSet(data_dir)
        data_loader = DataLoader(dataset=self.testing_set, batch_size=batch_size, shuffle=False, collate_fn=self.self_collate_fn, drop_last=True)

        return data_loader

    def exists_test_triple(self, triple):
        """
        If triple is contained in instance_triple of testing set

        Args:
            triple: (entity1, entity2, relation) tuple
        Return:
            True or False
        """

        return self.testing_set.exists_triple(triple)

    def find_entity_pair(self, index):
        """
        Find entity pair in testing set according to index

        Args:
            index: list index 
        """

        return self.testing_set.find_entity_pair(index)

    def load_pretrained_word2vec(self, data_dir):
        """
        Load word_embedding from data_dir
        """

        data_path = os.path.join(data_dir, 'vec.pt')
        word2vec = torch.load(data_path)
        return torch.from_numpy(word2vec)
