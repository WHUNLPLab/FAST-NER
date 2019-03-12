"""
    Preprocess the text data and convert them into numpy format
"""

from __future__ import print_function

import argparse
import torch
import codecs
import os
import numpy as np
from tqdm import tqdm

from nre.utils.logging import init_logger, logger
from nre.utils.file_helper import ensure_folder, remove_file_or_folder
import nre.options as opts
from nre.utils.logging import init_logger


def word2id_fn(word2vec_file):
    logger.info('Reading word embedding data...')

    # A map, key is word, value is its id
    word2id = {}

    with codecs.open(word2vec_file, 'r', encoding='utf-8') as f:
        total, size = f.readline().strip().split()[:2]
        total = (int)(total)
        size = (int)(size)
        vec = np.ones((total, size), dtype=np.float32)

        for i in range(total):
            content = f.readline()
            content = content.strip().split()
            word2id[content[0]] = len(word2id)
            for j in range(size):
                vec[i][j] = np.float32(content[j+1])

        word2id['UNK'] = len(word2id)
        word2id['BLANK'] = len(word2id)

        unk_embedding = np.random.normal(size=size, loc=0, scale=0.05).reshape(1, size)
        blank_embedding = np.zeros(size, dtype=np.float32).reshape(1, size)
        vec = np.concatenate((vec, unk_embedding), axis=0)
        vec = np.concatenate((vec, blank_embedding), axis=0)
    
    return word2id, vec


def relation2id_fn(relation_map_file):
    """
    Read relation to id
    """

    logger.info('Reading relation to id')
    relation2id = {}

    with codecs.open(relation_map_file,'r', encoding='utf-8') as f:
        while True:
            content = f.readline()
            if content == '':
                break
            content = content.strip().split()
            relation2id[content[0]] = int(content[1])

    return relation2id


def type2id_fn(type_map_file, type_num):
    """
    Read type to id
    """

    logger.info("Reading type to id")
    type2id = {}
    pair2id = {}

    with codecs.open(type_map_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        type_lines = lines[: type_num+1]
        for content in type_lines:
            content = content.strip().split()
            type2id[content[0]] = int(content[1])

        pair_lines = lines[type_num+2: type_num+2+type_num*type_num]
        for content in pair_lines:
            content = content.strip().split()
            pair2id[content[0]] = int(content[1])

    return type2id, pair2id


def gather_bag(data_file, relation2id, tmp_bag_file):
    """
    Gather instances into bags
    bag_id: (en1_id, en2_id, relation), Hence a bag only contains one relation
    total: total number of sentences, not bags

    Args:
        data_file: train.txt or test.txt
        relation2id: dictionary
        tmp_bag_file: file to store results
    Return:
        None
    """

    logger.info('Gathering bags for {}'.format(data_file))

    bags = {}
    total = 0

    with codecs.open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            total += 1

            content = line.strip().split()
            en1_id = content[0]
            en2_id = content[1]
            relation = content[6]
            if relation in relation2id:
                relation = relation2id[relation]
            else:
                relation = relation2id['NA']
            bag_id = '{}#{}#{}'.format(en1_id, en2_id, relation)

            if bag_id not in bags:
                bags[bag_id] = []
            bags[bag_id].append(line)

    with codecs.open(tmp_bag_file, 'w', encoding='utf-8') as r:
        r.write('%d\n'%(total))
        for bag_id in bags:
            for instance in bags[bag_id]:
                r.write(instance)


def read_train_files(file, word2id, relation2id, type2id, pair2id, opt):
    """
    Split a sentence into word, position, type..., for training

    Args:
        file: train data file
        word2id: a dictionary
        relation2id: a dictionary
        type2id: a dictionary
        pair2id: a dictionary
        opt: preprocess options
    Return:
        instance_triple: a list of bag id
        instance_scope: a list of scops for each bag
        sen_len: real length of each sentence
        sen_label: relation label of each sentence
        sen_word: word ids of each sentence
        sen_pos1: left position ids of each sentence
        sen_pos2: right position ids of each sentence
        sen_type: type ids of each sentence
        sen_mask: mask of each sentence
    """

    logger.info('Reading train data ...')

    f = codecs.open(file, 'r', encoding='utf-8')

    total = (int)(f.readline().strip())

    fixlen = opt.num_steps

    sen_word = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos1 = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos2 = np.zeros((total, fixlen), dtype=np.int32)
    sen_mask = np.zeros((total, fixlen, 3), dtype=np.float32)
    sen_type = np.zeros((total, fixlen), dtype=np.int32)
    # Actual length of sentence
    sen_len = np.zeros((total), dtype=np.int32)
    # Relation id
    sen_label = np.zeros((total), dtype=np.int32)
    # entity type pair id
    sen_entity_type = np.zeros((total), dtype=np.int32)
    # Bag scope
    instance_scope = []
    # Bag triplet
    instance_triple = []

    for s in range(total):
        content = f.readline()
        content = content.strip().split()

        relation = 0
        if content[6] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[6]]

        sentence = content[7:-1]
        # entits' positions
        en1 = content[2]
        en2 = content[3]
        en1pos = 0
        en2pos = 0
        
        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i

        for i in range(fixlen):
            sen_word[s][i] = word2id['BLANK']
            sen_pos1[s][i] = pos_embed(i - en1pos, opt.position_num)
            sen_pos2[s][i] = pos_embed(i - en2pos, opt.position_num)
            mask = 0
            if i >= len(sentence):
                mask = [0, 0, 0]
            elif i - en1pos <= 0:
                mask = [100, 0, 0]
            elif i - en2pos <= 0:
                mask = [0, 100, 0]
            else:
                mask = [0, 0, 100]
            sen_mask[s][i] = mask
            sen_type[s][i] = type2id['NA']

        for i in range(min(fixlen,len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            
            sen_word[s][i] = word

        # entity type
        if en1pos < fixlen:
            sen_type[s][en1pos] = type2id[content[4]]
        if en2pos < fixlen:
            sen_type[s][en2pos] = type2id[content[5]]
        
        type_pair = content[4] + '_' + content[5]
        sen_entity_type[s] = pair2id[type_pair]
        
        sen_len[s] = min(fixlen, len(sentence))
        sen_label[s] = relation

        # tup is bag_id
        tup = (content[0], content[1], relation)
        # If a new bag start
        if len(instance_triple) == 0 or instance_triple[len(instance_triple)-1] != tup:
            instance_triple.append(tup)
            instance_scope.append([s, s])
        # Update the scope
        instance_scope[len(instance_triple)-1][1] = s

    f.close()

    # so the length of instance_scope equals that of instance_triple
    # scopes of bags(en1_id, en2_id, relation)
    instance_scope = np.array(instance_scope)
    # list of bag_id
    instance_triple = np.array(instance_triple)
    return instance_triple, instance_scope, sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_type, sen_mask, sen_entity_type


def read_test_files(file, word2id, relation2id, type2id, pair2id, opt):
    """
    Split a sentence into word, position, type..., for training

    Args:
        file: train data file
        word2id: a dictionary
        relation2id: a dictionary
        type2id: a dictionary
        pair2id: a dictionary
        opt: preprocess options
    Return:
        instance_entity: a list of entity pair
        instance_entity_no_bag:
        instance_triple: a list of triples without relation NA
        instance_scope: a list of scops for each bag
        sen_len: real length of each sentence
        sen_label: relation label of each sentence
        sen_word: word ids of each sentence
        sen_pos1: left position ids of each sentence
        sen_pos2: right position ids of each sentence
        sen_type: type ids of each sentence
        sen_mask: mask of each sentence
        sen_entity_type: entity type pair label of each sentence
    """

    logger.info('Reading test data ...')

    f = codecs.open(file, 'r', encoding='utf-8')

    total = (int)(f.readline().strip())

    fixlen = opt.num_steps

    sen_word = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos1 = np.zeros((total, fixlen), dtype=np.int32)
    sen_pos2 = np.zeros((total, fixlen), dtype=np.int32)
    sen_mask = np.zeros((total, fixlen, 3), dtype=np.float32)
    sen_type = np.zeros((total, fixlen), dtype=np.int32)
    # Actual length of sentence
    sen_len = np.zeros((total), dtype=np.int32)
    # Relation id
    sen_label = np.zeros((total), dtype=np.int32)
    # entity type pair id
    sen_entity_type = np.zeros((total), dtype=np.int32)
    # Bag scope
    instance_scope = []
    # Bag triplet
    instance_triple = []
    instance_scope_with_NA = []
    instance_entity = []
    instance_entity_no_bag = []

    instances = []
    for _ in range(total):
        content = f.readline()
        content = content.strip().split()
        en1 = content[2]
        en2 = content[3]
        en1_id = content[0]
        en2_id = content[1]
        en1_type = content[4]
        en2_type = content[5]
        sentence = content[7:-1]
        relation = 0
        if content[6] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[6]]

        tup = (str(en1_id)+'\t'+str(en2_id)+'\t'+str(relation), sentence, en1_id, en2_id, en1, en2, en1_type, en2_type, relation)
        instances.append(tup)
    # sorted by bag_id
    instances = sorted(instances, key=lambda x:x[0])

    for s in range(total):
        #unique_id, sentence, en1_id, en2_id, en1_name, en2_name, relation = instances[s]
        _, sentence, en1_id, en2_id, en1_name, en2_name, en1_type, en2_type, relation = instances[s]

        en1pos = 0
        en2pos = 0
        
        for i in range(len(sentence)):
            if sentence[i] == en1_name:
                en1pos = i
            if sentence[i] == en2_name:
                en2pos = i

        for i in range(fixlen):
            sen_word[s][i] = word2id['BLANK']
            sen_pos1[s][i] = pos_embed(i - en1pos, opt.position_num)
            sen_pos2[s][i] = pos_embed(i - en2pos, opt.position_num)
            mask = 0
            if i >= len(sentence):
                mask = [0, 0, 0]
            elif i - en1pos <= 0:
                mask = [100, 0, 0]
            elif i - en2pos <= 0:
                mask = [0, 100, 0]
            else:
                mask = [0, 0, 100]
            sen_mask[s][i] = mask
            sen_type[s][i] = type2id['NA']

        for i in range(min(fixlen,len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            sen_word[s][i] = word

        # entity type
        if en1pos < fixlen:
            sen_type[s][en1pos] = type2id[en1_type]
        if en2pos < fixlen:
            sen_type[s][en2pos] = type2id[en2_type]

        type_pair = en1_type + '_' + en2_type
        sen_entity_type[s] = pair2id[type_pair]

        sen_len[s] = min(fixlen, len(sentence))
        sen_label[s] = relation

        tup = (en1_id, en2_id, relation)
        # tup[:2] = (en1_id, en2_id)
        instance_entity_no_bag.append(tup[:2])

        # instance_scope_with_NA contains bags with relation == 0
        # if len(instance_scope_with_NA) == 0 or the last element of it != tup. It means a new bag begins
        if len(instance_scope_with_NA) == 0 or instance_scope_with_NA[len(instance_scope_with_NA)-1] != tup:

            # It means a new entity pair begins
            if len(instance_scope_with_NA) == 0 or instance_scope_with_NA[len(instance_scope_with_NA)-1][:2] != tup[:2]:
                instance_scope.append([s, s])
                instance_entity.append(tup[:2])

            instance_scope_with_NA.append(tup)

            # Add triple without relation of NA
            if tup[2] != 0:
                instance_triple.append(tup)
        
        # When begin a new entity pairs, instance_scope adds a new elements, otherwise only updates the scope
        # So it represents the scope of entity pairs. Hence within a scope there exists mutiple relations
        instance_scope[len(instance_scope)-1][1] = s
    
    f.close()

    # so the length of instance_scope doesn't equals to that of instance_triple

    # entity pairs
    instance_entity = np.array(instance_entity)
    instance_entity_no_bag = np.array(instance_entity_no_bag)
    # triples without relation of NA
    instance_triple = np.array(instance_triple)
    # scopes of entity pairs
    instance_scope = np.array(instance_scope)

    return instance_entity, instance_entity_no_bag, instance_triple, instance_scope, sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_type, sen_mask, sen_entity_type


def pos_embed(x, position_num):
    """
    get position embedding of x
    """
    maxlen = int(position_num / 2)
    return max(0, min(x + maxlen, position_num))


def find_index(x,y):
    """
    find the index of x in y, if x not in y, return -1
    """
    for index, item in enumerate(y):
        if x == item:
            return index
        return -1


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    opts.preprocess_opts(parser)
    opts.shared_opts(parser)
    opt = parser.parse_args()

    return opt


def save_array(opt, name, variable):
    """
    Save processed data as torch format

    Args:
        opt: processed options
        variable: data to save
        name: variable name, eg. '[train|test]_len'
    Return:
        None
    """

    pt_file = os.path.join(opt.processed_dir, name+'.pt')
    torch.save(variable, pt_file)


def main():
    """
    Transform dataset from txt to numpy data
    """

    init_logger()

    opt = parse_args()

    ensure_folder(opt.processed_dir)

    word2id, vec = word2id_fn(opt.word2vec_file)
    relation2id = relation2id_fn(opt.relation2id_file)
    type2id, pair2id = type2id_fn(opt.type2id_file, opt.type_num)
    gather_bag(opt.train_file, relation2id, opt.tmp_train_bag)
    gather_bag(opt.test_file, relation2id, opt.tmp_test_bag)

    instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_type, train_mask, train_type_label =  \
        read_train_files(opt.tmp_train_bag, word2id, relation2id, type2id, pair2id, opt)

    save_array(opt, 'train_instance_triple', instance_triple)
    save_array(opt, 'train_instance_scope', instance_scope)
    save_array(opt, 'train_len', train_len)
    save_array(opt, 'train_label', train_label)
    save_array(opt, 'train_word', train_word)
    save_array(opt, 'train_pos1', train_pos1)
    save_array(opt, 'train_pos2', train_pos2)
    save_array(opt, 'train_type', train_type)
    save_array(opt, 'train_mask', train_mask)
    save_array(opt, 'train_type_label', train_type_label)
    logger.info('Length of train_instance_scope: {}'.format(len(instance_scope)))

    instance_entity, instance_entity_no_bag, instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_type, test_mask, test_type_label = \
        read_test_files(opt.tmp_test_bag, word2id, relation2id, type2id, pair2id, opt)

    save_array(opt, 'test_instance_entity', instance_entity)
    save_array(opt, 'test_instance_entity_no_bag', instance_entity_no_bag)
    save_array(opt, 'test_instance_triple', instance_triple)
    save_array(opt, 'test_instance_scope', instance_scope)
    save_array(opt, 'test_len', test_len)
    save_array(opt, 'test_label', test_label)
    save_array(opt, 'test_word', test_word)
    save_array(opt, 'test_pos1', test_pos1)
    save_array(opt, 'test_pos2', test_pos2)
    save_array(opt, 'test_type', test_type)
    save_array(opt, 'test_mask', test_mask)
    save_array(opt, 'test_type_label', test_type_label)
    logger.info('Length of test_instance_scope: {}'.format(len(instance_scope)))

    # Remove temp files
    remove_file_or_folder(opt.tmp_train_bag)
    remove_file_or_folder(opt.tmp_test_bag)

    # Save word2vec file
    save_array(opt, 'vec', vec)


if __name__ == '__main__':
    main()
