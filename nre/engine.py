"""
    Run model and get results
"""

import torch
import os
import torch.optim as optim

from nre.utils.file_helper import ensure_folder, check_file, remove_file_or_folder, makedirs_before_saving
from nre.utils.logging import logger
from nre.dataset.data_manager import DataManager
from nre.utils.statistic import Statistics
from nre.utils.gpu_utils import GpuUtils
from nre.utils.checkpoint_manager import CheckpointManager
from nre.model.model_helper import ModelHelper
from nre.utils.report_manager import ReportManager


class Engine(object):
    def __init__(self, opt):
        """
        Initialize model engine

        Args:
            train_data_loader: loader for training set
            test_data_loader: loader for testing set
        """

        self.data_manager = DataManager()
        self.gpu_utils = GpuUtils(opt)
        self.stats = Statistics()
        self.checkpoint_manager = CheckpointManager(opt)
        self.model_helper = ModelHelper(opt, self.gpu_utils)

        self.optim_step = 0
        self.model = None
        self.optimizer = None

    def launch(self, opt):
        """
        Call before train and evaluation.
        Build a model, and load a pretrained model if specified
        """

        self.model = self.build_model(opt)
        self.load_model(opt)

    def train(self, opt):
        """
        Train and evaluate model
        """

        step = self.optim_step + 1
        while step <= opt.train_steps:
            for _, batch in enumerate(
                self.data_manager.get_train_data_loader(opt.processed_dir, opt.batch_size)):
                # A train step
                batch_word = self.gpu_utils.to_cuda(batch.word)
                batch_pos1 = self.gpu_utils.to_cuda(batch.pos1)
                batch_pos2 = self.gpu_utils.to_cuda(batch.pos2)
                batch_type = self.gpu_utils.to_cuda(batch.type)
                batch_mask = self.gpu_utils.to_cuda(batch.mask)
                # batch_lens = self.gpu_utils.to_cuda(batch.lens)
                batch_scope = self.gpu_utils.to_cuda(batch.scope)
                label_for_select = self.gpu_utils.to_cuda(batch.label_for_select)
                batch_label = self.gpu_utils.to_cuda(batch.label)
                batch_type_label = self.gpu_utils.to_cuda(batch.type_label)

                self.optimizer.zero_grad()

                loss, accuracy = self.model(True, batch_word, batch_pos1, batch_pos2, batch_mask, batch_type, batch_scope, batch_type_label, batch_label, label_for_select)

                loss.backward()
                self.optimizer.step()

                if step % opt.print_steps == 0:
                    logger.info('step: {}, loss: {:.6f}, accuracy {:.6f}'.format(step, loss.data, accuracy))

                if step % opt.check_steps == 0:
                    self.optim_step = step
                    self.eval(False, opt)

                    if self.stats.stop_training_or_not(opt.stop_after_n_eval):
                        break

                    self.model.train()

                step += 1
                if step > opt.train_steps:
                    break

            if self.stats.stop_training_or_not(opt.stop_after_n_eval):
                break

        logger.info('Saving result in {}'.format(opt.res_dir))
        self.stats.final_save(opt.model_type, opt.res_dir)

    def eval(self, is_eval, opt):
        """
        Test the model using the same networks

        Args:
            is_eval: for evaluation or training
            opt: options for evaluation
        """

        logger.info('Evaluate {} model for step {} in {}'.format(opt.model_type, self.optim_step, 'evaluation' if is_eval else 'training'))
        self.model.eval()

        test_result = []
        total_recall = 0

        with torch.no_grad():
            for i, batch in enumerate(
                self.data_manager.get_test_data_loader(opt.processed_dir, opt.batch_size)):
                # A test step
                batch_word = self.gpu_utils.to_cuda(batch.word)
                batch_pos1 = self.gpu_utils.to_cuda(batch.pos1)
                batch_pos2 = self.gpu_utils.to_cuda(batch.pos2)
                batch_type = self.gpu_utils.to_cuda(batch.type)
                batch_mask = self.gpu_utils.to_cuda(batch.mask)
                # batch_lens = self.gpu_utils.to_cuda(batch.lens)
                batch_scope = self.gpu_utils.to_cuda(batch.scope)
                # batch_label = self.gpu_utils.to_cuda(batch.label)
                batch_type_label = self.gpu_utils.to_cuda(batch.type_label)

                test_output = self.model(False, batch_word, batch_pos1, batch_pos2, batch_mask, batch_type, batch_scope, batch_type_label)

                for j in range(len(test_output)):
                    pred = test_output[j]
                    # entity_pair = self.data_manager.find_entity_pair(j + i * opt.batch_size)
                    entity_pair = batch.bag_ids[j]

                    for rel in range(1, len(pred)):
                        flag = int(self.data_manager.exists_test_triple((entity_pair[0], entity_pair[1], rel)))
                        total_recall += flag
                        # pred[rel] is a tensor, we should make it a cpu-variable
                        test_result.append([(entity_pair[0], entity_pair[1], rel), flag, float(pred[rel])])

                if (i+1) % opt.print_steps == 0:
                    logger.info('Have predicted {} batchs of entity pairs'.format(i+1))

                if opt.debug_mode and i == 0:
                    total_recall = 100 # Avoid DivideZeroError
                    break

        is_new_best = self.stats.calculate_test_result(test_result, total_recall)

        if not is_eval:
            self.save_model(self.optim_step, is_new_best, opt)

    def build_model(self, opt):
        """
        Build model for training and evaluation.
        Here, training model and testing model are the same.

        Args:
            opt: trianing options

        Return:
            model: a model instance
        """

        logger.info('Build {} model for training and evaluation'.format(opt.model_type))

        model = self.model_helper.create_model(self.data_manager.load_pretrained_word2vec(opt.processed_dir), opt)
        model = self.gpu_utils.model_to_cuda(model)

        ReportManager.print_model(model.parameters())

        if opt.optimizer == 'Adam' or opt.optimizer == 'adam':
            self.optimizer = optim.Adam(
                params=model.parameters(), 
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay)
        elif opt.optimizer == 'SGD' or opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay
            )
        elif opt.optimizer == 'Adadelta' or opt.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(
                model.parameters(),
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay
            )
        elif opt.optimizer == 'Adagrad' or opt.optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(
                model.parameters(),
                lr=opt.learning_rate,
                lr_decay=opt.learning_rate_decay,
                weight_decay=opt.weight_decay
            )
        else:
            raise NotImplementedError

        return model

    def load_model(self, opt):
        """
        Load checkpoints history file if existed.
        Load a model from pretrained model

        Args:
            opt: train options
        """

        # Load pretrained model if specified
        if opt.pretrain_model != None and opt.pretrain_model != '':
            if not check_file(opt.pretrain_model):
                logger.error('Can not find pretrained model')
            else:
                logger.info('Load pretrained model from {}'.format(opt.pretrain_model))
                self.model.load_state_dict(torch.load(opt.pretrain_model))
                self.model = self.gpu_utils.model_to_cuda(self.model)
                self.optim_step = self.checkpoint_manager.step_of_checkpoint(opt.pretrain_model)
        else:
            logger.info('Pretrained model has not been specified')

    def save_model(self, step, new_best, opt):
        """
        Save model using the format of '"model_prefix"_step.pt'

        Args:
            step: the train step when saving model
            new_best: bool, if this checkpoint is a new best one
            opt: train options
        """

        model_path = self.checkpoint_manager.create_model_filename(step, False)

        # Save model
        makedirs_before_saving(model_path)
        torch.save(self.model.state_dict(), model_path)

        logger.info('Save model {} to {}'.format(opt.model_type, model_path))

        # Save history
        self.checkpoint_manager.update_checkpoint_history(step, new_best, model_path)
        self.checkpoint_manager.save_checkpoints_history()

        logger.info('Save checkpoints history to {}'.format(opt.checkpoints_history_file))
