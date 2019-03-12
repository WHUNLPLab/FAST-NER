"""
    Manage checkpoints, including the number, history
"""

import torch

from nre.utils.file_helper import check_file, remove_file_or_folder, makedirs_before_saving, copy_file

class CheckpointManager(object):
    def __init__(self, opt):
        self.max_checkpoint_num = opt.model_max_num
        self.checkpoints_history_file = opt.checkpoints_history_file
        self.model_prefix = opt.model_prefix

        # A dictionary whose key is model path and value is step
        self.checkpoint_history = {}

        self._load_checkpoint_history()

    def _load_checkpoint_history(self):
        """
        Load checkpoints history file if existed
        """

        if check_file(self.checkpoints_history_file):
            self.checkpoint_history = torch.load(self.checkpoints_history_file)

    def save_checkpoints_history(self):
        """
        Save to file
        """    

        makedirs_before_saving(self.checkpoints_history_file)
        torch.save(self.checkpoint_history, self.checkpoints_history_file)

    def update_checkpoint_history(self, step, new_best, model_path):
        """
        Add new checkpoint information into history. 
        If the length of checkpoint_history is greater than max_checkpoint_num, the oldest must be dropout.

        Args:
            step: the step of the model to be saved
            new_best: If this checkpoint is a best one
            model_path: the path of the model
        """

        if len(self.checkpoint_history) >= self.max_checkpoint_num:
            min_step = min(self.checkpoint_history.values())
            min_checkpoint = self.create_model_filename(min_step, False)
            self.checkpoint_history.pop(min_checkpoint)

            remove_file_or_folder(min_checkpoint)

        self.checkpoint_history[model_path] = step

        if new_best:
            model_best_path = self.create_model_filename(0, True)
            copy_file(model_path, model_best_path)

    def step_of_checkpoint(self, checkpoint):
        """
        When restore an existed checkpoint, we also restore the corresponding step

        Args:
            checkpoint: model_path, also called pretrain_model
        Return:
            the step of the checkpoint
        """

        if checkpoint in self.checkpoint_history:
            return self.checkpoint_history[checkpoint]
        else:
            return 0

    def create_model_filename(self, step, is_best):
        """
        Args:
            step: step of the model
            is_best: if this model is the best model
        Return:
            filename
        """

        if is_best:
            return self.model_prefix + 'best.pt'
        else:
            return self.model_prefix + str(step) + '.pt'
