"""
    Helper of file operations
"""

import os
import shutil
import glob


def ensure_folder(folder):
    """
    If folder doesn't exist, make it

    Args:
        folder: path of folder
    """

    if not os.path.exists(folder):
        os.makedirs(folder)


def check_file(file):
    """
    If file is existed
    """

    return os.path.exists(file)


def makedirs_before_saving(file_path):
    """
    Make dirs before saving a file.
    e.g. models/model_step_100.pt, it ensure models directory

    Args:
        file_path: the full path of file
    """

    dir = os.path.dirname(file_path)
    ensure_folder(dir)


def remove_file_or_folder(fdes):
    if os.path.isdir(fdes):
        shutil.rmtree(fdes)
    elif os.path.isfile(fdes):
        os.remove(fdes)


def find_files(folder, pattern):
    """
    Find files in folder whose file name satisfies to the pattern

    Args:
        folder: folder which contains files to find
        pattern: pattern file name, e.g. model_*.pt
    Return:
        Return a possibly-empty list of path names that match pattern.
    """

    full_pattern_name = os.path.join(folder, pattern)
    return glob.glob(full_pattern_name)


def move_file(srcfile, dstfile):
    """
    Move srcfile to dstfile

    Args:
        srcfile: the file to move
        dstfile: destination of file
    """

    shutil.move(srcfile, dstfile)


def copy_file(srcfile, dstfile):
    """
    Copy srcfile to dstfile

    Args:
        srcfile: the file to copy
        dstfile: destination of file
    """

    shutil.copyfile(srcfile, dstfile)
