import logging
import math
import os
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime
from shutil import get_terminal_size

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def OrderedYaml():
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info("Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_output_folders(opt):
    # store trained models path
    models_save_path = opt.saveTrainedModelsPath + '_x' + str(opt.upscale_factor) + '/'
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)

    # store training states path
    training_states_save_path = models_save_path + opt.save_path_training_states
    if not os.path.exists(training_states_save_path):
        os.makedirs(training_states_save_path)

    # store trained netG path
    netG_save_path = models_save_path + opt.save_path_netG
    if not os.path.exists(netG_save_path):
        os.makedirs(netG_save_path)

    # store trained netD path
    netD_save_path = models_save_path + opt.save_path_netD
    if not os.path.exists(netD_save_path):
        os.makedirs(netD_save_path)

    # store trained model best PSNR path
    best_psnr_save_path = models_save_path + opt.save_path_best_psnr
    if not os.path.exists(best_psnr_save_path):
        os.makedirs(best_psnr_save_path)

    # store trained model best LPIPS path
    best_lpips_save_path = models_save_path + opt.save_path_best_lpips
    if not os.path.exists(best_lpips_save_path):
        os.makedirs(best_lpips_save_path)

    # save train images path
    save_train_imgs_path = opt.saveImgsPath + '_x' + str(opt.upscale_factor) + '/train_imgs/'
    if not os.path.exists(save_train_imgs_path):
        os.makedirs(save_train_imgs_path)

    # save test images path
    save_test_imgs_path = opt.saveImgsPath + '_x' + str(opt.upscale_factor) + '/test_imgs/'
    if not os.path.exists(save_test_imgs_path):
        os.makedirs(save_test_imgs_path)

    # logs path
    logs_save_path = opt.saveLogsPath + '_x' + str(opt.upscale_factor) + '/'
    if not os.path.exists(logs_save_path):
        os.makedirs(logs_save_path)

    return logs_save_path, training_states_save_path, save_test_imgs_path,best_psnr_save_path, best_lpips_save_path, netG_save_path , netD_save_path,save_train_imgs_path


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


class ProgressBar(object):
    """A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = bar_width if bar_width <= max_bar_width else max_bar_width
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(
                "terminal width is too small ({}), please consider widen the terminal for better "
                "progressbar visualization".format(terminal_width)
            )
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(
                "[{}] 0/{}, elapsed: 0s, ETA:\n{}\n".format(
                    " " * self.bar_width, self.task_num, "Start..."
                )
            )
        else:
            sys.stdout.write("completed: 0, elapsed: 0s")
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg="In progress..."):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = ">" * mark_width + "-" * (self.bar_width - mark_width)
            sys.stdout.write("\033[2F")  # cursor up 2 lines
            sys.stdout.write(
                "\033[J"
            )  # clean the output (remove extra chars since last display)
            sys.stdout.write(
                "[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n".format(
                    bar_chars,
                    self.completed,
                    self.task_num,
                    fps,
                    int(elapsed + 0.5),
                    eta,
                    msg,
                )
            )
        else:
            sys.stdout.write(
                "completed: {}, elapsed: {}s, {:.1f} tasks/s".format(
                    self.completed, int(elapsed + 0.5), fps
                )
            )
        sys.stdout.flush()
