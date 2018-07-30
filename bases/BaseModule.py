#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import torch as t
import os
from utils.utils import mkdir_if_not_exist
from torch.nn import functional as F
from torch.nn import Parameter
import time


class BaseModule(t.nn.Module):
    """
    packaged nn.Module giving save() and load()
    """

    def forward(self, *input_data):
        """
        Defines the computation performed at every call.

        Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def __init__(self, config):
        super(BaseModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字
        self.config = config

    def load(self, path):
        """
        Load Saved Model
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        save model, to file "experiments/model_name/checkpoints/ + time" by default
        """
        if name is None:
            time_str = time.strftime('%m%d_%H:%M:%S.pth')
            if mkdir_if_not_exist([os.path.join(self.config.logdir, self.model_name, 'checkpoints')]):
                name = os.path.join(self.config.logdir, self.model_name, 'checkpoints', time_str)
            else:
                raise SystemError('[ERROR] Cannot make checkpoint directory due to access deny or other issues')

        t.save(self.state_dict(), name)
        return name


