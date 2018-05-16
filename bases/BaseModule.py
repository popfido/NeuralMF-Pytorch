#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import torch as t
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
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        save model, to file "checkpoints/model_name + _ + time" by default
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


