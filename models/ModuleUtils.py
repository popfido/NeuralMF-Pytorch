#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import torch as t


class Flatten(t.nn.Module):
    """
    flatten input to（batch_size,dim_length）
    """

    def __init__(self):
        super(Flatten, self).__init__()
        # self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)