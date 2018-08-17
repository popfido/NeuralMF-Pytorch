#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import torch
from bases.BaseModule import BaseModule
from models.ModuleUtils import lecunn_uniform
from utils.utils import _save_model, _save_onnx_model, _generate_model_path

import torch.nn as nn
import time

class GeneralizedMatrixFactorization(BaseModule):
    def __init__(self, config, nb_users, nb_items):
        super(GeneralizedMatrixFactorization, self).__init__(config)
        self.model_name = "GeneralizedMF"
        self.user_embed = nn.Embedding(nb_users, config.factors)
        self.item_embed = nn.Embedding(nb_items, config.factors)

        self.fc_final = nn.Linear(config.layers[-1], 1)

        self.user_embed.weight.data.normal_(0., 0.01)
        self.item_embed.weight.data.normal_(0., 0.01)

        lecunn_uniform(self.fc_final)

        if config.cuda:
            self.cuda()

    def forward(self, input_data, sigmoid=False):
        xmfu = self.mf_user_embed(input_data[0])
        xmfi = self.mf_item_embed(input_data[1])
        xmf = xmfu * xmfi

        x = self.fc_final(xmf)
        if sigmoid:
            x = nn.functional.sigmoid(x)
        return x

    def save(self, name=None, time_str=time.strftime('%m%d_%H:%M:%S'), use_onnx=False):
        path = _generate_model_path(self.config.logdir, self.model_name, time_str)
        if not use_onnx:
            _save_model(path, self.state_dict())
        else:
            if self.config.cuda:
                self.cpu()
            _save_onnx_model(path, self, [torch.tensor([1]),
                                              torch.tensor([1])])
            if self.config.cuda:
                self.gpu()
