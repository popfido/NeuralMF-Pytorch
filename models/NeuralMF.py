#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import torch
from bases.BaseModule import BaseModule
from models.ModuleUtils import golorot_uniform, lecunn_uniform
from utils.utils import _save_model, _save_onnx_model, _generate_model_path

import torch.nn as nn
import time


class NeuralMF(BaseModule):
    def __init__(self, config, nb_users, nb_items):
        super(NeuralMF, self).__init__(config)
        self.model_name = "NeuralMF"
        nb_mlp_layers = len(config.layers)
        # TODO: regularization?
        self.mf_user_embed = nn.Embedding(nb_users, config.factors)
        self.mf_item_embed = nn.Embedding(nb_items, config.factors)
        self.mlp_user_embed = nn.Embedding(nb_users, config.layers[0] // 2)
        self.mlp_item_embed = nn.Embedding(nb_items, config.layers[0] // 2)

        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(config.layers[i - 1], config.layers[i])])

        self.fc_final = nn.Linear(config.layers[-1] + config.factors, 1)

        self.mf_user_embed.weight.data.normal_(0., 0.01)
        self.mf_item_embed.weight.data.normal_(0., 0.01)
        self.mlp_user_embed.weight.data.normal_(0., 0.01)
        self.mlp_item_embed.weight.data.normal_(0., 0.01)

        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)
        lecunn_uniform(self.fc_final)

        if config.cuda:
            self.cuda()

    def forward(self, input_data, sigmoid=False):
        xmfu = self.mf_user_embed(input_data[0])
        xmfi = self.mf_item_embed(input_data[1])
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(input_data[0])
        xmlpi = self.mlp_item_embed(input_data[1])
        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)

        x = torch.cat((xmf, xmlp), dim=1)
        x = self.fc_final(x)
        if sigmoid:
            x = nn.functional.sigmoid(x)
        return x

    def save(self, name=None, time_str=time.strftime('%m%d_%H:%M:%S'), use_onnx=False):
        path = _generate_model_path(self.config.logdir, self.model_name, time_str)
        if not use_onnx:
            _save_model(path, self.state_dict())
        else:
            if self.config.cuda:
                _save_onnx_model(path, self, [torch.tensor([1]).cuda(self.config.cuda_device),
                                              torch.tensor([1]).cuda(self.config.cuda_device)])
            else:
                _save_onnx_model(path, self, [torch.tensor([1]),
                                              torch.tensor([1])])

