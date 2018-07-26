#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import torch
from bases.BaseModule import BaseModule
from models.ModuleUtils import golorot_uniform, lecunn_uniform
import torch.nn as nn


class NeuralMF(BaseModule):
    def __init__(self, config, nb_users, nb_items):
        super(NeuralMF, self).__init__(config)
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

    def forward(self, user, item, sigmoid=False):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)

        x = torch.cat((xmf, xmlp), dim=1)
        x = self.fc_final(x)
        if sigmoid:
            x = nn.functional.sigmoid(x)
        return x