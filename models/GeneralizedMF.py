#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import torch
from bases.BaseModule import BaseModule
from models.ModuleUtils import lecunn_uniform
import torch.nn as nn


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

    def forward(self, user, item, sigmoid=False):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        x = self.fc_final(xmf)
        if sigmoid:
            x = nn.functional.sigmoid(x)
        return x