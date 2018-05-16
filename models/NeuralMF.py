# coding=utf-8

import numpy as np
import torch
from bases.BaseModule import BaseModule
import torch.nn as nn

class NeuMF(BaseModule):
    def __init__(self, config):
        super(NeuMF, self).__init__(config)
        nb_mlp_layers = len(config.mlp_layer_sizes)
        # TODO: regularization?
        self.mf_user_embed = nn.Embedding(config.nb_users, config.mf_dim)
        self.mf_item_embed = nn.Embedding(config.nb_items, config.mf_dim)
        self.mlp_user_embed = nn.Embedding(config.nb_users, config.mlp_layer_sizes[0] // 2)
        self.mlp_item_embed = nn.Embedding(config.nb_items, config.mlp_layer_sizes[0] // 2)

        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(config.mlp_layer_sizes[i - 1], config.mlp_layer_sizes[i])])  # noqa: E501

        self.final = nn.Linear(config.mlp_layer_sizes[-1] + config.mf_dim, 1)

        self.mf_user_embed.weight.data.normal_(0., 0.01)
        self.mf_item_embed.weight.data.normal_(0., 0.01)
        self.mlp_user_embed.weight.data.normal_(0., 0.01)
        self.mlp_item_embed.weight.data.normal_(0., 0.01)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)

        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, *input_data):
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
        x = self.final(x)
        if input_data[2]:
            x = nn.functional.sigmoid(x)
        return x