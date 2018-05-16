#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by H. L. Wang on 2018/4/18
"""
from torch import nn
import torch.nn.functional as F

from bases.BaseModule import BaseModule


class SimpleMnistModel(BaseModule):
    """
    SimpleMnist模型
    """

    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)
        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc_model = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, *input_data):
        # plot_model(model, to_file=os.path.join(self.config.img_dir, "model.png"), show_shapes=True)  # 绘制模型图
        x = self.conv_model(input_data[0])
        x = x.view(-1, 320)
        x = self.fc_model(x)
        return F.log_softmax(x)

