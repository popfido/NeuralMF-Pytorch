# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from bases.BaseDataLoader import BaseDataLoader


class SimpleMnistDL(BaseDataLoader):
    def __init__(self, config=None):
        super(SimpleMnistDL, self).__init__(config)
        self.train = MNIST('data',
                           download=True,
                           train=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),  # first, convert image to PyTorch tensor
                               transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
                           ]))

        self.test = MNIST('data',
                          download=True,
                          train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),  # first, convert image to PyTorch tensor
                              transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
                          ]))
        print("[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(self.train.train_data.size()), str(self.train.train_labels.size())))
        print("[INFO] X_test.shape: %s, y_test.shape: %s" \
              % (str(self.test.test_data.size()), str(self.test.test_labels.size())))

    def get_train_data(self):
        return DataLoader(self.train,
                          batch_size=self.config.batch_size,
                          shuffle=True)

    def get_test_data(self):
        return DataLoader(self.test,
                          batch_size=self.config.batch_size,
                          shuffle=True)
