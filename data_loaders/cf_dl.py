#!/usr/bin/env python
# -- coding: utf-8 --

import numpy as np
import scipy
import scipy.sparse
from torch.utils.data import DataLoader
import torch.utils.data

from bases.BaseDataLoader import BaseDataLoader


class CFDataLoader(BaseDataLoader):
    def __init__(self, config=None, only_test=False):
        super(CFDataLoader, self).__init__(config)
        self.train_data = None
        if not only_test:
            self.train_data = CFDataset(config.data, config.negative_samples)
        self.test_data = zip(_load_test_ratings(), _load_test_negs())

    def get_train_data(self):
        if not self.train_data:
            raise ValueError("Training Dataset Not Set")
        return DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True)

    def get_test_data(self):
        if not self.train_data:
            raise ValueError("Test Dataset Not Set")
        return DataLoader(self.test_data,  batch_size=self.config.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True)


class CFDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, fname, nb_neg):
        self._load_train_matrix(fname)
        self.nb_neg = nb_neg

    def _load_train_matrix(self, train_fname):
        def process_line(line):
            tmp = line.split('\t')
            # user, item, rating???
            return [int(tmp[0]), int(tmp[1]), float(tmp[2]) > 0]

        # these files are a few hundred megs tops
        # TODO: be unlazy? use pandas?
        lines = open(train_fname, 'r').readlines()[1:]
        data = list(map(process_line, lines))
        self.nb_users = max(data, key=lambda x: x[0])[0] + 1
        self.nb_items = max(data, key=lambda x: x[1])[1] + 1

        self.data = list(filter(lambda x: x[2], data))
        self.mat = scipy.sparse.dok_matrix(
            (self.nb_users, self.nb_items), dtype=np.float32)
        for user, item, _ in data:
            self.mat[user, item] = 1.

    def __len__(self):
        return (self.nb_neg + 1) * len(self.data)

    def __getitem__(self, idx):
        if idx % (self.nb_neg + 1) == 0:
            idx = idx // (self.nb_neg + 1)
            return self.data[idx][0], self.data[idx][1], np.ones(1, dtype=np.float32)  # noqa: E501
        else:
            idx = idx // (self.nb_neg + 1)
            u = self.data[idx][0]
            j = np.random.randint(self.nb_items)
            while (u, j) in self.mat:
                j = np.random.randint(self.nb_items)
            return u, j, np.zeros(1, dtype=np.float32)


def _load_test_ratings(fname):
    def process_line(line):
        tmp = map(int, line.split('\t')[0:2])
        return list(tmp)

    lines = open(fname, 'r').readlines()
    ratings = map(process_line, lines)
    return list(ratings)


def _load_test_negs(fname):
    def process_line(line):
        tmp = map(int, line.split('\t')[1:])
        return list(tmp)

    lines = open(fname, 'r').readlines()
    negs = map(process_line, lines)
    return list(negs)
