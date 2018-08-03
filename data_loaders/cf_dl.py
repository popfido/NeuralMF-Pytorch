#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import numpy as np
import os
import scipy
import scipy.sparse
from torch.utils.data import DataLoader
import torch.utils.data

from bases.BaseDataLoader import BaseDataLoader
from data.get_movielen import (_TEST_NEG_FILENAME, _TEST_RATINGS_FILENAME,
                               _TRAIN_RATINGS_FILENAME)


class CFDataLoader(BaseDataLoader):
    def __init__(self, config=None, only_test=False, test_file=None):
        super(CFDataLoader, self).__init__(config)
        self.train_data = None
        if not only_test:
            self.train_data = CFDataset(os.path.join(config.data, _TRAIN_RATINGS_FILENAME), config.negative_samples)
            self.test_data = CFValidDataSet(os.path.join(config.data, _TEST_RATINGS_FILENAME),
                                        os.path.join(config.data, _TEST_NEG_FILENAME))
        else:
            self.test_data = CFTestDataSet(test_file)

    def get_train_data(self):
        if not self.train_data:
            raise ValueError("Training Dataset Not Set")
        return DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True,
                          num_workers=8, pin_memory=False, )

    def get_test_data(self):
        if not self.test_data:
            raise ValueError("Test Dataset Not Set")
        return self.test_data

    def get_num_user_and_item(self):
        if not self.train_data:
            raise ValueError("Training Dataset Not Set")
        return self.train_data.nb_users, self.train_data.nb_items


class CFDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, fname, nb_neg):
        self._load_train_matrix(fname)
        self.nb_neg = nb_neg

    def _load_train_matrix(self, train_fname):
        def process_line(line):
            tmp = line.split('\t')
            # user, item, rating
            return [int(tmp[0]), int(tmp[1]), float(tmp[2]) > 0]

        # these files are a few hundred megs tops
        with open(train_fname, 'r') as file:
            data = list(map(process_line, file))
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
            j = torch.LongTensor(1).random_(0, self.nb_items).item()
            while (u, j) in self.mat:
                j = torch.LongTensor(1).random_(0, self.nb_items).item()
            return u, j, np.zeros(1, dtype=np.float32)


class CFValidDataSet(torch.utils.data.dataset.Dataset):
    # Container of (rating, items)
    def __init__(self, fname_ratings, fname_negs):
        self.data = [(rating, items) for rating, items in
                     zip(_load_valid_ratings(fname_ratings), _load_valid_negs(fname_negs))]

    def __len__(self):
        return len(self.data)

    # Return ( [user] * len(items + 1), items + [test_item] ), test_item
    def __getitem__(self, idx):
        return ([self.data[idx][0][0]] * (len(self.data[idx][1]) + 1), self.data[idx][1] + [self.data[idx][0][1]]), \
               self.data[idx][0][1]


class CFTestDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, fname):
        self.data = [(rating, items) for rating, items in
                     _load_test_pairs(fname)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


def _load_test_pairs(fname):
    def process_line(line):
        tmp = map(int, line.strip().split('\t')[0:2])
        return list(tmp)

    pairs = map(process_line, open(fname, 'r'))
    return list(pairs)

def _load_valid_ratings(fname):
    def process_line(line):
        tmp = map(int, line.split('\t')[0:2])
        return list(tmp)

    ratings = map(process_line, open(fname, 'r'))
    return list(ratings)


def _load_valid_negs(fname):
    def process_line(line):
        tmp = map(int, line.split('\t'))
        return list(tmp)

    negs = map(process_line, open(fname, 'r'))
    return list(negs)
