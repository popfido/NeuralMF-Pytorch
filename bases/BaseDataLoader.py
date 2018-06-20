#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""


class BaseDataLoader(object):
    """
    Data Loader Base Class
    """

    def __init__(self, config):
        self.config = config  # Set config

    def get_train_data(self):
        """
        Get training Data
        """
        raise NotImplementedError

    def get_test_data(self):
        """
        Get Test Data
        """
        raise NotImplementedError
