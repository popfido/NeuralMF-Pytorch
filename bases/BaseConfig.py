#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""


from __future__ import print_function
from __future__ import absolute_import

import warnings
import argparse

from utils.configUtils import process_config


class BaseConfig(object):
    def parse_by_kwargs(self, kwargs):
        """
        update config parameter according to kwargs
        :kwargs: dictionary of updating parameters
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

    def parse_args(self, args=None):
        raise NotImplementedError

    @staticmethod
    def parse_by_json():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            '-c', '--cfg',
            dest='config',
            metavar='path',
            default='None',
            help='add a configuration file')
        args = parser.parse_args()
        return process_config(args.config)

    def save(self):
        raise NotImplementedError
