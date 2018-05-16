# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by H. L. Wang on 2018/5/11
"""


class BaseDataLoader(object):
    """
    数据加载的基类
    """

    def __init__(self, config):
        self.config = config  # 设置配置信息

    def get_train_data(self):
        """
        获取训练数据
        """
        raise NotImplementedError

    def get_test_data(self):
        """
        获取测试数据
        """
        raise NotImplementedError
