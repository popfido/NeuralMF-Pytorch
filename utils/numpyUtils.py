#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

Reference: https://juejin.im/post/5acfef976fb9a028db5918b5
"""

import numpy as np


def prp_2_onehot_array(arr):
    """
    Prob Maxtrix to OneHot Matrix
    arr = np.array([[0.1, 0.5, 0.4], [0.2, 0.1, 0.6]])
    :param arr:
        np.array, Probability Matrix
    :return:
        np.array, One Hot Matrix
    """
    arr_size = arr.shape[1]  # num of category
    arr_max = np.argmax(arr, axis=1)  # pos of max val
    return np.eye(arr_size)[arr_max]  # One Hot
