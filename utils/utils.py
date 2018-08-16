#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

from __future__ import print_function
from __future__ import absolute_import

import os
import time
import shutil
import datetime
import warnings
import codecs
import numpy as np
import torch as t
import json

try:
    from inspect import signature
except:
    warnings.warn('inspect.signature not available... '
                  'you should upgrade to Python 3.x')


def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")


def _generate_model_path(logdir, model_name, time_str):
    if mkdir_if_not_exist([os.path.join(logdir, model_name, 'checkpoints')]):
        name = os.path.join(logdir, model_name, 'checkpoints', time_str)
    else:
        raise SystemError('[ERROR] Cannot make checkpoint directory due to access deny or other issues')
    return name


def _save_model(path, state_dict):
    t.save(state_dict, path + ".pth")


def _save_onnx_model(path, model, x):
    t.onnx._export(model,  # model being run
                   x,  # model input (or a tuple for multiple inputs)
                   path + ".onnx",  # where to save the model (can be a file or file-like object)
                   export_params=True)


def mkdir_if_not_exist(dirs, is_delete=False):
    """
    创建文件夹
    :param dirs: list of directory to checkout
    :param is_delete: whether delete old directory if it exists
    :return: Boolean that indicate successful or not
    """
    try:
        for dir_ in dirs:
            if dir != '':
                if is_delete:
                    if os.path.exists(dir_):
                        shutil.rmtree(dir_)
                        print(u'[INFO] directory "%s" has already been exist, delete it.' % dir_)

                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                    print(u'[INFO] directory "%s" do not exist, make a new directory.' % dir_)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def write_output(test_dl, predict, file):
    mkdir_if_not_exist([os.path.dirname(file)])

    with codecs.open(file, 'w') as f:
        for idx, (user, item) in enumerate(test_dl):
            f.write(','.join(map(str, [user, item, np.array(predict[idx])[0]])) + '\n')
    return True
