#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import datetime
import warnings
import json

try:
    from inspect import signature
except:
    warnings.warn('inspect.signature not available... '
        'you should upgrade to Python 3.x')


def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")


def mkdir_if_not_exist(dirs, is_delete=False):
    """
    创建文件夹
    :param dirs: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        for dir_ in dirs:
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
