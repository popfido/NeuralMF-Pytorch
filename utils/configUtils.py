#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

from __future__ import print_function
from __future__ import absolute_import

import argparse
import json

import os
from bunch import Bunch

from utils.utils import mkdir_if_not_exist


def get_config_from_json(json_file):
    """
    Parse Config File to config class
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # 配置字典

    config = Bunch(config_dict)  # 将配置字典转换为类

    return config, config_dict


def process_config(json_file):
    """
    Parse Json file
    :param json_file: Json config file
    :return: config
    """
    config, _ = get_config_from_json(json_file)
    config.tb_dir = os.path.join("experiments", config.exp_name, "logs/")  # 日志
    config.cp_dir = os.path.join("experiments", config.exp_name, "checkpoints/")  # 模型
    config.img_dir = os.path.join("experiments", config.exp_name, "images/")  # 网络

    mkdir_if_not_exist([config.tb_dir, config.cp_dir, config.img_dir])  # 创建文件夹
    return config


def save_config(config, run_dir, timestamp):
    path = os.path.join(run_dir, "config_{}.json".format(timestamp))
    with open(path, 'w') as config_file:
        json.dump(config, config_file)
        config_file.write('\n')


def get_train_args():
    """
    Train Parameter Getter
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='path',
        default='None',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser


def get_test_args():
    """
    Test parameter Getter
    :return: 参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='C',
        default='None',
        help='add a configuration file')
    parser.add_argument(
        '-m', '--mod',
        dest='model',
        metavar='',
        default='None',
        help='add a model file')
    args = parser.parse_args()
    return args, parser
