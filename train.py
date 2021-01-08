# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2020/07/07 14:29
@Author  : Johnson
@Email   : 593956670@qq.com
"""
from idcard_identity import IDCardIdentify
from config import *

def train():
    identify= IDCardIdentify()
    TRAIN = identify.train(classes_path,anchors_path)


if __name__ == '__main__':
    train()







