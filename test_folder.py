# -*- encoding: utf-8 -*-
"""
@File    : test_folder.py
@Time    : 2020/07/07 19:24
@Author  : Johnson
@Email   : 593956670@qq.com
"""
from yolo_Mobilenet import YOLO
from idcard_identity import IDCardIdentify
import os
from config import *

def detect_img(yolo,test_dir,target_dir):

    identity = IDCardIdentify()
    detect,img = identity.predict(yolo,test_dir,target_dir)
    return detect, img


if __name__ == '__main__':
    if os.path.exists(target_dir): ###测试路径以及结果保存路径注意修改为自己的路径
        print('File exists !!!')
    else:
        os.mkdir(target_dir)

    detect_img(YOLO(),test_dir,target_dir)

