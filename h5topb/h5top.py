# -*- encoding: utf-8 -*-
"""
@File    : h5top.py
@Time    : 2020/07/07 19:21
@Author  : Johnson
@Email   : 593956670@qq.com
"""
######模型转换
from keras import backend as K
import tensorflow as tf
from tensorflow.python import saved_model
from tensorflow.python.saved_model.signature_def_utils_impl import (
    build_signature_def, predict_signature_def
)
from keras.models import load_model
from yolo3.model_Mobilenet import yolo_eval, yolo_body, tiny_yolo_body
import shutil
import os
from keras.layers import Input
import keras
from keras.utils.generic_utils import CustomObjectScope
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #(or "1" or "2")


# model_path = '/opt/zhongls/object_detect/keras-YOLOv3-mobilenet-master/logs/carMobilenet/001_Mobilenet_finetune/ep120-loss7.479-val_loss6.658.h5'
model_path ='/opt/zhongls/object_detect/keras-YOLOv3-mobilenet-master/logs/carMobilenet/001_Mobilenet_finetune_03/ep456-loss4.194-val_loss3.792.h5'

num_anchors = 9 #len(anchor)
num_classes = 2 #类别数，替换成自己的类别数

is_tiny_version = num_anchors==6 # default setting
try:
    yolo_model = load_model(model_path, compile=False)
except:
    yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
    yolo_model.load_weights(model_path) # make sure model, anchors and classes match

model = yolo_model

export_path = "model/card/2"

if os.path.isdir(export_path):
    shutil.rmtree(export_path)
builder = saved_model.builder.SavedModelBuilder(export_path)

signature = predict_signature_def(
    inputs={'images': model.input},
    outputs={
        'output0': model.output[0],
        'output1': model.output[1],
        'output2': model.output[2]
    }
)

sess = K.get_session()
builder.add_meta_graph_and_variables(sess=sess,
                                     tags=[saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict': signature,
                                                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                            signature})
builder.save()


