#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
import grpc
import numpy as np
from tensorflow_serving.apis import model_service_pb2_grpc, model_management_pb2, get_model_status_pb2, predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.config import model_server_config_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import os
import io
gpu_num=0
from functools import wraps
import tensorflow as tf
import base64
import cv2
from .utils import *
from .config import *
from .processResult import *


thresh = 0.3  #设置阈值

anchors = get_anchors() #获取anchors

def cardPredict(image):

    if (imgsize, imgsize) != (None, None):
        assert (imgsize, imgsize)[0] % 32 == 0, 'Multiples of 32 required'
        assert (imgsize, imgsize)[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed((imgsize, imgsize))))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    #    print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    channel = grpc.insecure_channel(serving_config['hostport'],
                                    options=[('grpc.max_send_message_length', serving_config['max_message_length']), (
                                        'grpc.max_receive_message_length', serving_config['max_message_length'])])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = serving_config['model_name']
    request.model_spec.signature_name = serving_config['signature_name']
    ### 注意：inputs["images"] 输入自己导出模型时的参数
    request.inputs['images'].CopyFrom(make_tensor_proto(
        image_data, dtype=dtypes.float32))
    predict_result = stub.Predict(request, serving_config['timeout'])

    boxes_, scores_, classes_ = getResult(predict_result) #引入结果处理程序

    dic = {}
    sess =tf.Session()
    # num = len([i for i in list(sess.run(scores_)) if i >= thresh])  # session
    scores = sess.run(scores_).tolist()
    maxindex = scores.index(max(scores))
    num = sess.run(classes_)  # session
    num = num[maxindex]
    tf.reset_default_graph() #释放内存
    sess.close()
    if len(num)==0:
        return ""
    return num[0]


def cardcal(path):
    """
    Args:
        path:输入必须是base64格式
    """
    de = base64.b64decode(path) #将base64转换成Image格式
    img = io.BytesIO(de)
    img = Image.open(img)
    count = cardPredict(image=img)
    dic = {}
    dic["card_num"] = str(count)
    return str(dic)


