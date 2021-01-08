#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import grpc
import numpy as np
from tensorflow_serving.apis import model_service_pb2_grpc, model_management_pb2, get_model_status_pb2, predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.config import model_server_config_pb2
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.core.framework import types_pb2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions



def square(path):
    serving_config = {
        "hostport": "localhost:8500",
        "max_message_length": 1000 * 1024 * 1024,
        "timeout": 30,
        "signature_name": "serving_default",
        "model_name": "cardmobile"
    }
    image = cv2.imread(path)
    img_resize = cv2.resize(image,(150, 150))
    img_resize = img_to_array(img_resize)
    img_resize = np.expand_dims(img_resize,axis=0)
    channel = grpc.insecure_channel(serving_config['hostport'], options=[('grpc.max_send_message_length', serving_config['max_message_length']), (
        'grpc.max_receive_message_length', serving_config['max_message_length'])])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = serving_config['model_name']
    request.model_spec.signature_name = serving_config['signature_name']
    ### 注意：inputs[""] 输入自己导出模型时的参数
    request.inputs['images'].CopyFrom(make_tensor_proto(
        img_resize, shape=[1,150,150,3]))
    result = stub.Predict(request, serving_config['timeout'])
    channel.close()
    prob = result.outputs["output"].float_val[0]
    return prob


print(square("/opt/zhongls/01.jpg"))