# -*- encoding: utf-8 -*-
"""
@File    : processResult.py
@Time    : 2020/07/10 10:58
@Author  : Johnson
@Email   : 593956670@qq.com
"""
from .utils import *
from .config import *

def getResult(result):

    shape1 = (
        predict_result.outputs["output0"].tensor_shape.dim[1].size,
        predict_result.outputs["output0"].tensor_shape.dim[2].size,
        predict_result.outputs["output0"].tensor_shape.dim[3].size)
    shape2 = (
        predict_result.outputs["output1"].tensor_shape.dim[1].size,
        predict_result.outputs["output1"].tensor_shape.dim[2].size,
        predict_result.outputs["output1"].tensor_shape.dim[3].size)
    shape3 = (
        predict_result.outputs["output2"].tensor_shape.dim[1].size,
        predict_result.outputs["output2"].tensor_shape.dim[2].size,
        predict_result.outputs["output2"].tensor_shape.dim[3].size)

    yolos = [np.array(predict_result.outputs["output0"].float_val).reshape(shape1),
             np.array(predict_result.outputs["output1"].float_val).reshape(shape2),
             np.array(predict_result.outputs["output2"].float_val).reshape(shape3)]

    # yolos = [np.array(predict_result.outputs["output0"].float_val).reshape(10, 10, 21),
    #          np.array(predict_result.outputs["output1"].float_val).reshape(20, 20, 21),
    #          np.array(predict_result.outputs["output2"].float_val).reshape(40, 40, 21)]

    boxes_, scores_, classes_ = yolo_eval(yolos, anchors, class_num, [imgsize, imgsize], max_boxes=max_boxes,
                                          score_threshold=thresh,
                                          iou_threshold=iou_threshold)

    return boxes_, scores_, classes_
