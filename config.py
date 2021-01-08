# -*- encoding: utf-8 -*-
"""
@File    : config.py
@Time    : 2020/07/07 14:56
@Author  : Johnson
@Email   : 593956670@qq.com
"""
#################训练参数####################
batch_num = 16 #batch_size
epochs = 500 #
input_shape = (416,416)

train_path = 'train.txt' #训练集
val_path = 'val.txt' #验证集
log_dir = 'logs/carMobilenet/001_Mobilenet_finetune_03/' #模型保存路径
classes_path = 'model_data/voc_classes.txt' # classes路径
anchors_path = 'model_data/yolo_anchors.txt' #anchors路径

#数据验证集比例,360
val_split = 0.3



#################测试参数####################

# 最终模型保存路径
model_path = log_dir+'/ep456-loss4.194-val_loss3.792.h5'

# 测试数据路径
test_dir = "./test_image"

#测试结果保存路径
target_dir = "./results"


#预测参数
score = 0.3 #边框得分
iou = 0.5 #IOU值






