# -*- encoding: utf-8 -*-
"""
@File    : idcard_identity.py
@Time    : 2020/07/10 16:50
@Author  : Johnson
@Email   : 593956670@qq.com
"""
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils.utils import get_anchors,get_classes,create_model,create_tiny_model,data_generator_wrapper
from config import *
import os
from yolo_Mobilenet import YOLO
from PIL import Image


class IDCardIdentify(object):
    def __init__(self):
        pass
    def train(self,classes_path,anchors_path):
        '''
        Args:
            classes_path:classes路径
            anchors_path:anchor路径
        '''
        classes_names = get_classes(classes_path)
        num_classes = len(classes_names)
        anchors = get_anchors(anchors_path)

        is_tiny_version = len(anchors) == 6  # default setting
        if is_tiny_version:
            model = create_tiny_model(input_shape, anchors, num_classes, freeze_body=2)
        else:
            model = create_model(input_shape, anchors, num_classes, load_pretrained=False)

        logging = TensorBoard(log_dir=log_dir)
        # checkpoint = ModelCheckpoint(log_dir + 'car_mobilenet_yolov3.ckpt',
        #    monitor='val_loss', save_weights_only=False, period=1)
        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=False, save_best_only=True, period=3)

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, min_lr=1e-9, patience=5, verbose=1)
        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        with open(train_path) as t_f:
            t_lines = t_f.readlines()
        np.random.seed(666)
        np.random.shuffle(t_lines)
        num_val = int(len(t_lines) * val_split)
        num_train = len(t_lines) - num_val
        t_lines = t_lines[:num_train]
        v_lines = t_lines[num_train:]

        # train with frozen layers first ,to get a stable loss.
        # adjust num epochs to your dataset,This step is enough to obtrain a not bad model
        if True:
            model.compile(optimizer=Adam(lr=1e-3), loss={
                # use custom yolo_loss Lambda layer.
                'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_num))
        model.fit_generator(data_generator_wrapper(t_lines, batch_num, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_num),
                            validation_data=data_generator_wrapper(v_lines, batch_num, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_num),
                            epochs=epochs,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save(log_dir + 'trained_weights_stage_1.h5')

        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if True:
            print("Unfreeze and continue training, to fine-tune.")
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=1e-4),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
            batch_size = 16  # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,
                                                                                       batch_size))
            model.fit_generator(data_generator_wrapper(t_lines, batch_size, input_shape, anchors, num_classes),
                                steps_per_epoch=max(1, num_train // batch_size),
                                validation_data=data_generator_wrapper(v_lines, batch_size, input_shape, anchors,
                                                                       num_classes),
                                validation_steps=max(1, num_val // batch_size),
                                epochs=20,
                                initial_epoch=0,
                                callbacks=[logging, checkpoint, reduce_lr])
            model.save(log_dir + 'trained_weights_final.h5')

    def predict(self,yolo,test_dir,target_dir):
        """
        Args:
            yolo:加载模型框架
            test_dir:预测图片路径
            target_dir:保存路径
        return:

        """
        pic_temp = []
        pic = os.listdir(test_dir)
        for name in pic:
            pic_temp.append(name)
        for i in range(len(pic_temp)):
            img = test_dir + '/' + pic_temp[i]
            print('the pic is {}'.format(pic_temp[i]))
            image = Image.open(img)
            detect = yolo.detect_image(image)
            #         detect.show()
            detect.save(target_dir + '/' + pic_temp[i])
        #     yolo.close_session()
        return detect, img

