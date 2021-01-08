# -*- encoding: utf-8 -*-
"""
@File    : yolo_Mobilenet.py
@Time    : 2020/07/07 19:26
@Author  : Johnson
@Email   : 593956670@qq.com
"""
import colorsys
import os
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.layers import Input
from PIL import Image,ImageFont,ImageDraw
from yolo3.model_mobilenet import yolo_eval,yolo_body,tiny_yolo_body
from utils.utils import letterbox_image
import os
from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.utils import multi_gpu_model
gpu_num = 1

class YOLO(object):
    def __init__(self):
        self.model_path = model_path
        self.anchor_path = anchors_path
        self.classes_path = classes_path
        self.score = score
        self.iou = iou
        self.classes_names  = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = input_shape #输入图片大小
        self.boxes,self.scores,self.classes = self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    
    def generate(self):
        """to generate the bounging boxes"""
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5") #保证是h5格式文件
        
        #加载模型
        num_anchors = len(self.anchors)
        num_classes = len(self.classes_names)
        
        print("**********classes***********",num_anchors)
        is_tiny_version = num_anchors==6 #default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
            print("*********************success*********************")
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        #generate colors for drawing bounding boxes
        hsv_tuples = [(x/len(self.classes_names)) for x in range(len(self.classes_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        #generate output tensor targets fro filters bounding boxes
        self.input_image_shape = K.placeholder(shape=(2,))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model,gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        # default arg
        # self.yolo_model->'model_data/yolo.h5'
        # self.anchors->'model_data/yolo_anchors.txt'-> 9 scales for anchors
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        rects = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # tf.Session.run(fetches, feed_dict=None)
        # Runs the operations and evaluates the tensors in fetches.
        #
        # Args:
        # fetches: A single graph element, or a list of graph elements(described above).
        #
        # feed_dict: A dictionary that maps graph elements to values(described above).
        #
        # Returns:Either a single value if fetches is a single graph element, or a
        # list of values if fetches is a list(described above).
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            y1, x1, y2, x2 = box
            y1 = max(0, np.floor(y1 + 0.5).astype('float32'))
            x1 = max(0, np.floor(x1 + 0.5).astype('float32'))
            y2 = min(image.size[1], np.floor(y2 + 0.5).astype('float32'))
            x2 = min(image.size[0], np.floor(x2 + 0.5).astype('float32'))

            # image = image.crop((x1, y1, x2, y2))  #该代码可将检测的目标从图像中裁剪出来

            print(label, (x1, y1), (x2, y2))
            bbox = dict([("score", str(score)), ("x1", str(x1)), ("y1", str(y1)), ("x2", str(x2)), ("y2", str(y2))])
            rects.append(bbox)

            if y1 - label_size[1] >= 0:
                text_origin = np.array([x1, y1 - label_size[1]])
            else:
                text_origin = np.array([x1, y1 + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [x1 + i, y1 + i, x2 - i, y2 - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        #
        end = timer()
        print(str(end - start))
        return image

    def close_session(self):
        self.sess.close()

    def detect_video(yolo, video_path, output_path=""):
        import cv2
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            image = yolo.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        yolo.close_session()




