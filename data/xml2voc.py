# -*- encoding: utf-8 -*-
"""
@File    : xml2voc.py
@Time    : 2020/07/07 19:20
@Author  : Johnson
@Email   : 593956670@qq.com
"""
import xml.etree.ElementTree as ET
from os import getcwd

sets=['train'] #修改
classes = ["zheng","fan"] #修改


def convert_annotation(image_id, list_file):
    in_file = open('/opt/zhongls/object_detect/keras-YOLOv3-mobilenet-master/card/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for  image_set in sets:
    image_ids = open('/opt/zhongls/object_detect/keras-YOLOv3-mobilenet-master/card/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s.txt'%(image_set), 'w')
    for index in image_ids:
        image_id = index.split("/")[-1].split(".")[0]
        image_path = index.split("/")[-1]
#         print(image_path)
        list_file.write('%s/card/JPEGImages/%s'%(wd,image_path))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
