# keras_yolov3_mobilnet


### 0-背景
change the backend of darknet53 into 
- [x] Mobilenet
- [x] VGG16
- [x] ResNet101
- [x] ResNeXt101


### 1-training
1.制作自己的数据集

行形式：image_file_path box1 box2 ... boxN

框形式：x_min,y_min,x_max,y_max,class_id (no space).

转换成VOC格式数据：python voc_annotation.py

举例：

path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3

path/to/img2.jpg 120,300,250,600,2

 ...
 
2.运行下面脚本

python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5

3.开始训练

python train.py  

#### 3-log
tensorboard --logdir ./logs/carMobilenet/001_Mobilenet_finetune_02/

#### 4-test
python test_folder.py


#### 5-h5模型格式转换成tensorflow-serving格式
python h52pb.py -path "./*.h5" -num 2 -anchor 9 -export "./pb_folder"

#### 6-serving部署以及测试
仔细查阅tensorflow-serving

