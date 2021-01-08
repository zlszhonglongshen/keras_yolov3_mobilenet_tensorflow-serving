### GRPC服务端口设置，根据自己的实际情况进行修改
serving_config = {
        "hostport": "192.168.33.2:8500",
        "max_message_length": 1000 * 1024 * 1024,
        "timeout": 300,
        "signature_name": "serving_default",
        "model_name": "cardmobile"
    }

# 输入size,必须是32的倍数，建议：320,416,608
imgsize = 320

# 分类数目
class_num = 2

# 预测其他超参数
max_boxes = 20
score_threshold = 0.1
iou_threshold = .5

