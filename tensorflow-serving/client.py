# -*- encoding: utf-8 -*-
"""
@File    : client.py
@Time    : 2020/1/4 16:31
@Author  : Johnson
@Email   : 593956670@qq.com
"""
import grpc
import calculate_pb2
import calculate_pb2_grpc
import base64

def client(path):
    """
    Args:
        path: 输入图片路径
    """
    with open(path,'rb') as f:
        base64_data = base64.b64encode(f.read())

    # 打开 gRPC channel，连接到 localhost:50051
    channel = grpc.insecure_channel('localhost:5010')
    # 创建一个 stub (gRPC client)
    stub = calculate_pb2_grpc.CalculateStub(channel)
    # 创建一个有效的请求消息 Number
    result = calculate_pb2.Cardpred(value=base64_data)
    # 带着 Number 去调用 Square
    response = stub.Square(result)
    print (response.value)
    
    
if __name__ == '__main__':
    client("test.jpg")