# -*- encoding: utf-8 -*-
"""
@File    : server.py
@Time    : 2020/1/4 16:21
@Author  : Johnson
@Email   : 593956670@qq.com
"""

import grpc
import calculate_pb2
import calculate_pb2_grpc
import calculate
from concurrent import futures
import time


# 创建一个CalculateServicer 继承calculate_pb2_grpc.CalculateServicer
class CalculateServicer(calculate_pb2_grpc.CalculateServicer):
    def Square(self, request, context):
        response = calculate_pb2.result()
        response.value = calculate.cardcal(request.value)
        return response

# 创建gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
# 利用 add_CalculateServicer_to_server 这个方法把上面定义的 CalculateServicer 加到 server
calculate_pb2_grpc.add_CalculateServicer_to_server(CalculateServicer(), server)
# server跑在port 50051
print ('Starting server. Listening on port 5010.')
#server.add_insecure_port('[::]:5010')
server.add_insecure_port('0.0.0.0:5010')
server.start()

# 因为 server.start() 不会阻塞，添加睡眠循环以持续服务
try:
    while True:
        time.sleep(24 * 60 * 60)
except KeyboardInterrupt:
    server.stop(0)