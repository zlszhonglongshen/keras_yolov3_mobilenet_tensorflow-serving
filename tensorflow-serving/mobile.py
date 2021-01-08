def personNum(image):
    serving_config = {
        "hostport": "192.168.33.2:8500",
        "max_message_length": 1000 * 1024 * 1024,
        "timeout": 300,
        "signature_name": "serving_default",
        "model_name": "cardmobile"
    }
    # image = Image.open("8.jpg")
    # with open("001.jpg",'rb') as f:base64_data = base64.b64encode(f.read())

    shape = image.size
    # image = image.resize((416, 416))

    if (320, 320) != (None, None):
        assert (320, 320)[0] % 32 == 0, 'Multiples of 32 required'
        assert (320, 320)[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed((320, 320))))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    #    print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    channel = grpc.insecure_channel(serving_config['hostport'],
                                    options=[('grpc.max_send_message_length', serving_config['max_message_length']), (
                                        'grpc.max_receive_message_length', serving_config['max_message_length'])])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = serving_config['model_name']
    request.model_spec.signature_name = serving_config['signature_name']
    ### 注意：inputs[""] 输入自己导出模型时的参数
    request.inputs['images'].CopyFrom(make_tensor_proto(
        image_data, dtype=dtypes.float32))
    predict_result = stub.Predict(request, serving_config['timeout'])
    yolos = [np.array(predict_result.outputs["output0"].float_val).reshape(10, 10, 21),
             np.array(predict_result.outputs["output1"].float_val).reshape(20, 20, 21),
             np.array(predict_result.outputs["output2"].float_val).reshape(40, 40, 21)]

    boxes_, scores_, classes_ = yolo_eval(yolos, anchors, 2, [320,320], max_boxes=20, score_threshold=thresh,
                                          iou_threshold=.5)
    dic = {}
    sess =tf.Session()
    # num = len([i for i in list(sess.run(scores_)) if i >= thresh])  # session
    num = sess.run(classes_)  # session
    tf.reset_default_graph()
    sess.close()
    if len(num)==0:
        return ""
    return num[0]