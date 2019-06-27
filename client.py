#!/usr/bin/env python


import json
import requests
import argparse
import base64
import time
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=argparse.FileType("rb"))
parser.add_argument("-m", "--model-name", required=True, choices={
    "vgg16", "resnet50", "inception_v3", "xception", "mobilenet", "mobilenet_v2",
    })
args = parser.parse_args()

# URL = "https://httpbin.org/post"
URL = "http://localhost:8501/v1/models/{}:predict".format(args.model_name)


data = args.filename.read()

jpeg_bytes = base64.b64encode(data).decode("utf-8")
predict_request = json.dumps({"instances": [{"b64": jpeg_bytes}]})


start_time = time.time()
r = requests.post(URL, data=predict_request)
end_time = time.time()


result = r.json()
predictions = np.array(result["predictions"])

result = tf.keras.applications.vgg16.decode_predictions(predictions)
print(result)
print("Time taken: {} s".format(end_time - start_time))
