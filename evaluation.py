# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def main(image):
  model_file = "a.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  batch_size = 32

  file_name = image

  input_poets = "import/input"
  output_poets = "import/InceptionV3/Predictions/Reshape_1"
  labels = load_labels('dir/labels.txt')
  graph_poets = load_graph('a.pb')

  t = read_tensor_from_image_file(image)

  input_operation= graph_poets.get_operation_by_name(input_poets);
  output_operation = graph_poets.get_operation_by_name(output_poets);

  with tf.Session(graph=graph_poets) as sess:
    results1 = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]:t
                       })
  results1 = np.squeeze(results1)
  top_k = results1.argsort()[-5:][::-1]

  return {'image_name': image,
          'result': {labels[i][2:]: float(results1[i]) for i in top_k}
        }

'''python -m scripts.label_image --image=tf_files\flower_photos\classify\144099102_bf63a41e4f_n.jpg --input_layer=fifo_queue_Dequeue --output_layer=InceptionV3/Predictions/Reshape_1 --graph=result1.pb --labels=tf_files\labels.txt --input_mean=0 --input_std=255 --input_height=299 --input_width=299'''
