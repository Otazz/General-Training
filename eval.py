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

import argparse
import sys
import time
import os

import numpy as np
import tensorflow as tf

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

if __name__ == "__main__":
  file_name = "tf_files/flower_photos/classify"
  model_file = "a.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  batch_size = 32

  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--batch", type=int, help="batch_size")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")

  #parser.add_argument("--")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  file_name = 'a.jpg'
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  if args.batch:
    batch_size = args.batch

  input_poets = "import/input"
  output_poets = "import/InceptionV3/Predictions/Reshape_1"
  labels = load_labels('dir/labels.txt')
  graph_poets = load_graph('a.pb')


  images = ['a.jpg']
  for image_p in images:
    t = read_tensor_from_image_file(image_p)

    input_operation= graph_poets.get_operation_by_name(input_poets);
    output_operation = graph_poets.get_operation_by_name(output_poets);
    print(output_operation.name)
    with tf.Session(graph=graph_poets) as sess:
      results1 = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]:t
                         })
    results1 = np.squeeze(results1)
    top_k = results1.argsort()[-5:][::-1]

    print("Imagem classificada como:")
    for i in top_k:
      print(labels[i], results1[i])

'''python -m scripts.label_image --image=tf_files\flower_photos\classify\144099102_bf63a41e4f_n.jpg --input_layer=fifo_queue_Dequeue --output_layer=InceptionV3/Predictions/Reshape_1 --graph=result1.pb --labels=tf_files\labels.txt --input_mean=0 --input_std=255 --input_height=299 --input_width=299'''
