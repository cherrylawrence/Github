#!/usr/bin/env python
"""
convert txt file of the format:
  image-path label-id
into tfrecords

python compile_data.py txt-path

encoded image. regular jpg file
assume images are properly encoded jpg, although if otherwise it will be converted to jpeg

the directory data/txt-path-name.tf will be created
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import os

import random
import tensorflow as tf
import threading

def vocab_path():
  return osp.join(os.environ.get("DATA_DIR", '.'), 'vocab.txt')

def _num_classes():
  vocab = []
  with open(vocab_path()) as reader:
    for line in reader:
      vocab.append(line.strip())

  return vocab, len(vocab)

VOCAB, NUM_CLASSES = _num_classes()

from cnns import utils
def _convert_to_example(data):
  path = data[0]
  labels = data[1]
  image_buffer = utils.ensure_jpeg(path)
  example = tf.train.Example(
      features=tf.train.Features(feature={
      'image/path': utils.bytes_feature(path),
      'image/encoded': utils.bytes_feature(image_buffer),
      'labels': utils.int64_feature(labels)}))
  return example

def _parse(fields):
  path = fields[0]
  labels = [int(f) for f in fields[1].split(',')]
  if not osp.exists(path):
    return None
  return (path, labels)

def main(_):
  utils._compile_main(_convert_to_example, _parse)

if __name__ == "__main__":
  tf.app.run()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
