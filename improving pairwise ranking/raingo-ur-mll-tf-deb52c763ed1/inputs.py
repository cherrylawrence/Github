#!/usr/bin/env python
"""
read the tf record, supply the images and labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import tensorflow as tf
import cnns

from compile_data import NUM_CLASSES

MAX_PAIRS = 1000

def _lm2lp(label_map):
  pos = tf.reshape(tf.where(label_map), [-1])
  neg = tf.reshape(tf.where(tf.logical_not(label_map)), [-1])

  neg_pos = tf.meshgrid(neg, pos, indexing='ij')
  neg_pos_mat = tf.reshape(tf.transpose(tf.stack(neg_pos)), [-1, 2])
  neg_pos_rand = tf.random_shuffle(neg_pos_mat)
  neg_pos_pad = tf.pad(neg_pos_rand, [[0, MAX_PAIRS], [0, 0]])
  neg_pos_res = tf.slice(neg_pos_pad, [0,0], [MAX_PAIRS, -1])

  # MAX_PAIRS x 2
  return neg_pos_res

def _parse_example_proto(example_serialized):
  # parse record
  # decode jpeg
  # compute the length of the caption
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
      'image/path': tf.FixedLenFeature([], dtype=tf.string),
      'labels': tf.VarLenFeature(dtype=tf.int64),
  }

  features = tf.parse_single_example(example_serialized, feature_map)

  image = tf.image.decode_jpeg(
      features['image/encoded'],
      channels=3,
      try_recover_truncated=True)

  labels = features['labels']

  label_map = tf.sparse_to_indicator(labels, NUM_CLASSES)
  label_map.set_shape([NUM_CLASSES])

  label_pairs = _lm2lp(label_map)
  label_map = tf.cast(label_map, tf.float32)

  labels = tf.sparse_tensor_to_dense(labels)
  random_label = tf.random_shuffle(labels)[0]

  labels = {}
  labels['label_map'] = label_map
  labels['label_pair'] = label_pairs
  labels['random_label'] = random_label
  labels['image'] = image
  labels['path'] = features['image/path']

  return labels

from cnns import utils
inputs = lambda is_train, split, batch_size, _CNN=None: \
    utils.inputs(is_train, split, _parse_example_proto, batch_size, _CNN)

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
