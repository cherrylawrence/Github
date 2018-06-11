#!/usr/bin/env python
"""
definition of the loss and inference for label count
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp

from cnns import ops
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

from compile_data import NUM_CLASSES
import numpy as np

# cnt_loss
tf.app.flags.DEFINE_string('cnt_loss', '',
    """specifiy the loss function to use to estimate label count""")

# cnt_features
tf.app.flags.DEFINE_string('cnt_features', 'features',
    """specify which features to use to estimate counts; features or logits""")

class LossConfig(object):
  def __init__(self, name):
    self.name = name
    self.loss = None
    self.infer = lambda x: x
    self.logits_size = NUM_CLASSES

configs = {}

def loss(func):
  loss, infer, lsize = func()
  config = LossConfig(func.func_name)
  config.infer = infer
  config.loss = loss
  config.logits_size = lsize
  configs[config.name] = config
  return func

@loss
def linear():

  def _loss(lcnt, label_count):
    loss = tf.square(label_count - lcnt)
    loss = tf.reduce_mean(loss)
    return loss

  def _infer(lcnt):
    return tf.nn.relu(lcnt)
  return _loss, _infer, 1

@loss
def poisson():

  def _loss(lcnt, label_count):
    label_count = tf.expand_dims(label_count, axis=1)
    loss = tf.nn.log_poisson_loss(label_count, lcnt)
    loss = tf.reduce_mean(loss)
    return loss

  def _infer(lcnt):
    return tf.exp(lcnt)

  return _loss, _infer, 1


@loss
def log():

  def _loss(lcnt, label_count):
    label_count = tf.log(label_count)
    loss = tf.square(label_count - lcnt)
    loss = tf.reduce_mean(loss)
    return loss

  def _infer(lcnt):
    return tf.exp(lcnt)

  return _loss, _infer, 1

@loss
def bins():

  num_bins = 4

  def _loss(lcnt, label_count):
    tails = num_bins * tf.ones_like(label_count)
    bins = tf.where(label_count > num_bins, tails, label_count)
    labels = bins - 1
    labels = tf.cast(labels, tf.int64)

    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=lcnt, labels=labels)
    return tf.reduce_mean(xent)

  def _infer(lcnt):
    return tf.argmax(lcnt, 1) + 1

  return _loss, _infer, num_bins

def load_cnt(loss_name = None):
  if loss_name is None:
    loss_name = FLAGS.cnt_loss
  return configs[loss_name]

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
