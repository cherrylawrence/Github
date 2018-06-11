#!/usr/bin/env python
"""
definition of the loss functions
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

# loss
tf.app.flags.DEFINE_string('loss', 'softmax',
    """specifiy the loss function to use, softmax, mll_ce, mll_exp""")

# weighted_pairs
tf.app.flags.DEFINE_boolean('weighted_pairs', False,
     """whether to put weights on the ranking pairs""")

def _batch_gather(input, indices):
  """
  output[i, ..., j] = input[i, indices[i, ..., j]]
  """
  shape_output = indices.get_shape().as_list()
  shape_input = input.get_shape().as_list()
  assert len(shape_input) == 2
  batch_base = shape_input[1] * np.arange(shape_input[0])
  batch_base_shape = [1] * len(shape_output)
  batch_base_shape[0] = shape_input[0]

  batch_base = batch_base.reshape(batch_base_shape)
  indices = batch_base + indices

  input = tf.reshape(input, [-1])
  return tf.gather(input, indices)


class LossConfig(object):
  def __init__(self, name):
    self.name = name
    self.loss = None
    self.infer = lambda x: x
    self.logits_size = NUM_CLASSES

configs = {}

def loss(func):
  config = LossConfig(func.func_name)
  config.loss = func
  configs[config.name] = config
  return func

@loss
def softmax(logits, labels):
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels['random_label'])
  return tf.reduce_mean(xent)

def _get_M():
  alpha = [1./(i+1) for i in range(NUM_CLASSES)]
  alpha = np.cumsum(alpha)
  return alpha.astype(np.float32)

def _pairwise(label_pairs, logits):
  mapped = _batch_gather(logits, label_pairs)
  neg, pos = tf.split(mapped, 2, 2)
  delta = neg - pos

  neg_idx, pos_idx = tf.split(label_pairs, 2, 2)
  _, indices = tf.nn.top_k(tf.stop_gradient(logits), NUM_CLASSES)
  _, ranks = tf.nn.top_k(-indices, NUM_CLASSES)
  pos_ranks = _batch_gather(ranks, pos_idx)

  weights = _get_M()
  pos_weights = tf.gather(weights, pos_ranks)

  delta_nnz = tf.cast(tf.not_equal(neg_idx, pos_idx), tf.float32)
  return delta, delta_nnz, pos_weights

@loss
def mll_exp(logits, labels):
  # compute label pairs
  # batch_size x num_pairs x 2
  delta, delta_nnz, pos_weights = _pairwise(labels['label_pair'], logits)

  delta_max = tf.reduce_max(delta, 1, keep_dims=True)
  delta_max_nnz = tf.nn.relu(delta_max)

  inner_exp_diff = tf.exp(delta - delta_max_nnz)
  inner_exp_diff *= delta_nnz

  if FLAGS.weighted_pairs:
    inner_exp_diff *= pos_weights

  inner_sum = tf.reduce_sum(inner_exp_diff, 1, keep_dims=True)

  loss = delta_max_nnz + tf.log(tf.exp(-delta_max_nnz) + inner_sum)
  return tf.reduce_mean(loss)

@loss
def mll_bp(logits, labels):
  """
  Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with applications to
  functional genomics and text categorization." IEEE transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.
  """
  # compute label pairs
  # batch_size x num_pairs x 2
  delta, delta_nnz, pos_weights = _pairwise(labels['label_pair'], logits)
  delta = tf.exp(delta)
  inner_sum = tf.reduce_sum(delta, 1)
  return tf.reduce_mean(inner_sum)

margin = 1.0
@loss
def mll_rank(logits, labels):
  delta, delta_nnz, pos_weights = _pairwise(labels['label_pair'], logits)
  delta = tf.nn.relu(margin + delta)
  delta *= delta_nnz

  if FLAGS.weighted_pairs:
    delta *= pos_weights
  return tf.reduce_mean(tf.reduce_sum(delta, 1))

@loss
def mll_ce(logits, labels):
  return tf.reduce_mean(ops.cross_entropy(logits,
    labels['label_map'], name='mll_ce'))

def load_loss(loss_name = None):
  if loss_name is None:
    loss_name = FLAGS.loss
  return configs[loss_name]

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
