#!/usr/bin/env python
"""
multi label image classification
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import tensorflow as tf
import cnns
from cnns import runner
from cnns import ops
from cnns import optim

from loss import load_loss, NUM_CLASSES
from inputs import inputs
import time
import metrics
import numpy as np
from functools import partial
from cnt import load_cnt
import random

FLAGS = tf.app.flags.FLAGS

# thresh_features
tf.app.flags.DEFINE_string('thresh_features', '',
    """specify which features to use to estimate counts; features or logits""")

def get_iter(split, is_train, batch_size):
  CNN = cnns.load_cnn()
  LOSS = load_loss()

  with tf.name_scope("inputs"):
    data, epoch_size = inputs(is_train, split, batch_size, _CNN=CNN)
    images = data['image']

  cnn_is_train = is_train and len(FLAGS.cnt_loss) == 0 and len(FLAGS.thresh_features) == 0

  with tf.variable_scope("classifier"):
    cnn = CNN(images, cnn_is_train)
    cnn_update = cnn.update_ops
    features = cnn.features
    features = tf.nn.dropout(features, .5) if cnn_is_train else features
    logits = ops.fc_layer(features, LOSS.logits_size, 'fc8-nus')

  if is_train:
    with tf.name_scope("loss", values=data.values()+ [logits]):
      loss = LOSS.loss(logits, data)
  else:
    logits = LOSS.infer(logits)
    outputs_T = {'logits':logits, 'label_map':data['label_map'], 'path':data['path']}

  features_map = {'features':features,
      'logits':logits, 'concat':tf.concat([features, logits], 1)}

  if len(FLAGS.thresh_features) > 0:
    with tf.variable_scope('threshold') as vs:
      features = tf.stop_gradient(features_map[FLAGS.thresh_features])
      h1 = tf.nn.relu(ops.fc_layer(features, NUM_CLASSES * 4, 'thresh-h1'))
      h2 = tf.nn.relu(ops.fc_layer(h1, NUM_CLASSES * 2, 'thresh-h2'))
      thresh = ops.fc_layer(h2, NUM_CLASSES, 'thresh')
      logits = thresh+tf.stop_gradient(logits)

      if is_train:
        loss = tf.reduce_mean(ops.cross_entropy(logits,
          data['label_map'], name='thresh-loss'))
      else:
        outputs_T['logits'] = logits
        outputs_T['thresh'] = thresh

  if len(FLAGS.cnt_loss) > 0:
    CNT = load_cnt()
    with tf.variable_scope("count") as vs:
      features = tf.stop_gradient(features_map[FLAGS.cnt_features])
      h1 = tf.nn.relu(ops.fc_layer(features, 100, 'cnt-h1'))
      h2 = tf.nn.relu(ops.fc_layer(h1, 10, 'cnt-h2'))

      lcnt = ops.fc_layer(h2, CNT.logits_size, 'cnt')

      label_count = tf.reduce_sum(data['label_map'], 1)
      if is_train:
        loss = CNT.loss(lcnt, label_count)
      else:
        outputs_T['lcnt'] = CNT.infer(lcnt)
        outputs_T['label-count'] = label_count

  def _norm_cnt_lst(lst):
    return np.round(lst.flatten()).astype(np.int32).tolist()

  def _eval(outputs_cat):
    if 'lcnt' in outputs_cat:
      lcnts = _norm_cnt_lst(outputs_cat['lcnt'])
      gt = _norm_cnt_lst(outputs_cat['label-count'])
      cmb = zip(lcnts, gt)
      random.shuffle(cmb)
      print('lcnt', *cmb[:5])

    metric, score = metrics.evaluate(outputs_cat)
    metric['num_samples'] = outputs_cat.values()[0].shape[0]
    metric['timestamp'] = time.time()

    return metric, score

  if is_train:
    loss_list = [(loss, 1, None)]
    return epoch_size, loss_list, cnn
  else:
    return epoch_size, outputs_T, _eval

def main(_):
  runner.main('classifier', get_iter)

if __name__ == "__main__":
  tf.app.run(main=main)

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
