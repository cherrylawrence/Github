#!/usr/bin/env python
"""
testing setup for mll models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

import loss
import inputs
import cnns
import metrics
import numpy as np

batch_size = 10
num_pairs = 100

def gen_data(logits_size = loss.NUM_CLASSES):
  logits = tf.random_normal([batch_size, logits_size])
  labels = {}
  labels['random_label'] = tf.random_uniform([batch_size],
      maxval=logits_size, dtype=tf.int64)
  labels['label_pair'] = tf.random_uniform([batch_size, num_pairs, 2],
      maxval=logits_size, dtype=tf.int64)
  lm = tf.random_uniform([batch_size, logits_size],
      maxval=2, dtype=tf.int64)
  labels['label_map'] = tf.cast(lm, tf.float32)
  return logits, labels


class TestLoss(object):
  _CONFIG = None
  def test_shape(self):
    logits, labels = gen_data(self._CONFIG.logits_size)
    loss = self._CONFIG.loss(logits, labels)
    with self.test_session():
      loss_ = loss.eval()
      self.assertEqual(loss_.shape, ())

# common test for loss
for name in loss.configs.keys():
  test_cls = type('Test_Loss_Common_%s' % name,
      (TestLoss, tf.test.TestCase), {'_CONFIG':loss.load_loss(name)})
  globals()[test_cls.__name__] = test_cls

class TestExpLoss(tf.test.TestCase):

  def setup(self):
    FLAGS.weighted_pairs = False

  def test_batch_gather(self):

    logits, labels = gen_data()
    lp = labels['label_pair']
    mapped = loss._batch_gather(logits, lp)
    with self.test_session() as sess:
      logits_, labels_, mapped_ = sess.run([
        logits, lp, mapped])
      for i in range(batch_size):
        self.assertAllEqual(logits_[i][labels_[i]], mapped_[i])

  def test_lm2lp(self):
    _, labels = gen_data()
    lm0 = labels['label_map'][0, :]
    lm = tf.greater(lm0, 0)
    lp = inputs._lm2lp(lm)
    with self.test_session() as sess:
      lm_, lp_ = sess.run([lm, lp])
      self.assertEqual(lp_.shape[1], 2)
      nnz = 0
      for i in range(lp_.shape[0]):
        if lp_[i, 0] != lp_[i, 1]:
          nnz += 1
          self.assertFalse(lm_[lp_[i, 0]])
          self.assertTrue(lm_[lp_[i, 1]])
      self.assertLess(0, nnz)

  def test_mll_exp(self):
    logits, labels = gen_data()
    loss_exp = loss.mll_exp(logits, labels)
    lp = labels['label_pair']
    with self.test_session() as sess:
      lp_, loss_, logits_ = sess.run([lp, loss_exp, logits])
      loss_gt = 0
      for i in range(batch_size):
        neg_idx = lp_[i, :, 0]
        pos_idx = lp_[i, :, 1]
        diff = [0]
        for neg, pos in zip(neg_idx, pos_idx):
          if neg != pos:
            diff.append(logits_[i, neg] - logits_[i, pos])
        max_diff = max(diff)
        this_loss = max_diff + np.log(sum([np.exp(f-max_diff) for f in diff]))
        loss_gt += this_loss
      self.assertAllCloseAccordingToType(loss_, loss_gt/batch_size)

  def test_mll_rank(self):
    logits, labels = gen_data()
    loss_exp = loss.mll_rank(logits, labels)
    lp = labels['label_pair']
    with self.test_session() as sess:
      lp_, loss_, logits_ = sess.run([lp, loss_exp, logits])
      loss_gt = 0
      for i in range(batch_size):
        neg_idx = lp_[i, :, 0]
        pos_idx = lp_[i, :, 1]
        diff = [0]
        for neg, pos in zip(neg_idx, pos_idx):
          if neg != pos:
            delta = logits_[i, neg] - logits_[i, pos] + loss.margin
            delta = 0 if delta < 0 else delta
            diff.append(delta)
        loss_gt += sum(diff)
      self.assertAllCloseAccordingToType(loss_, loss_gt/batch_size)

  def test_mll_warp(self):
    FLAGS.weighted_pairs = True
    logits, labels = gen_data()
    loss_exp = loss.mll_rank(logits, labels)
    lp = labels['label_pair']

    alpha = loss._get_M()

    with self.test_session() as sess:
      lp_, loss_, logits_ = sess.run([lp, loss_exp, logits])
      loss_gt = 0
      for i in range(batch_size):
        neg_idx = lp_[i, :, 0]
        pos_idx = lp_[i, :, 1]
        diff = [0]
        ranks = np.argsort(-logits_[i, :]).tolist()
        for neg, pos in zip(neg_idx, pos_idx):
          if neg != pos:
            rank = ranks.index(pos)
            weight = alpha[rank]

            delta = logits_[i, neg] - logits_[i, pos] + loss.margin
            delta = 0 if delta < 0 else delta
            diff.append(delta * weight)
        loss_gt += sum(diff)
      self.assertAllCloseAccordingToType(loss_, loss_gt/batch_size)


  def _gen_metric_data(self):
    ranks = np.array([[3,2,1],[1,2,3],[2,3,1]])-1
    labels = np.array([[1,2],[1,3],[2,3]])-1
    labelset = []
    for label in labels:
      labelset.append(set(label.tolist()))
    return ranks, labelset

  def test_violations(self):
    ranks, labels = self._gen_metric_data()
    res = metrics.compute_num_violations(ranks, labels)
    self.assertAllCloseAccordingToType(res['violation'], .5)

  def test_pr(self):
    ranks, labels = self._gen_metric_data()
    res = metrics.compute_pr(ranks, labels, 2)
    self.assertAllCloseAccordingToType(res['2-PCR'], 2/3)
    self.assertAllCloseAccordingToType(res['2-PCP'], 13/18)
    self.assertAllCloseAccordingToType(res['2-OVR'], 2/3)
    self.assertAllCloseAccordingToType(res['2-OVP'], 2/3)
    self.assertAllCloseAccordingToType(res['2-Hard-Acc'], 1/3)

  def test_ap(self):
    ranks, labels = self._gen_metric_data()
    res = metrics.compute_ap(ranks, labels)
    gt = ((2/3+1/2)/2 + (1+2/3)/2 + (1+1)/2)/3
    self.assertAllCloseAccordingToType(res['AP'], gt)

  def test_auc(self):
    from sklearn.metrics import auc
    ranks, labels = self._gen_metric_data()
    res = metrics.compute_auc(ranks, labels)
    TPRs = [1/3,2/3,1]
    FPRs = [1/3,2/3,1]
    self.assertAllCloseAccordingToType(res['AUC'], auc(FPRs, TPRs, reorder=True))

  def test_threshold2counts(self):
    logits = np.array([[100, 20, 40, 30], [50, 70, 40, 20]])
    counts = metrics._threshold2counts(logits, 49)
    self.assertAllCloseAccordingToType(counts, [1, 2])


if __name__ == "__main__":
  tf.test.main()
# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
