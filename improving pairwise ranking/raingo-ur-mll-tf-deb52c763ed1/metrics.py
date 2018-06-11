#!/usr/bin/env python
"""
compute evaluation metric for the npz results, generated from classifier.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import json

import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score

# ranks: the argsort of the logits
# labels: list of label set
# K: 3 or 5, the top labels to use
# Reference: Gong, Yunchao, et al. "Deep convolutional ranking for multilabel image annotation." arXiv preprint arXiv:1312.4894 (2013).
# This paper does not specify how to deal with 0/0 case, in this implementation, the 0/0 is removed from the per class computation, i.e., does not count
# mean([A/B for A, B if B != 0])
# fix, at least on 09/22/2016, 0/0 = 0
def compute_pr(ranks, labels, K, suffix=None):

  NiC = defaultdict(float)
  NiG = defaultdict(float)
  NiP = defaultdict(float)

  num_tags = ranks.shape[1]
  if suffix is None:
    suffix = '%d' % K

  if not isinstance(K, list):
    K = [K] * len(ranks)

  acc = 0.0
  hard_acc = []
  for rank, L, k in zip(ranks, labels, K):
    L = set(L)
    rank = rank[:k]

    for t in L:
      NiG[t] += 1.0

    A = 0.0
    exact = len(L) == len(rank)
    for t in rank:
      NiP[t] += 1.0
      if t in L:
        NiC[t] += 1.0
        A = 1.0
      else:
        exact = False
    acc += A
    hard_acc.append(exact)

  acc /= ranks.shape[0]
  pcr = []
  pcp = []

  num_samples = []
  num_predicted = []

  NC = 0.0
  NP = 0.0
  NG = 0.0

  #zero_div_zero = float('nan')
  zero_div_zero = 0.
  for t in range(num_tags):
    ir = 0
    ip = 0

    if t in NiG:
      pcr.append(NiC[t]/NiG[t])
      num_samples.append(NiG[t])
    else:
      pcr.append(zero_div_zero)
      num_samples.append(0)

    if t in NiP:
      pcp.append(NiC[t]/NiP[t])
      num_predicted.append(NiP[t])
    else:
      pcp.append(zero_div_zero)
      num_predicted.append(0)

    NC += NiC[t]
    NP += NiP[t]
    NG += NiG[t]

  res = {}
  kvs = {'PCR':np.nanmean(pcr),
      'PCP':np.nanmean(pcp),
      'OVR': NC/NG,
      'OVP': NC/NP if NP > 0 else 0,
      'Acc': acc,
      'Hard-Acc': sum(hard_acc)/len(hard_acc)}
  for key, value in kvs.items():
    res[suffix + '-' + key] = value

  res['%s-PCR-P' % suffix] = pcr
  res['%s-PCP-P' % suffix] = pcp
  res['#samples-P'] = num_samples
  res['#predicted-%s-P' % suffix] = num_predicted
  return res

## Reference:
# Zhang, Min-Ling, and Zhi-Hua Zhou. "ML-KNN: A lazy learning approach to multi-label learning." Pattern recognition 40.7 (2007): 2038-2048. Eqn. (5)
def compute_ap(ranks, labels):
  aps = []
  for rank, label in zip(ranks, labels):
    rank = rank.tolist()
    ap = 0.0
    for l in label:
      rank_l = rank.index(l)+1
      ap += len(set(rank[:rank_l]) & label) / rank_l
    aps.append(ap/len(label))
  return {'AP':sum(aps)/len(aps)}

import itertools
def compute_num_violations(ranks, labels):
  """
  compute the average number of violations
  """
  num_labels = ranks.shape[1]
  all_labels = set(range(num_labels))
  violations = []

  for rank, pos in zip(ranks, labels):
    inv_rank = [0] * num_labels
    for i, r in enumerate(rank):
      inv_rank[r] = i

    pos = set(pos)
    neg = all_labels - pos
    violation = 0
    total = 0
    for p, n in itertools.product(pos, neg):
      total += 1
      if inv_rank[p] > inv_rank[n]:
        violation += 1
    violations.append(violation/total)

  return {'violation': sum(violations)/len(violations)}

## Referece:
# Bucak, Serhat S., et al. "Efficient multi-label ranking for multi-class learning: application to object recognition." 2009 IEEE 12th International Conference on Computer Vision. IEEE, 2009.
def compute_auc(ranks, labels):
  N = ranks.shape[1]
  tp, fp, fn = [0]*N, [0]*N, [0]*N

  for rank, label in zip(ranks, labels):
    tp_i,fp_i = 0,0
    fn_i = len(label)
    for K in range(N):
      if rank[K] in label:
        tp_i += 1
        fn_i -= 1
      else:
        fp_i += 1

      fp[K] += fp_i
      tp[K] += tp_i
      fn[K] += fn_i

  TT = N * ranks.shape[0]
  fpr = []
  tpr = []
  for K in range(N):
    tn = TT - (tp[K]+fn[K]+fp[K])
    if (fp[K] + tn) != 0 and (tp[K] + fn[K]) != 0:
      fpr.append(fp[K]/(fp[K]+tn)) #fp / (fp + tn)
      tpr.append(tp[K]/(tp[K]+fn[K])) #tp / (tp + fn)
  return {'AUC': auc(fpr, tpr, reorder = True)}


def sparse2dense(sparse, dim):
  num = len(sparse)
  res = np.zeros((num, dim), dtype=np.int32)

  for i, vec in enumerate(sparse):
    for l in vec:
      res[i, l] = 1

  return res

def sk_f1_score(ranks, labels, K, suffix=None):
  num_labels = ranks.shape[1]
  if suffix is None:
    suffix = '%d' % K

  if not isinstance(K, list):
    K = [K] * len(ranks)

  preds = []
  for k, rank in zip(K, ranks):
    preds.append(rank[:k])

  y_pred = sparse2dense(preds, num_labels)
  y_true = sparse2dense(labels, num_labels)

  res = {}
  avgs = ['micro', 'macro', 'weighted', 'samples']
  for avg in avgs:
    res['%s-sk-f1-' % suffix + avg] = f1_score(y_true, y_pred, average=avg)
  return res

def prepare_eval(res):
  logits = res['logits']
  label_map = res['label_map']
  path = res['path']

  _, indices = np.unique(path, return_index=True)
  logits = logits[indices, :]
  label_map = label_map[indices, :]

  lcnt = None
  if 'lcnt' in res:
    lcnt = res['lcnt']
    lcnt = lcnt[indices].flatten()

  thresh = None
  if 'thresh' in res:
    thresh = res['thresh']
    thresh = thresh[indices, :]

  # largest comes first
  ranks = np.argsort(-logits, axis=1)

  labels = []
  for lm in label_map:
    labels.append(set(np.where(lm==1)[0].tolist()))

  return labels, ranks, lcnt, logits, thresh

def MAE(lcnt, labels):
  lcnt_gt = np.array([len(l) for l in labels])
  lcnt_int = np.round(lcnt)

  res = {}
  res['mean-abs-error'] = float(np.mean(np.abs(lcnt-lcnt_gt)))
  res['mean-rounded-error'] = float(np.mean(np.abs(lcnt_int-lcnt_gt)))
  return res

def var_len_acc(lcnt, labels, ranks):
  counts = np.round(lcnt).astype(int).flatten().tolist()
  counts_gt = [len(l) for l in labels]

  funcs = [compute_pr,
      sk_f1_score]

  res = {}
  for func in funcs:
    res.update(func(ranks, labels, counts, 'pred'))
    res.update(func(ranks, labels, counts_gt, 'gt'))

  return res, counts

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

# cv_k_threshold
tf.app.flags.DEFINE_boolean('cv_k_threshold', False,
     """whether to cross validate the best K and threshold""")

def _is_dev():
  return 'dev' in FLAGS.eval_split

def _dev_json():
  with open(FLAGS.ckpt_path + '.dev.tf.json') as reader:
    return json.load(reader)

def _cv_k(labels, ranks, counts):
  res = []
  for k in range(-10,11):
    K = [k+c if k+c > 0 else 0 for c in counts]
    f1 = sk_f1_score(ranks, labels, K=K, suffix='tmp')
    res.append((k, f1['tmp-sk-f1-macro']))
  return max(res, key=lambda x:x[1])[0]

def _get_best_info(res, key):
  res = [f[1] for f in res.items() if key in f[1]['info']]
  return max(res, key=lambda x:x['score'])['info'][key]

def _get_k(labels, ranks, counts):
  if _is_dev():
    return _cv_k(labels, ranks, counts)
  else:
    res = _dev_json()
    return _get_best_info(res, 'best-k')

def _threshold2counts(logits, threshold, skip_sort=False):
  threshold = -threshold
  logits = -logits

  if not skip_sort:
    logits = np.sort(logits, axis=1)
  counts = np.apply_along_axis(lambda x:x.searchsorted(threshold),
      axis=1, arr=logits).tolist()
  return counts

def _cv_threshold(labels, ranks, logits):
  logits = -np.sort(-logits, axis=1)
  scores = logits[:, :10]
  lower = scores.min()
  upper = scores.max()
  res = []
  num_bins = 50
  for threshold in np.linspace(lower, upper, num_bins):
    counts = _threshold2counts(logits, threshold, skip_sort=True)
    f1 = sk_f1_score(ranks, labels, counts, suffix='tmp')
    res.append((threshold, f1['tmp-sk-f1-macro']))
  return max(res, key=lambda x:x[1])[0]

def _get_threshold(labels, ranks, logits):
  if _is_dev():
    return _cv_threshold(labels, ranks, logits)
  else:
    res = _dev_json()
    return _get_best_info(res, 'best-threshold')

def cv_k_threshold(labels, ranks, logits, counts):
  res = {}

  # cv_k
  k = _get_k(labels, ranks, counts)
  res['best-k'] = k
  K = [k+c if k+c > 0 else 0 for c in counts]
  res.update(compute_pr(ranks, labels, K, 'best-k'))
  res.update(sk_f1_score(ranks, labels, K, 'best-k'))

  # cv_threshold
  threshold = _get_threshold(labels, ranks, logits)
  res['best-threshold'] = threshold
  counts = _threshold2counts(logits, threshold)
  res.update(compute_pr(ranks, labels, counts, 'best-threshold'))
  res.update(sk_f1_score(ranks, labels, counts, 'best-threshold'))
  return res

def evaluate(res):
  labels, ranks, lcnt, logits, thresh = prepare_eval(res)

  from functools import partial
  funcs = [compute_ap,
      #compute_auc,
      compute_num_violations,
      partial(compute_pr, K=3),
      partial(compute_pr, K=5),
      partial(sk_f1_score, K=3),
      partial(sk_f1_score, K=5),
      ]

  metrics = {}
  for func in funcs:
    metrics.update(func(ranks, labels))

  score = metrics['AP']
  counts = [0] * len(labels)
  if lcnt is not None:
    metrics.update(MAE(lcnt, labels))
    res, counts = var_len_acc(lcnt, labels, ranks)
    metrics.update(res)
    score = -metrics['mean-rounded-error']

  if thresh is not None:
    # this is 0, because the logits is already applied threshold
    counts = _threshold2counts(logits, 0)
    metrics.update(compute_pr(ranks, labels, counts, 'pred-thre'))
    metrics.update(sk_f1_score(ranks, labels, counts, 'pred-thre'))
    score = metrics['pred-thre-sk-f1-macro']

  if FLAGS.cv_k_threshold or FLAGS.eval_only:
    metrics.update(cv_k_threshold(labels, ranks, logits, counts))

  return metrics, score

def main():
  json_path = sys.argv[1]
  with open(json_path) as reader:
    res = json.load(reader)
  res = max(res.items(), key = lambda x:x[1]['score'])[1]['info']

  if len(sys.argv) < 3:
    print(*res.keys())
    return
  outputs = sys.argv[2:]

  outputs = [res[o] for o in outputs]
  print(json_path, *outputs)

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
