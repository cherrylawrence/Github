#!/usr/bin/env python
"""
read npz paths from stdin
dump the precision recall curve. treat them as binary
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import numpy as np
import random
from sklearn.metrics import precision_recall_curve

def _parse_json_path(json_path):
  name = osp.dirname(json_path)
  names = name.split('/')
  exp = names[3:]
  names = ['/'.join(names[:2]), '/'.join(exp[::-1])]
  return names

def main():
  print('dataset', 'exp', 'split', 'precision', 'recall', 'threshold', 'rand')
  samples = 1000
  for npz in sys.stdin:
    npz_name = osp.basename(npz)
    split = npz_name.split('.')[1]
    names = _parse_json_path(npz)
    names.append(split)

    data = np.load(npz.strip())

    _, indices = np.unique(data['path'], return_index=True)
    logits = data['logits'][indices, :]
    label_map = data['label_map'][indices, :]

    indices = range(logits.shape[0])
    random.shuffle(indices)

    y_true = label_map[indices[:samples],:].flatten()
    y_scores = logits[indices[:samples],:].flatten()

    for prec, reca, thre in zip(
        *precision_recall_curve(y_true, y_scores)):
      fields = names + [prec, reca, thre, random.random()]
      print(*fields)
  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
