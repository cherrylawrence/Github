#!/usr/bin/env python
"""
given label stats, and the label list,

write another label list, vocab.txt.stat.txt,

otherthan the top-10 labels, the rest is changed to BG
for the top-10, label (%)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp

def load_list(path):
  res = []
  with open(path) as reader:
    for line in reader:
     res.append(line.strip())
  return res

def main():
  stats_raw = [line.split() for line in load_list(sys.argv[1])]
  stats = sorted([(int(f[0]), ' '.join(f[1:])) for f in stats_raw])
  stats = {l:c for c, l in stats[-10:]}

  vocab = load_list(sys.argv[2])
  total = int(sys.argv[3])

  with open(sys.argv[2]+'.stat.txt', 'w') as writer:
    for l in vocab:
      if l in stats:
        print('%s (%.0f%%)' % (l, stats[l]/total*100), file=writer)
        stats.pop(l)
      else:
        print("BG", file=writer)
  print(stats.keys(), sys.argv[2])

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
