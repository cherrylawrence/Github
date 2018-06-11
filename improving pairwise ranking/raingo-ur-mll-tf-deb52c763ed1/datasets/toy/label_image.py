#!/usr/bin/env python
"""
label shape dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import networkx as nx

toy_meta_dir = '../../cnns/toy/'

def load_list(path):
  res = []
  with open(path) as reader:
    for line in reader:
      res.append(line.strip())
  return res

def main():
  leaves = load_list(osp.join(toy_meta_dir, 'leaves.txt'))
  relations = [line.split() for line in
      load_list(osp.join(toy_meta_dir, 'relations.txt'))]

  all_nodes = set(sum(relations, []))
  extras = list(all_nodes - set(leaves))
  vocab = leaves + extras
  w2i = {w:i for i, w in enumerate(vocab)}

  with open('vocab.txt', 'w') as writer:
    for w in vocab:
      print(w, file=writer)

  graph = nx.DiGraph()
  for edge in relations:
    graph.add_edge(edge[0], edge[1])

  for line in sys.stdin:
    dirname = osp.dirname(line.strip())
    wnid = osp.basename(dirname)
    if wnid in graph:
      labels = [wnid]
      labels.extend(nx.descendants(graph, wnid))
      labels = ','.join([str(w2i[w]) for w in labels])
      print(line.strip(), labels)

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
