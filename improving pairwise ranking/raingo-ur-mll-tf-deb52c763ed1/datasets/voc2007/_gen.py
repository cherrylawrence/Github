#!/usr/bin/env python
"""
generate the image list given the set name  for the pascal voc dataset
python raw-dir set
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp

from bs4 import BeautifulSoup

def _load_list(path):
  res = []
  with open(path) as reader:
    for line in reader:
      res.append(line.strip())
  return res

def main():
  raw_dir = sys.argv[1]
  setn = sys.argv[2]
  imgs = _load_list(osp.join(raw_dir, 'ImageSets', 'Main', setn+'.txt'))

  vocab = _load_list('vocab.txt')
  vocab = {w:str(i) for i,w in enumerate(vocab)}

  for img in imgs:
    img_path = osp.join(raw_dir, 'JPEGImages', img+'.jpg')
    anno_path = osp.join(raw_dir, 'Annotations', img+'.xml')
    with open(anno_path) as rd:
      anno = rd.read()
    soup = BeautifulSoup(anno, 'lxml')
    cls = []
    for obj in soup('object'):
      name = obj.find('name').string
      difficult = int(obj.find('difficult').string)
      if difficult == 0:
        cls.append(vocab[name])
    if len(cls) > 0:
      print(img_path,','.join(set(cls)))

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
