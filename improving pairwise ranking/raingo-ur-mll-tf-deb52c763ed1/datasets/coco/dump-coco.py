#!/usr/bin/env python
"""
dump the coco json files into txt file
image-path labels

python dump-coco.py path-to-instance-json-file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp
import simplejson

image_dir = './data/raw/'
image_dir = osp.realpath(image_dir)

def main():
  split = osp.splitext(sys.argv[1])[0].split('_')[1]
  with open(sys.argv[1]) as reader:
    coco = simplejson.load(reader)

  with open('dictionary.txt', 'w') as writer:
    vocab = {}
    for info in coco['categories']:
      vocab[info['id']] = info['name']

    i2w = [w for i, w in sorted(vocab.items())]
    w2i = {w:i for i, w in enumerate(i2w)}

    for name in i2w:
      print(name, file=writer)

  images = {}
  for info in coco['images']:
    images[info['id']] = osp.join(split, info['file_name'])

  from collections import defaultdict
  labels = defaultdict(list)
  for info in coco['annotations']:
    label = w2i[vocab[info['category_id']]]
    labels[info['image_id']].append(str(label))

  with open(osp.join('data', 'data2', split + '.txt'), 'w') as writer:
    for image, ls in labels.items():
      print(osp.join(image_dir, images[image]), ','.join(set(ls)), file=writer)

  with open(osp.join('data', 'data2', split + '.name.txt'), 'w') as writer:
    for image, ls in labels.items():
      ls = [i2w[int(i)] for i in set(ls)]
      print(osp.join(image_dir, images[image]), ','.join(ls), file=writer, sep='\t')


  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
