#!/usr/bin/env python
"""
given stream of intergers, compute the min, max and median
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp

def main():
  nums = [int(line.strip()) for line in sys.stdin]
  print(max(nums), min(nums), sorted(nums)[len(nums)//2], len(nums))

  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
