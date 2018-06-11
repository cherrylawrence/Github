#!/bin/bash
# vim ft=sh


function _stat {
  data=$1
  raw_dir=$2

  cat $raw_dir/*.name.txt | awk '{print $2}' | tr ',' '\n' | sort | uniq -c | sort -nr > $raw_dir/label.stat.txt
  total=`cat $raw_dir/*.name.txt | wc -l`
  python stat-vocab.py $raw_dir/label.stat.txt $data/vocab.txt $total
}

_stat coco coco/data/data2/
_stat nus-wide nus-wide/data/raw/
data=coco
raw_dir=
