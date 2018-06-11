#!/bin/bash
# vim ft=sh

base_dir='./data/data2/'
mkdir -p $base_dir

python dump-coco.py ./data/raw/annotations/instances_train2014.json
python dump-coco.py ./data/raw/annotations/instances_val2014.json

pushd $base_dir
sort -R train2014.txt | split -l 80000
rm train2014.txt
mv xaa train.txt
mv xab dev.txt
mv val2014.txt test.txt
popd

function _splits {
  echo data/data/train.txt
  echo data/data/dev.txt
  echo data/data/test.txt
}

_splits | xargs -L 1 ../../compile_data.sh
