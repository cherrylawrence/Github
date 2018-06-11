#!/bin/bash
# vim ft=sh

raw_dir='./data/raw/'
data_dir='./data/data/'

mkdir -p $data_dir
raw_dir=`readlink -f $raw_dir`

if [ ! -f vocab.txt ]
then
  find $raw_dir/ImageSets/Main/ -name '*_train*' | xargs -L 1 basename | awk -F'_' '{print $1}' | sort -u > vocab.txt
fi

python _gen.py $raw_dir train > data/data/train.txt
python _gen.py $raw_dir val > data/data/dev.txt
python _gen.py $raw_dir test > data/data/test.txt

function _splits {
  echo data/data/train.txt
  echo data/data/dev.txt
  echo data/data/test.txt
}

_splits | xargs -L 1 ../../compile_data.sh
