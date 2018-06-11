#!/bin/bash
# vim ft=sh

raw_dir='./data/raw/'
data_dir='./data/data/'

mkdir -p $data_dir

cat $raw_dir/Test.txt $raw_dir/Train.txt | awk 'NF>1 {print $0}' | sed 's/ /,/g' | sed 's/,/ /' > $data_dir/all

pushd $data_dir

sort -R all | split -l 150000
mv xab test.txt
mv xaa tmp
sort -R tmp | split -l 145000
mv xaa train.txt
mv xab dev.txt
rm tmp all

popd

function _splits {
  echo data/data/train.txt
  echo data/data/dev.txt
  echo data/data/test.txt
}

_splits | xargs -L 1 python ../../compile_data.py
