#!/bin/bash
# vim ft=sh

gpu=$1

exps="coco nus-wide"

source ~/.virtualenvs/tf/bin/activate
for exp in `echo $exps | tr ' ' '\n' | sort -R`
do
  pushd $exp
  ../../trains.sh data/exps/ $gpu
  popd
done
