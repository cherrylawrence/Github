#!/bin/bash
# vim ft=sh

ckpt_path=$1

this_dir=`dirname $0`
$this_dir/cnns/ckpt2graph.sh $ckpt_path eval/clone-0/classifier/inputs/batch_join eval/clone-0/classifier/classifier/fc8-nus/add
