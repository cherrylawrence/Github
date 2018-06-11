#!/bin/bash
# vim ft=sh

for config in `find ../mll/ -name config -follow`
do
  rel=`echo $config | sed 's/..\/mll\///'`
  savedir=`dirname $rel`
  reldir=root:mll/$savedir
  mkdir -p $savedir
  pushd $savedir
  echo "--depend $reldir/FINISHED" > config
  echo "--pretrained_ckpt $reldir/model.ckpt" >> config
  popd
done
