#!/bin/bash
# vim ft=sh

python ../../cnns/toy/gen_shapes.py 1000
find $PWD/data/imgs -type f | python label_image.py > data/all.txt

cd data
sort -R all.txt | split -l 4000
cat xaa xab xac > train.txt
mv xad dev.txt
mv xae test.txt
rm xa* all.txt
