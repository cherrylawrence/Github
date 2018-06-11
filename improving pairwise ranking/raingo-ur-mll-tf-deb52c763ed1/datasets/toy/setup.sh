#!/bin/bash
# vim ft=sh

# ls data/*.txt | xargs -L 1 python ../../compile_data.py

rm -Rf data/exps
rsync -arL ../mll-exps/base/alexnet_bn/ ./data/exps/
