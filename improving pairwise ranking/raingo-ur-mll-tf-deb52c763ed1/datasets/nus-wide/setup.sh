#!/bin/bash
# vim ft=sh

# ls data/*.txt | xargs -L 1 python ../../compile_data.py
rsync -rL ../mll-exps/base/ ./data/exps/
find data/exps/ -name config | grep -v vgg | xargs rm
find data/exps/ -name config | grep -v thresh | xargs rm
