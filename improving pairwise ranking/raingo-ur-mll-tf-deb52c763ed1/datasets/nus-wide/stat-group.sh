#!/bin/bash
# vim ft=sh

# *.name.txt: path label1 label2 ..

cat - | cut -d " " -f 2- | grep ' ' | sort | uniq -c | sort -nr
