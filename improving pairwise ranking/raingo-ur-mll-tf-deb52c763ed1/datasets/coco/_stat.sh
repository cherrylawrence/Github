#!/bin/bash
# vim ft=sh

txt_path=$1
cat $txt_path > tmp

function num_labels_per_image {
  awk '{print $2}' tmp | awk -F',' '{print NF}'
}


function num_images_per_label {
  awk '{print $2}' tmp | tr ',' '\n' | sort | uniq -c | sort -nr | awk '{print $1}'
}

num_labels_per_image | python _stat.py
num_images_per_label | python _stat.py
rm tmp
