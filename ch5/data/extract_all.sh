#!/bin/sh

for file in $1/*.txt; do
    echo ${file}
    nkf -d "${file}" | ./extract_text.py >> $1/input.txt
done
