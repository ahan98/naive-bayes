#!/bin/sh
for file in ./NominalData/*.arff; do
    echo "\n********* Using $file *********"
    python3 main.py "$file"
done
