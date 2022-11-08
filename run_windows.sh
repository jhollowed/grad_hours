#!/bin/bash

for i in {7..60}; do
    python ./trends.py $i
done

cd ./figs
convert -delay 15 -loop 0 *.png animation.gif
