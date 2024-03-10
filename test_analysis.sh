#!/bin/bash
export PYTHONPATH="./layer:$PYTHONPATH"

python ~/caffe_tools/model_analysis.py --model prototxt/stage2/deploy.prototxt --weights output/DAN_iter_63000\(final_2stage\).caffemodel --display



