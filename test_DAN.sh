#!/bin/sh
export PYTHONPATH="./layer:$PYTHONPATH"

python test_DAN.py  2>&1 | tee ./log/test_layer.log

