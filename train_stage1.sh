#!/bin/sh
export PYTHONPATH="./layer:$PYTHONPATH"

python train_DAN.py --solver prototxt/stage1/solver.prototxt --iters 175000 --weights output/stage1/DAN_vgglike_iter_42000.solverstate --gpu 0 2>&1 | tee ./log/vgglike_2.log

