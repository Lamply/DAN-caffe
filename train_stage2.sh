#!/bin/sh
export PYTHONPATH="./layer:$PYTHONPATH"

python train_DAN.py --solver prototxt/stage2/solver_2.prototxt --iters 240000 --weights output/stage2/DAN_s2_iter_189000.solverstate --gpu 0 2>&1 | tee ./log/vgglike_s2_2.log

# poweroff
