salloc -p lrz-hgx-h100-94x4 --gres=gpu:1

srun --pty bash

enroot start --root -m /dss/dsshome1/08/ga25ley2/code/crosscode/:/workspace/crosscode -m /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2:/workspace/data pytorch-25.12

uv run crosscode/trainers/topk_crosscoder/run.py crosscode/trainers/topk_crosscoder/config.yaml