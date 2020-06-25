#!/bin/bash
sudo apt-get install -y python-tk && \
pip3 install -r ~/models/official/requirements.txt

MODEL_DIR="./models/resnet50"
RESNET_CHECKPOINT="./models/resnet50"
TRAIN_FILE_PATTERN="./models/train2017"
EVAL_FILE_PATTERN="./models/val2017"
VAL_JSON_FILE="./models/captions_val2017.json"
python3 ~/models/official/vision/detection/main.py \
  --strategy_type=one_device \
  --num_gpus=1 \
  --model_dir="${MODEL_DIR?}" \
  --mode=eval \
  --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"