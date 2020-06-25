#!/bin/bash
sudo apt-get install -y python-tk && \
pip3 install -r ~/models/official/requirements.txt

# checkpoint 모델 저장할 경로
MODEL_DIR="./models/resnet50"
# Pre-trained Model 읽어오는 경로
RESNET_CHECKPOINT="./models/resnet50"
# train data 폴더 경로
TRAIN_FILE_PATTERN="./models/train2017"
# validation data 폴더 경로
EVAL_FILE_PATTERN="./models/val2017"
# validation annotation JSON 파일 경로
VAL_JSON_FILE="./models/captions_val2017.json"
python3 ~/models/official/vision/detection/main.py \
  --strategy_type=one_device \
  --num_gpus=1 \
  --model_dir="${MODEL_DIR?}" \
  --mode=eval \
  --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"

  # 자세한 건 https://github.com/tensorflow/models/tree/master/official/vision/detection 참조