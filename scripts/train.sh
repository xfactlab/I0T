#!/bin/bash

nohup accelerate launch --config_file ~/VisLang/configs/deepspeed.yaml ../training/train.py \
    --batch-size 64 \
    --model_name longcliponly_cyclip \
    --train_dataset_type coco \
    --run_name coco_lr1e6_seed7_MPCSE_encoder_freeze_clip \
    --epochs 3 \
    --lr 1e-6 \
    --seed 7 \
    --add_batch_norm True \
    --freeze True 

