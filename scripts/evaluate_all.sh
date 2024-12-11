#!/bin/bash

# retrieval
printf "007-25--16_05_12_longcliponly_cyclip_coco_lr1e6_seed7_MPCSE_epoch2.pt"
nohup python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name our_longcliponly --num_caption 1 --ckpt_file_name final/007-25--16_05_12_longcliponly_cyclip_coco_lr1e6_seed7_MPCSE_epoch2.pt --gpu 0
nohup python3 ../evaluation/evaluate_retrieval.py --dataset_type coco --model_name gold --ret_model_name our_longcliponly --num_caption 1 --ckpt_file_name final/007-25--16_05_12_longcliponly_cyclip_coco_lr1e6_seed7_MPCSE_epoch2.pt --gpu 0
nohup python3 ../evaluation/evaluate_retrieval.py --dataset_type nocaps --model_name gold --ret_model_name our_longcliponly --num_caption 1 --ckpt_file_name final/007-25--16_05_12_longcliponly_cyclip_coco_lr1e6_seed7_MPCSE_epoch2.pt --gpu 0

classification
nohup python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name our_longcliponly --ckpt_file_name final/007-25--16_05_12_longcliponly_cyclip_coco_lr1e6_seed7_MPCSE_epoch2.pt --gpu 0
nohup python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name our_longcliponly --ckpt_file_name final/007-25--16_05_12_longcliponly_cyclip_coco_lr1e6_seed7_MPCSE_epoch2.pt --gpu 0

# correlation
nohup python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model our_longcliponly --compute_refpac --ckpt_file_name final/007-25--16_05_12_longcliponly_cyclip_coco_lr1e6_seed7_MPCSE_epoch2.pt --gpu 0
nohup python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model our_longcliponly --compute_refpac --ckpt_file_name final/007-25--16_05_12_longcliponly_cyclip_coco_lr1e6_seed7_MPCSE_epoch2.pt --gpu 0
