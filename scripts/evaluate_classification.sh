#!/bin/bash

python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name our_longclip

python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name our_longclip

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name our_longclip --remove_artifacts t

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name our_longclip --remove_artifacts t

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name clip

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name clip

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name clip --random_init t

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name clip --random_init t

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name clip --shift_features t

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name clip --shift_features t --delta 1

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name clip --shift_features t

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name clip --shift_features t --delta 1

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name longclip 

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name longclip

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name longclip --remove_artifacts t

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name longclip --remove_artifacts t

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name longclip --shift_features t

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name clipcloob

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name clipcloob

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name pacscore

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name pacscore

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name blip --num_workers 2

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name blip --num_workers 2

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name blip2 --num_workers 2

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name blip2 --num_workers 2

# python3 ../evaluation/evaluate_classification.py --dataset_type cifar100 --ret_model_name cyclip --num_workers 2

# python3 ../evaluation/evaluate_classification.py --dataset_type birdsnap --ret_model_name cyclip --num_workers 2