#!/bin/bash

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name our_longclip --num_caption 1 

python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name our_longcliponly --num_caption 1 --is_distil t

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name our_longclip --num_caption 1 --remove_artifacts t

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name clip --num_caption 1

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name clip --num_caption 1 --shift_features t

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name clip --num_caption 1 --shift_features t --delta 1

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name clip --num_caption 1 --random_init t

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name blip --num_caption 1

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name blip2 --num_caption 1

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name blip2coco --num_caption 1

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name clipcloob --num_caption 1

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name pacscore --num_caption 1

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name longclip --num_caption 1 

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name longclip --num_caption 1 --remove_artifacts t

# python3 ../evaluation/evaluate_retrieval.py --dataset_type flickr --model_name gold --ret_model_name cyclip --num_caption 1