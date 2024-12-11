from tqdm import tqdm
import json
import cv2
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os
import numpy as np
import random
import torchvision.transforms as transforms 

f = open("../configs/root.txt", "r")
data_dir = f.read()

data4v_root = f'{data_dir}/sharegpt4v/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
image_root = f'{data_dir}/sharegpt4v/data/'


class share4v_val_dataset(data.Dataset):
    def __init__(self, base_model):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
        _ , self.preprocess = clip.load(base_model)
        
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self, base_model, sel_data, args):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        self.sel_data = sel_data
        if args.color_change:
            self.json_name = 'share-captioner_coco_lcs_sam_1246k_1107_nocolors.json'
        
        with open(data4v_root + self.json_name, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)[self.total_len:]
        num_data = len(json_data)

        if self.sel_data != 'all':
            self.json_data = []
            for idx in tqdm(range(num_data)):
                img_folder = json_data[idx]['image']
                if self.sel_data in img_folder:
                    self.json_data.append(json_data[idx])
        else:
            self.json_data = json_data
        print("# Training Data:", len(self.json_data))
        
        _ , self.preprocess = clip.load(base_model)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        

        caption_short = caption.split(". ")[0]
        
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return index, image_tensor, caption, caption_short