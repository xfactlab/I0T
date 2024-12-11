import os
import ast
import json
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import torch
import sys
sys.path.insert(0, '../training/')
import zeroshot_data
f = open("../configs/root.txt", "r")
data_dir = f.read()
f = open("../configs/subroot.txt", "r")
data_dir_sub = f.read()

def load_images(dataset_type, ret_model_name, preprocess, batch_size=128, num_workers=4):
    if 'cyclip' in ret_model_name:
        preprocess = preprocess.process_image
        
    if dataset_type == 'cifar100':
        from torchvision.datasets import CIFAR100
        dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
        classnames = zeroshot_data.classes
        prompt_templates = zeroshot_data.templates
        
    elif dataset_type == 'birdsnap':
        import torchvision.datasets as datasets
        data_path = f'{data_dir}/zeroshot_datasets/birdsnap/test'
        dataset = datasets.ImageFolder(data_path, transform=preprocess)
        classnames = zeroshot_data.birdsnap_classnames
        prompt_templates = zeroshot_data.birdsnap_templates
        
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return data, classnames, prompt_templates

def load_image_captions(dataset_type, model_name):
    if dataset_type == 'coco':
        principal_dir = f'{data_dir_sub}/PnP-VQA/lavis/coco/images/val2014'
    elif dataset_type == 'flickr':
        principal_dir = f'{data_dir_sub}/PnP-VQA/lavis/flickr30k/images/flickr30k-images'
    elif dataset_type == 'nocaps':
        principal_dir = f'{data_dir_sub}/PnP-VQA/lavis/nocaps/images'
            
    if model_name == 'gold':
        if dataset_type == 'coco':
            gold_caption_file = f'{data_dir_sub}/Evaluation/COCO_Captions/NewCaptions/COCO_GOLD.json'
        elif dataset_type == 'flickr':
            gold_caption_file = f'{data_dir_sub}/Evaluation/Flickr_Captions/NewCaptions/Flickr_GOLD.json'
        elif dataset_type == 'nocaps':
            gold_caption_file = f'{data_dir_sub}/Evaluation/Nocaps_Captions/NewCaptions/NoCaps_GOLD.json'

        f = open(gold_caption_file)
        gold_caption_data = json.load(f)

        if dataset_type == 'nocaps':
            num_data = len(gold_caption_data['images'])
        else:
            num_data = len(gold_caption_data)
        print('# Data: ', num_data)

        image_captions = {}
        for i in tqdm(range(num_data)):
            if dataset_type == 'nocaps':
                generated_captions = []
                for j in [10*i + k for k in range(10)]:
                    image_id = gold_caption_data['images'][i]['id']
                    image_path = 'val/' + gold_caption_data['images'][i]['file_name']
                    annot_id = gold_caption_data['annotations'][j]['image_id']
                    assert image_id == annot_id
                    generated_captions.append(gold_caption_data['annotations'][j]['caption'])
                image_captions[image_path] = generated_captions
            else:
                line = gold_caption_data[i]
                image_name = line['image_id'].split('/')[-1]
                gold_captions = line['captions']
                image_captions[image_name] = gold_captions
    else:
        dataframe_folder = '../../ICI/data/captions'
        dataframe_name = f'df_{dataset_type}_{model_name}_clean.csv'
        dataframe_path = os.path.join(dataframe_folder, dataframe_name)
        df = pd.read_csv(dataframe_path)

        num_data = df.shape[0]
        print('# Data: ', num_data)

        image_captions = {}
        for i in tqdm(range(num_data)):
            image_path = df['id'][i]
            candidates = df['cap'][i]
            candidates = ast.literal_eval(candidates)
            candidates = [c.replace("[", '') for c in candidates]
            generated_captions = [c.replace("]", '') for c in candidates]
            image_captions[image_path] = generated_captions

    return principal_dir, image_captions, num_data


def load_image_refcaptions(dataset_type, model_name):
    # References
    if dataset_type == 'coco':
        gold_caption_file = f'{data_dir_sub}/Evaluation/COCO_Captions/NewCaptions/COCO_GOLD.json'
    elif dataset_type == 'flickr':
        gold_caption_file = f'{data_dir_sub}/Evaluation/Flickr_Captions/NewCaptions/Flickr_GOLD.json'
    elif dataset_type == 'nocaps':
        gold_caption_file = f'{data_dir_sub}/Evaluation/Nocaps_Captions/NewCaptions/NoCaps_GOLD.json'

    f = open(gold_caption_file)
    gold_caption_data = json.load(f)

    if dataset_type == 'nocaps':
        num_data = len(gold_caption_data['images'])
    else:
        num_data = len(gold_caption_data)
    print('# Reference Data: ', num_data)

    # Generated Captions
    dataframe_folder = '../../ICI/data/captions'
    dataframe_name = f'df_{dataset_type}_{model_name}_clean.csv'
    dataframe_path = os.path.join(dataframe_folder, dataframe_name)
    df = pd.read_csv(dataframe_path)

    num_data2 = df.shape[0]
    assert num_data == num_data2
    print('# Caption Data: ', num_data)

    image_references = {}
    image_captions = {}
    for i in tqdm(range(num_data)):
        # References
        if dataset_type == 'nocaps':
            generated_captions = []
            for j in [10*i + k for k in range(10)]:
                image_id = gold_caption_data['images'][i]['id']
                image_path = 'val/' + gold_caption_data['images'][i]['file_name']
                annot_id = gold_caption_data['annotations'][j]['image_id']
                assert image_id == annot_id
                generated_captions.append(gold_caption_data['annotations'][j]['caption'])
            image_references[image_path] = generated_captions
        else:
            line = gold_caption_data[i]
            image_name = line['image_id'].split('/')[-1]
            gold_captions = line['captions']
            image_references[image_name] = gold_captions

        # Generated Captions
        image_path = df['id'][i]
        candidates = df['cap'][i]
        candidates = ast.literal_eval(candidates)
        candidates = [c.replace("[", '') for c in candidates]
        generated_captions = [c.replace("]", '') for c in candidates]
        image_captions[image_path] = generated_captions
    
    return image_references, image_captions
