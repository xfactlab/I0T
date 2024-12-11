import json
from tqdm import tqdm
import os
from PIL import Image
import torch
import numpy as np
import clip
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sys
sys.path.insert(0, '../')
from src.args import get_args_4_evaluate_classification
from src.load import load_images
from src.prepare import prepare_models_only
from src.utils import zero_shot_classifier
from src.features import move_features, remove_artifact_features
f = open("../configs/root.txt", "r")
data_dir = f.read()


def main(dataset_type, ret_model_name, batch_size, num_workers,
         shift_features=False, delta=-1, random_init=False, remove_artifacts=False, is_distil=False, is_batch_norm=False, eval_ckpt_file_name=None, gpu=0):

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if 'cuda' in device.type:
        print(torch.cuda.get_device_name(0))

    model, preprocess, tok = prepare_models_only(ret_model_name, random_init, is_distil, device, is_batch_norm=is_batch_norm, eval_ckpt_file_name=eval_ckpt_file_name)
    model.eval()
    data, classnames, prompt_templates = load_images(dataset_type, ret_model_name, preprocess, batch_size=batch_size, num_workers=num_workers)

    if remove_artifacts:
        image_features_mean = torch.load(f'{data_dir}/mean_features/image_features_mean_{ret_model_name}.pt')
        text_features_mean = torch.load(f'{data_dir}/mean_features/text_features_mean_{ret_model_name}.pt')
    else:
        text_features_mean = None
        
    if dataset_type == 'cifar100':
        for i,template in enumerate(prompt_templates):
            if 'cyclip' in ret_model_name:
                text_inputs = [preprocess.process_text(template(c)) for c in classnames]
                input_ids = torch.cat([t['input_ids'] for t in text_inputs]).to(device)
                attention_mask = torch.cat([t['attention_mask'] for t in text_inputs]).to(device)
            elif 'blip' not in ret_model_name:
                text_inputs = torch.cat([tok.tokenize(template(c)) for c in classnames]).to(device)
            else:
                text_inputs = [tok(template(c)) for c in classnames]
                
            with torch.no_grad():
                if 'cyclip' in ret_model_name:
                    text_features_temp = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
                elif 'blip' not in ret_model_name:
                    text_features_temp = model.encode_text(text_inputs)
                else:
                    sample = {"image": None, "text_input": text_inputs}
                    text_features_temp = model.extract_features(sample, mode="text").text_embeds_proj[:, 0, :]
                    
            text_features_temp = torch.unsqueeze(text_features_temp, 0)
            if i == 0:
                text_features = text_features_temp
            else:
                text_features = torch.cat([text_features, text_features_temp], 0)
                
        text_features = torch.mean(text_features, 0)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        if remove_artifacts:
            text_features = remove_artifact_features(text_features, None, text_features_mean, None, both=False)
        
        tot_num, cor = 0, 0
        for image_input, target in tqdm(data):
            image_input = image_input.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                if 'cyclip' in ret_model_name:
                    image_features = model.get_image_features(image_input)
                elif 'blip' not in ret_model_name:
                    image_features = model.encode_image(image_input)
                else:
                    sample = {"image": image_input, "text_input": None}
                    if 'blip2' in ret_model_name:
                        image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, :, :]
                    else:
                        image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0, :]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if remove_artifacts:
                image_features = remove_artifact_features(image_features, None, image_features_mean, None, both=False)
                
            if shift_features:
                image_features, text_features = move_features(image_features, text_features, delta=delta, direction_vec=None)
        
            logits = 100.0 * image_features @ text_features.T
            if 'blip2' in ret_model_name:
                logits = torch.max(logits, dim=1)[0]
                
            _, preds = torch.max(logits, dim=1)
            cor += accuracy_score(preds.cpu(), target.cpu(), normalize=False)
            tot_num += len(preds)

        print("Average Accuracy over 18 prompts: ", np.round(cor*100/tot_num, decimals = 2), "%")
        print()

    elif dataset_type == 'birdsnap':
        if 'cyclip' in ret_model_name:
            classifier = zero_shot_classifier(model, preprocess, ret_model_name, classnames, prompt_templates, remove_artifacts, text_features_mean, device)
        else:
            classifier = zero_shot_classifier(model, tok, ret_model_name, classnames, prompt_templates, remove_artifacts, text_features_mean, device)
            
        with torch.no_grad():
            all_logits = []
            all_targets = []
            
            for image_input, target in tqdm(data):
                image_input = image_input.to(device)
                target = target.to(device)

                if 'cyclip' in ret_model_name:
                    image_features = model.get_image_features(image_input)
                elif 'blip' not in ret_model_name:
                    image_features = model.encode_image(image_input)
                else:
                    sample = {"image": image_input, "text_input": None}
                    if 'blip2' in ret_model_name:
                        image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, :, :]
                    else:
                        image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0, :]
                    
                image_features /= image_features.norm(dim=-1, keepdim=True)

                if shift_features:
                    image_features, classifier = move_features(image_features, classifier, delta=delta, direction_vec=None, classifier=True)
                elif remove_artifacts:
                    image_features = remove_artifact_features(image_features, None, image_features_mean, None, both=False)
                logits = image_features @ classifier
                if 'blip2' in ret_model_name:
                    logits = torch.max(logits, dim=1)[0]
    
                all_logits.append(logits.cpu())
                all_targets.append(target.cpu())
    
            all_logits = torch.cat(all_logits).numpy()
            all_targets = torch.cat(all_targets).numpy()
            
            acc = accuracy_score(all_targets, all_logits.argmax(axis=1)) * 100.0
            print("Top-1 Accuracy: ", acc, "%")
            print()
            
            acc = balanced_accuracy_score(all_targets, all_logits.argmax(axis=1)) * 100.0
            print("Top-1 Balanced Accuracy: ", acc, "%")

if __name__ == '__main__':
    args = get_args_4_evaluate_classification()
    
    dataset_type = args.dataset_type 
    ret_model_name = args.ret_model_name
    batch_size = args.batch_size
    num_workers = args.num_workers
    shift_features = args.shift_features
    delta = args.delta
    random_init = args.random_init
    remove_artifacts = args.remove_artifacts
    is_distil = args.is_distil
    is_batch_norm = args.is_batch_norm
    eval_ckpt_file_name=args.ckpt_file_name
    gpu = args.gpu

    main(dataset_type, ret_model_name, batch_size, num_workers, shift_features, delta, random_init, remove_artifacts, is_distil, is_batch_norm, eval_ckpt_file_name, gpu)