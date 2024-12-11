import os
import math
import torch
import clip 
import nltk
import random
from nltk.util import ngrams
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
import sys
sys.path.insert(0, '../')
from src.features import move_features, remove_artifact_features
import model.clip_cloob.clip_cloob as clip_cloob
f = open("../configs/root.txt", "r")
data_dir = f.read()


def zero_shot_classifier(model, tok, ret_model_name, classnames, templates, remove_artifacts, text_features_mean, device):
    if templates == None:
        templates = [lambda c: f'{c}'] 
        
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates] 

            if 'cyclip' in ret_model_name:
                preprocess = tok
                text_inputs = [preprocess.process_text(t) for t in texts]
                input_ids = torch.cat([t['input_ids'] for t in text_inputs]).to(device)
                attention_mask = torch.cat([t['attention_mask'] for t in text_inputs]).to(device)
                class_embeddings = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
            elif 'blip' not in ret_model_name:
                texts = tok.tokenize(texts).to(device) 
                class_embeddings = model.encode_text(texts)
            else:
                sample = {"image": None, "text_input": texts}
                class_embeddings = model.extract_features(sample, mode="text").text_embeds_proj[:, 0, :]
                
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            if remove_artifacts:
                class_embedding = remove_artifact_features(class_embedding, None, text_features_mean, None, both=False)
        
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights
    

def initialize_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
        elif "bias" in name:
            torch.nn.init.constant_(param.data, 0)
    return model


def calculate_cross_similarity_scores(images, captions, shift_features, random_init, model, output_folder, ret_model_name, dataset_type, batch_size, device,
                                      delta=-1, save=True, remove_artifacts=False, is_distil=False):
    num_images = len(images)
    num_texts = len(captions)

    similarity_matrix = np.zeros((num_images, num_texts))

    tot_epoch_images = math.ceil(num_images/batch_size)
    tot_epoch_texts = math.ceil(num_texts/batch_size)

    if remove_artifacts:
        image_features_mean = torch.load(f'{data_dir}/mean_features/image_features_mean_{ret_model_name}.pt')
        text_features_mean = torch.load(f'{data_dir}/mean_features/text_features_mean_{ret_model_name}.pt')
        
    for i in tqdm(range(0, num_images, batch_size)):
        for j in range(0, num_texts, batch_size):
            batch_images = images[i:i + batch_size]
            batch_texts = captions[j:j + batch_size]

            with torch.no_grad():
                with autocast():
                    if ret_model_name in ['clip', 'pacscore', 'clipcloob']:
                        image_features = torch.cat([model.encode_image(image) for image in batch_images])
                        if ret_model_name == 'clipcloob':
                            text_tokens = clip_cloob.tokenize(batch_texts).to(device)
                        else:
                            text_tokens = clip.tokenize(batch_texts).to(device)
                        text_features = model.encode_text(text_tokens)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    elif ret_model_name == 'cyclip':
                        input_ids = torch.cat([t['input_ids'] for t in batch_texts]).to(device)
                        attention_mask = torch.cat([t['attention_mask'] for t in batch_texts]).to(device)
                        text_features = model.get_text_features(input_ids = input_ids, attention_mask = attention_mask)
                        image_features = model.get_image_features(batch_images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    elif ret_model_name in ['longclip', 'our_longclip', 'our_longcliponly']:
                        if not is_distil:
                            image_features = torch.cat([model.encode_image(image) for image in batch_images])
                            text_features = torch.cat([model.encode_text(text) for text in batch_texts])
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                        else:
                            image_features = torch.cat([model.encode_image(image) for image in batch_images])
                            text_features = torch.cat([model.encode_text(text) for text in batch_texts])
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            
                            image_features_mean = torch.cat([model.encode_image_new(image) for image in batch_images])
                            text_features_mean = torch.cat([model.encode_text_new(text) for text in batch_texts])
                            image_features_mean = image_features_mean[0].repeat(image_features.shape[0], 1)
                            text_features_mean = text_features_mean[0].repeat(text_features.shape[0], 1)
                            
                            image_features = image_features - image_features_mean
                            text_features = text_features - text_features_mean
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                        
                    elif ret_model_name == 'blip':
                        sample = {"image": batch_images, "text_input": batch_texts}
                        image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0, :] # (batch, proj_image_patch=14*14+1=196, hidden_dim=256)
                        text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0, :] # (batch, seq_len=12, hidden_dim=256)
                        image_features = F.normalize(image_features, dim=-1)
                        text_features = F.normalize(text_features, dim=-1)

                    elif 'blip2' in ret_model_name:
                        sample = {"image": batch_images, "text_input": batch_texts}
                        image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, :, :] # (batch, proj_image_patch=32, hidden_dim=256)
                        text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0, :] # (batch, seq_len=1, hidden_dim=256)

                    if shift_features:
                        image_features, text_features = move_features(image_features, text_features, delta=delta, direction_vec=None)
                    elif remove_artifacts:
                        image_features, text_features = remove_artifact_features(image_features, text_features, image_features_mean, text_features_mean)
                                                                      
                    if save:
                        if random_init:
                            save_ret_model_name = f"{ret_model_name}init"
                        elif shift_features:
                            save_ret_model_name = f"{ret_model_name}shift_delta{delta}"
                        elif remove_artifacts:
                            save_ret_model_name = f"{ret_model_name}_0artifact"
                        else:
                            save_ret_model_name = ret_model_name
                            
                        if (batch_size > num_images and batch_size > num_texts):
                            torch.save(image_features, os.path.join(output_folder, f'image_features_{save_ret_model_name}_{dataset_type}.pt'))
                            torch.save(text_features, os.path.join(output_folder, f'text_features_{save_ret_model_name}_{dataset_type}.pt'))
                            
                        elif ((i == batch_size*(tot_epoch_images-1)) and (j == batch_size*(tot_epoch_texts-1))):
                            svg_image_features = torch.cat((svg_image_features, image_features), 0)
                            svg_text_features = torch.cat((svg_text_features, text_features), 0)                        
                            torch.save(svg_image_features, os.path.join(output_folder, f'image_features_{save_ret_model_name}_{dataset_type}.pt'))
                            torch.save(svg_text_features, os.path.join(output_folder, f'text_features_{save_ret_model_name}_{dataset_type}.pt'))

                        elif ((i == 0) and (j == 0)):
                            svg_image_features, svg_text_features = image_features, text_features
                        else:
                            svg_image_features = torch.cat((svg_image_features, image_features), 0)
                            svg_text_features = torch.cat((svg_text_features, text_features), 0)
                        
                    batch_similarity_scores = (image_features @ text_features.T)
                    if 'blip2' in ret_model_name:
                        batch_similarity_scores = torch.max(batch_similarity_scores, dim=1)[0]
                    similarity_matrix[i:i + image_features.shape[0], j:j + text_features.shape[0]] = batch_similarity_scores.cpu().numpy()
         
                torch.cuda.empty_cache()

    return similarity_matrix


def calculate_similarity_scores(images, captions, model, output_folder, ret_model_name, dataset_type, batch_size, device):
    num_images = len(images)
    num_texts = len(captions)

    sim_i2i = np.zeros((num_images, num_images))
    sim_t2t = np.zeros((num_texts, num_texts))

    # Image-to-Image
    for i in tqdm(range(0, num_images, batch_size)):
        for j in tqdm(range(0, num_images, batch_size)):
            batch_images1 = images[i:i + batch_size]
            batch_images2 = images[j:j + batch_size]
    
            with torch.no_grad():
                with autocast():
                    if ret_model_name in ['clip', 'pacscore', 'clipcloob']:
                        image_features1 = torch.cat([model.encode_image(image) for image in batch_images1])         
                        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
                        image_features2 = torch.cat([model.encode_image(image) for image in batch_images2])        
                        image_features2 /= image_features2.norm(dim=-1, keepdim=True)

                    elif ret_model_name == 'cyclip':
                        image_features1 = model.get_image_features(batch_images1)
                        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
                        image_features2 = model.get_image_features(batch_images2)
                        image_features2 /= image_features2.norm(dim=-1, keepdim=True)

                    elif ret_model_name in ['longclip', 'our_longclip', 'our_longcliponly']:
                        image_features1 = torch.cat([model.encode_image(image) for image in batch_images1])
                        image_features2 = torch.cat([model.encode_image(image) for image in batch_images2])
                        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
                        image_features2 /= image_features2.norm(dim=-1, keepdim=True)
                        
                    elif ret_model_name == 'blip':
                        sample1 = {"image": batch_images1}
                        image_features1 = model.extract_features(sample1, mode="image").image_embeds_proj[:, 0, :]
                        image_features1 = F.normalize(image_features1, dim=-1)
                        sample2 = {"image": batch_images2}
                        image_features2 = model.extract_features(sample2, mode="image").image_embeds_proj[:, 0, :]
                        image_features2 = F.normalize(image_features2, dim=-1)
    
                    elif 'blip2' in ret_model_name:
                        sample1 = {"image": batch_images1}
                        image_features1 = model.extract_features(sample1, mode="image").image_embeds_proj[:, :, :]
                        sample2 = {"image": batch_images2}
                        image_features2 = model.extract_features(sample2, mode="image").image_embeds_proj[:, 0, :]

                    batch_similarity_scores = (image_features1 @ image_features2.T)
                    if 'blip2' in ret_model_name:
                        batch_similarity_scores = torch.max(batch_similarity_scores, dim=1)[0]
                    sim_i2i[i:i + image_features1.shape[0], j:j + image_features2.shape[0]] = batch_similarity_scores.cpu().numpy()
         
                torch.cuda.empty_cache()

    # Text-to-Text
    for i in tqdm(range(0, num_texts, batch_size)):
        for j in tqdm(range(0, num_texts, batch_size)):
            batch_texts1 = captions[i:i + batch_size]
            batch_texts2 = captions[j:j + batch_size]
    
            with torch.no_grad():
                with autocast():
                    if ret_model_name in ['clip', 'pacscore']:
                        text_tokens1 = clip.tokenize(batch_texts1).to(device)
                        text_features1 = model.encode_text(text_tokens1) 
                        text_features1 /= text_features1.norm(dim=-1, keepdim=True)
                        text_tokens2 = clip.tokenize(batch_texts2).to(device)
                        text_features2 = model.encode_text(text_tokens2) 
                        text_features2 /= text_features2.norm(dim=-1, keepdim=True)

                    elif ret_model_name == 'cyclip':
                        input_ids1 = torch.cat([t['input_ids'] for t in batch_texts1]).to(device)
                        attention_mask1 = torch.cat([t['attention_mask'] for t in batch_texts1]).to(device)
                        text_features1 = model.get_text_features(input_ids = input_ids1, attention_mask = attention_mask1)
                        text_features1 /= text_features1.norm(dim=-1, keepdim=True)
                        input_ids2 = torch.cat([t['input_ids'] for t in batch_texts2]).to(device)
                        attention_mask2 = torch.cat([t['attention_mask'] for t in batch_texts2]).to(device)
                        text_features2 = model.get_text_features(input_ids = input_ids2, attention_mask = attention_mask2)
                        text_features2 /= text_features2.norm(dim=-1, keepdim=True)

                    elif ret_model_name == 'clipcloob':
                        text_tokens1 = clip_cloob.tokenize(batch_texts1).to(device)
                        text_features1 = model.encode_text(text_tokens1) 
                        text_features1 /= text_features1.norm(dim=-1, keepdim=True)
                        text_tokens2 = clip_cloob.tokenize(batch_texts2).to(device)
                        text_features2 = model.encode_text(text_tokens2) 
                        text_features2 /= text_features2.norm(dim=-1, keepdim=True)

                    elif ret_model_name in ['longclip', 'our_longclip', 'our_longcliponly']:
                        text_features1 = torch.cat([model.encode_text(text) for text in batch_texts1])
                        text_features2 = torch.cat([model.encode_text(text) for text in batch_texts2])
                        text_features1 /= text_features1.norm(dim=-1, keepdim=True)
                        text_features2 /= text_features2.norm(dim=-1, keepdim=True)
                        
                    elif ret_model_name == 'blip':
                        sample1 = {"text_input": batch_texts1}
                        text_features1 = model.extract_features(sample1, mode="text").text_embeds_proj[:, 0, :]
                        text_features1 = F.normalize(text_features1, dim=-1)
                        sample2 = {"text_input": batch_texts2}
                        text_features2 = model.extract_features(sample2, mode="text").text_embeds_proj[:, 0, :]
                        text_features2 = F.normalize(text_features2, dim=-1)
    
                    elif 'blip2' in ret_model_name:
                        sample1 = {"text_input": batch_texts1}
                        text_features1 = model.extract_features(sample1, mode="text").text_embeds_proj[:, 0, :] 
                        sample2 = {"text_input": batch_texts2}
                        text_features2 = model.extract_features(sample2, mode="text").text_embeds_proj[:, 0, :] 
        
                    batch_similarity_scores = (text_features1 @ text_features2.T)
                    sim_t2t[i:i + text_features1.shape[0], j:j + text_features2.shape[0]] = batch_similarity_scores.cpu().numpy()
         
                torch.cuda.empty_cache()

    return sim_i2i, sim_t2t


def calculate_cross_recall(scores, indices, output_folder, ret_model_name, dataset_type, shift_features, random_init, type,
                           save=True, delta=-1, remove_artifacts=False, is_distil=False):
    if save:
        if random_init:
            ret_model_name = f"{ret_model_name}init"
        elif shift_features:
            ret_model_name = f"{ret_model_name}shift_delta{delta}"
        elif remove_artifacts:
            ret_model_name = f"{ret_model_name}_0artifact"
        elif is_distil:
            ret_model_name = f"{ret_model_name}_posthoc"
        torch.save(scores, os.path.join(output_folder, f'scores_{ret_model_name}_{dataset_type}_{type}.pt'))
            
    if type == 'i2t':
        img2txt = indices
        
        # Images->Text
        ranks = np.zeros(scores.shape[0])
        for index, score in enumerate(scores):
            inds = np.argsort(score)[::-1]
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
        
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        return tr1, tr5, tr10

    elif type == 't2i':
        txt2img = indices
        
        # Text->Images
        ranks = np.zeros(scores.shape[0])
        
        for index, score in enumerate(scores):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]
        
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        return ir1, ir5, ir10

        
def calculate_recall(scores, indices, output_folder, model_name, dataset_type, type, save=True):
    if save:
        torch.save(scores, os.path.join(output_folder, f'scores_{model_name}_{dataset_type}_{type}.pt'))
      
    # Image->Images, Text->Texts
    ranks = np.zeros(scores.shape[0])
    
    for index, score in enumerate(scores):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == indices[index])[0][0]
    
    recall1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    recall5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    recall10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return recall1, recall5, recall10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)