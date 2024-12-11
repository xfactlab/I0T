from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import collections
import sys
sys.path.insert(0, '../')
from src.features import move_features, remove_artifact_features
f = open("../configs/root.txt", "r")
data_dir = f.read()
        
class CapDataset(torch.utils.data.Dataset):
    def __init__(self, data, model_name, tok, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '
        self.model_name = model_name
        self.tok = tok

    def __getitem__(self, idx):
        c_data = self.data[idx]
        if self.model_name == 'cyclip':
            c_data = self.tok(self.prefix + c_data)
        elif self.model_name == 'longclip':
            c_data = self.tok.tokenize(self.prefix + c_data).squeeze()
        elif 'blip' in self.model_name:
            c_data = self.tok(self.prefix + c_data)
        else:
            c_data = self.tok.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if transform:
            self.preprocess = transform
        else:
            self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(model_name, captions, model, tok, device, batch_size=128, num_workers=2):
    data = torch.utils.data.DataLoader(CapDataset(captions, model_name, tok), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            if model_name == 'cyclip':
                b = b['caption']
                input_ids = b['input_ids'].squeeze().to(device)
                attention_mask = b['attention_mask'].squeeze().to(device)
                all_text_features.append(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).detach().cpu().numpy())
            elif 'blip' not in model_name:
                b = b['caption'].to(device)
                all_text_features.append(model.encode_text(b).detach().cpu().numpy())
            else:
                sample = {"image": None, "text_input": b['caption']}
                all_text_features.append(model.extract_features(sample, mode="text").text_embeds_proj[:, 0, :].detach().cpu().numpy())
                
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(model_name, images, model, transform, device, batch_size=64, num_workers=4):
    data = torch.utils.data.DataLoader(ImageDataset(images, transform), batch_size=batch_size, num_workers=num_workers,
                                       shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if model_name == 'cyclip':
                all_image_features.append(model.get_image_features(b).detach().cpu().numpy())
            elif 'blip' not in model_name:
                all_image_features.append(model.encode_image(b).detach().cpu().numpy())
            else:
                sample = {"image": b, "text_input": None}
                if 'blip' in model_name:
                    image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0, :]
                all_image_features.append(image_features.detach().cpu().numpy())
                
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def PACScore(model_name, model, transform, tok, images, candidates, shift_features, delta, remove_artifacts, device, w=2.0):
    '''
    compute the unreferenced PAC score.
    '''
    len_candidates = [len(c.split()) for c in candidates] 
    if isinstance(images, list):
        # extracting image features
        if model_name == 'cyclip':
            images = extract_all_images(model_name, images, model, transform.process_image, device) 
        else:
            images = extract_all_images(model_name, images, model, transform, device) # (num_img, d_model)

    if model_name == 'cyclip':
        tok = transform.process_text
    candidates = extract_all_captions(model_name, candidates, model, tok, device)

    images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
    candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    if shift_features:
        images, candidates = move_features(images, candidates, delta=delta, direction_vec=None)
    elif remove_artifacts:
        image_features_mean = torch.load(f'{data_dir}/mean_features/image_features_mean_{model_name}.pt')
        text_features_mean = torch.load(f'{data_dir}/mean_features/text_features_mean_{model_name}.pt')
        
        images = remove_artifact_features(images, None, image_features_mean, None, both=False)       
        candidates = remove_artifact_features(candidates, None, text_features_mean, None, both=False)       

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates, len_candidates


def RefPACScore(model_name, model, transform, tok, references, candidates, shift_features, delta, remove_artifacts, device, len_candidates):
    '''
    compute the RefPAC score, extracting only the reference captions.
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(model_name, candidates, model, tok, device)

    len_references = []
    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        len_r = [len(r.split()) for r in refs]
        len_references.append(len_r)
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    if model_name == 'cyclip':
        tok = transform.process_text
    flattened_refs = extract_all_captions(model_name, flattened_refs, model, tok, device)
    
    candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))
    flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs ** 2, axis=1, keepdims=True))

    if shift_features:
        candidates, flattened_refs = move_features(candidates, flattened_refs, delta=delta, direction_vec=None)
    elif remove_artifacts:
        text_features_mean = np.load(f'{data_dir}/mean_features/text_features_mean_{model_name}.npy')
        
        candidates = remove_artifact_features(candidates, None, text_features_mean, None, both=False)       
        flattened_refs = remove_artifact_features(flattened_refs, None, text_features_mean, None, both=False)       

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, (cand, l_ref, l_cand) in enumerate(zip(candidates, len_references, len_candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())

        per.append(np.max(all_sims))

    return np.mean(per), per