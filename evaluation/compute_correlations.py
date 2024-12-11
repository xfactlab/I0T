import os
import argparse
import torch
import scipy.stats
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '../Github/pacscore/')
# import evaluation
# from models.clip import clip
from utils import collate_fn
# from models import open_clip
from data import Flickr8k

sys.path.insert(0, '../')
from src.args import get_args_4_evaluate_correlation
from src.prepare import prepare_models_only
from src.pac_score import PACScore, RefPACScore
f = open("../configs/root.txt", "r")
data_dir = f.read()


_MODELS = {
    "pacscore": f"{data_dir}/pacscore_checkpoints/clip_ViT-B-32.pth",
    "pacscore_open_clip_ViT-L/14": f"{data_dir}/pacscore_checkpoints/openClip_ViT-L-14.pth",
    "long_clip": f"{data_dir}/longclip-B.pt",
    "cloob": f"{data_dir}/cloob/cloob_rn50_yfcc_epoch_28.pt"
}

def compute_correlation_scores(model_name, dataloader, model, preprocess, tok,
                               compute_refpac, shift_features, delta, random_init, remove_artifacts, device):
    gen = {}
    gts = {}

    human_scores = list()
    ims_cs = list()
    gen_cs = list()
    gts_cs = list()
    all_scores = dict()
    model.eval()

    for it, (images, candidates, references, scores) in enumerate(iter(dataloader)):
        for i, (im_i, gts_i, gen_i, score_i) in enumerate(zip(images, references, candidates, scores)):
            gen['%d_%d' % (it, i)] = [gen_i, ]
            gts['%d_%d' % (it, i)] = gts_i

            ims_cs.append(im_i)
            gen_cs.append(gen_i)
            gts_cs.append(gts_i)
            human_scores.append(score_i)

    
    # Reference-free metric
    _, pac_scores, candidate_feats, len_candidates = PACScore(model_name, model, preprocess, tok, ims_cs, gen_cs, shift_features, delta, remove_artifacts, device, w=2.0)
    all_scores[model_name] = pac_scores
    
    # Reference-based metric
    if compute_refpac:
        _, per_instance_text_text = RefPACScore(model_name, model, preprocess, tok, gts_cs, candidate_feats, shift_features, delta, remove_artifacts, device, torch.tensor(len_candidates))
        refpac_scores = 2 * pac_scores * per_instance_text_text / (pac_scores + per_instance_text_text)
        all_scores[f'Ref-{model_name}'] = refpac_scores

    for k, v in all_scores.items():
        kendalltau_b = 100 * scipy.stats.kendalltau(v, human_scores, variant='b')[0]
        kendalltau_c = 100 * scipy.stats.kendalltau(v, human_scores, variant='c')[0]
        pearson_r = 100 * scipy.stats.pearsonr(v, human_scores)[0]
        print('%s \t Kendall Tau-b: %.3f \t  Kendall Tau-c: %.3f \t Pearson R: %.3f'
              % (k, kendalltau_b, kendalltau_c, pearson_r))

def compute_scores(model_name, model, preprocess, tok, d, batch_size_compute_score, compute_refpac, shift_features, delta, remove_artifacts, device):
    print("Computing correlation scores on dataset: " + d)
    
    if d == 'flickr8k_expert':
        dataset = Flickr8k(json_file='flickr8k.json')
        dataloader = DataLoader(dataset, batch_size=batch_size_compute_score, shuffle=False, collate_fn=collate_fn)
    elif d == 'flickr8k_cf':
        dataset = Flickr8k(json_file='crowdflower_flickr8k.json')
        dataloader = DataLoader(dataset, batch_size=batch_size_compute_score, shuffle=False, collate_fn=collate_fn)
    
    compute_correlation_scores(model_name, dataloader, model, preprocess, tok, compute_refpac, shift_features, delta, random_init, remove_artifacts, device)


def main(datasets, clip_model, batch_size_compute_score, compute_refpac, shift_features, delta, random_init, remove_artifacts, is_distil, is_batch_norm, eval_ckpt_file_name, gpu):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if 'cuda' in device.type :
        print(torch.cuda.get_device_name(0))

    model, preprocess, tok = prepare_models_only(clip_model, random_init, is_distil, device, is_batch_norm=is_batch_norm, eval_ckpt_file_name=eval_ckpt_file_name)
    model.eval()

    compute_scores(clip_model, model, preprocess, tok, datasets, batch_size_compute_score, compute_refpac, shift_features, delta, remove_artifacts, device)
    

if __name__ == '__main__':
    args = get_args_4_evaluate_correlation()

    datasets = args.datasets 
    clip_model = args.clip_model
    batch_size_compute_score = args.batch_size_compute_score
    compute_refpac = args.compute_refpac
    shift_features = args.shift_features
    delta = args.delta
    random_init = args.random_init
    remove_artifacts = args.remove_artifacts
    is_distil = args.is_distil
    is_batch_norm = args.is_batch_norm
    eval_ckpt_file_name=args.ckpt_file_name
    gpu = args.gpu

    main(datasets, clip_model, batch_size_compute_score, compute_refpac, shift_features, delta, random_init, remove_artifacts, is_distil, is_batch_norm, eval_ckpt_file_name, gpu)
    
    