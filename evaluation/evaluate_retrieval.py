import json
from tqdm import tqdm
import os
from PIL import Image
import torch
import numpy as np
import sys
sys.path.insert(0, '../')
from src.args import get_args_4_evaluate_retrieval
from src.utils import initialize_weights, calculate_cross_similarity_scores, calculate_cross_recall, calculate_similarity_scores, calculate_recall
from src.load import load_image_captions
from src.prepare import prepare_models

def main(dataset_type, ret_model_name, model_name, output_folder,
        num_caption=1, shift_features=False, delta=-1, random_init=False, remove_artifacts=False, is_distil=False, is_batch_norm=False, eval_ckpt_file_name=None, gpu=0):

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if 'cuda' in device.type:
        print(torch.cuda.get_device_name(0))

    principal_dir, image_captions, num_data = load_image_captions(dataset_type, model_name)

    images = []
    texts = []
    
    img2img, img2txt = {}, {}
    txt2txt, txt2img = {}, {}

    if ret_model_name in ['clip', 'pacscore', 'clipcloob']:
        max_length = 330
    else:
        max_length = 1000000
    txt_id = 0
    
    for img_id, (image_path, captions) in enumerate(image_captions.items()):
        path_image = os.path.join(principal_dir, image_path)
        image = Image.open(path_image).convert("RGB")
        images.append(image)
        img2txt[img_id] = []
        img2img[img_id] = img_id
    
        for caption in captions[:num_caption]:
            texts.append(caption[:max_length])
            img2txt[img_id].append(txt_id)
            txt2img[txt_id] = img_id
            txt2txt[txt_id] = txt_id
            txt_id += 1

    images, texts, model, batch_size = prepare_models(images, texts, ret_model_name, dataset_type, random_init, is_distil, device, is_batch_norm=is_batch_norm, eval_ckpt_file_name=eval_ckpt_file_name)


    # Standard I2T & T2I tasks
    scores_i2t = calculate_cross_similarity_scores(images, texts, shift_features, random_init, model, output_folder, ret_model_name, dataset_type,
                                                    batch_size=batch_size, device=device, delta=delta, remove_artifacts=remove_artifacts, is_distil=is_distil)
    print('I2T shape: ', scores_i2t.shape)
    
    tr1, tr5, tr10 = calculate_cross_recall(scores_i2t, img2txt, output_folder, ret_model_name, dataset_type, shift_features, random_init,  type='i2t', delta=delta, remove_artifacts=remove_artifacts, is_distil=is_distil)
    print('Image-to-Text Retrieval Result')
    print(tr1, tr5, tr10)
    
    scores_t2i = np.transpose(scores_i2t)
    print('T2I shape: ', scores_t2i.shape)
    ir1, ir5, ir10 = calculate_cross_recall(scores_t2i, txt2img, output_folder, ret_model_name, dataset_type, shift_features, random_init,  type='t2i', delta=delta, remove_artifacts=remove_artifacts, is_distil=is_distil)
    print('Text-to-Image Retrieval Result')
    print(ir1, ir5, ir10)


if __name__ == '__main__':
    args = get_args_4_evaluate_retrieval()
    
    dataset_type = args.dataset_type 
    ret_model_name = args.ret_model_name
    model_name = args.model_name
    output_folder = args.output_folder
    num_caption = args.num_caption
    shift_features = args.shift_features
    delta = args.delta
    random_init = args.random_init
    remove_artifacts = args.remove_artifacts
    is_distil = args.is_distil
    is_batch_norm = args.is_batch_norm
    eval_ckpt_file_name = args.ckpt_file_name
    gpu = args.gpu

    main(dataset_type, ret_model_name, model_name, output_folder,
         num_caption, shift_features, delta, random_init, remove_artifacts, is_distil, is_batch_norm=is_batch_norm, eval_ckpt_file_name=eval_ckpt_file_name, gpu=gpu)