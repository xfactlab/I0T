import os
import json
from PIL import Image
import torch
import sys
sys.path.insert(0, '../')
from src.utils import initialize_weights
f = open("../configs/root.txt", "r")
data_dir = f.read()

def prepare_models_only(ret_model_name, random_init, is_distil, device, is_batch_norm=False, eval_ckpt_file_name=None):
    if ret_model_name in ['clip', 'pacscore']:
        import clip
        model, preprocess = clip.load('ViT-B/32', device)
        if 'pacscore' in ret_model_name:
            checkpoint = torch.load(f"{data_dir}/pacscore_checkpoints/clip_ViT-B-32.pth")
            model.load_state_dict(checkpoint['state_dict'])
        if random_init:
            model = initialize_weights(model)
        model, item1, item2 = model, preprocess, clip

    elif ret_model_name in ['longclip', 'longcliponly']:
        from model import longclip
        # model, preprocess = longclip.load_from_clip("ViT-B/32", device=device, download_root=f"{data_dir}/cache") #longclipinit (Initialized LongCLIP = CLIP)
        model, preprocess = longclip.load(f"{data_dir}/longclip-B-32.pt", device=device, add_batch_norm=is_batch_norm) # longclip-B.pt
        model, item1, item2 = model, preprocess, longclip

    elif ret_model_name in ['our_longclip', 'our_longcliponly']:
        from model import longclip, longcyclip
        if 'cyclip' in eval_ckpt_file_name:
            model = longcyclip
        else:
            model = longclip
        if not is_distil:
            model, preprocess = model.load(f"{data_dir}/checkpoints/{eval_ckpt_file_name}", device=device, add_batch_norm=is_batch_norm)    
        else:
            model, preprocess = model.load(f"{data_dir}/checkpoints/distil_checkpoints/{eval_ckpt_file_name}", device=device, add_batch_norm=is_batch_norm) 
        model, item1, item2 = model, preprocess, longclip

    elif ret_model_name == 'blip':
        from lavis.models import load_model_and_preprocess
        model, vis_processors, txt_processors = load_model_and_preprocess("blip_feature_extractor", 'base', is_eval=True)
        model, item1, item2 = model, vis_processors['eval'], txt_processors['eval']
        
    elif ret_model_name == 'blip2':
        from lavis.models import load_model_and_preprocess
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
        model, item1, item2 = model, vis_processors['eval'], txt_processors['eval']

    elif ret_model_name == 'blip2coco':
        from lavis.models import load_model_and_preprocess
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device)
        model, item1, item2 = model, vis_processors['eval'], txt_processors['eval']

    elif ret_model_name == 'clipcloob':
        import model.clip_cloob.clip_cloob as clip_cloob
        from model.clip_cloob.clip_cloob import _transform
        from model.clip_cloob.cloob_model import CLIPGeneral

        checkpoint_path = f'{data_dir}/cloob/cloob_rn50_yfcc_epoch_28.pt' # clip_rn50_yfcc_epoch_28.pt' # cloob_rn50_yfcc_epoch_28.pt
        checkpoint = torch.load(checkpoint_path)
        src_path = '../'
        model_config_file = os.path.join(src_path, 'training/model_configs/', checkpoint['model_config_file'])
        assert os.path.exists(model_config_file)
        
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        model = CLIPGeneral(**model_info)
        preprocess= _transform(model.visual.input_resolution, is_train=False)

        sd = checkpoint["state_dict"]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        if 'logit_scale_hopfield' in sd:
            sd.pop('logit_scale_hopfield', None)
        model.load_state_dict(sd)
        model, item1, item2 = model, preprocess, clip_cloob

    elif ret_model_name == 'cyclip':
        from model.clip import load as load_model
        model, processor = load_model(name='RN50', pretrained=False)
        checkpoint = f'{data_dir}/cyclip.pt/best.pt' # clip.pt/best.pt' # cyclip-3M.pt/best.pt'
        state_dict = torch.load(checkpoint, map_location = device)["state_dict"]
        if(next(iter(state_dict.items()))[0].startswith("module")):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        model, item1, item2 = model, processor, None

    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    print("# of parameters: ", mem_params)

    model.to(device)
    model.eval()
    
    if random_init:
        model = initialize_weights(model)
        
    return model, item1, item2

    
def prepare_models(images, texts, ret_model_name, dataset_type, random_init, is_distil, device, is_batch_norm, eval_ckpt_file_name):
    batch_size = 256
    
    if ret_model_name in ['clip', 'pacscore']:
        model, preprocess, clip = prepare_models_only(ret_model_name, random_init, is_distil, device)
        images = [preprocess(image).unsqueeze(0).to(device) for image in images]
        batch_size = 1024

    elif ret_model_name in ['longclip', 'our_longclip', 'our_longcliponly', 'longcliponly']:
        model, preprocess, longclip = prepare_models_only(ret_model_name, random_init, is_distil, device, is_batch_norm, eval_ckpt_file_name=eval_ckpt_file_name)
        images = [preprocess(image).unsqueeze(0).to(device) for image in images]
        texts = [longclip.tokenize(text).to(device) for text in texts]
        batch_size = 1024

    elif ret_model_name == 'blip':
        model, vis_processors, txt_processors = prepare_models_only(ret_model_name, random_init, is_distil, device)
        images = [vis_processors(image) for image in images]
        images = torch.stack(images).to(device)
        texts = [txt_processors(text) for text in texts]
        
    elif ret_model_name == 'blip2':
        model, vis_processors, txt_processors = prepare_models_only(ret_model_name, random_init, is_distil, device)
        images = [vis_processors['eval'](image) for image in images]
        images = torch.stack(images).to(device)
        texts = [txt_processors["eval"](text) for text in texts]

    elif ret_model_name == 'blip2coco':
        model, vis_processors, txt_processors = prepare_models_only(ret_model_name, random_init, is_distil, device)
        images = [vis_processors['eval'](image) for image in images]
        images = torch.stack(images).to(device)
        texts = [txt_processors["eval"](text) for text in texts]
        if dataset_type in ['coco', 'nocaps']:
            batch_size = 16
        elif dataset_type == 'flickr':
            batch_size = 64

    elif ret_model_name == 'clipcloob':
        model, preprocess, _ = prepare_models_only(ret_model_name, random_init, is_distil, device)
        images = [preprocess(image).unsqueeze(0).to(device) for image in images]
        batch_size = 1024

    elif ret_model_name == 'cyclip':
        model, processor, _ = prepare_models_only(ret_model_name, random_init, is_distil, device)
        images = [processor.process_image(image) for image in images]
        images = torch.stack(images).to(device)
        texts = [processor.process_text(text) for text in texts]

    return images, texts, model, batch_size