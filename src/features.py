import numpy as np
import torch

def move_features(orig_image_features, orig_text_features, delta, direction_vec=None, classifier=False):
    if not torch.is_tensor(orig_image_features) and not torch.is_tensor(orig_text_features):
        image_features = torch.from_numpy(orig_image_features)
        text_features = torch.from_numpy(orig_text_features)
    elif classifier:
        image_features = orig_image_features
        text_features = torch.transpose(orig_text_features, 0, 1)
    else:
        image_features = orig_image_features
        text_features = orig_text_features
        
    if direction_vec is None:
        modality_gap = image_features.mean(axis=0) - text_features.mean(axis=0)
        modality_gap = modality_gap / modality_gap.norm()
        direction_vec = modality_gap
    
    modified_text_features = text_features + 0.5 * delta * direction_vec
    modified_text_features /= modified_text_features.norm(dim=-1, keepdim=True)

    modified_image_features = image_features - 0.5 * delta * direction_vec
    modified_image_features /= modified_image_features.norm(dim=-1, keepdim=True)

    if classifier:
        modified_text_features = torch.transpose(modified_text_features, 0, 1)
    elif not torch.is_tensor(orig_image_features) and not torch.is_tensor(orig_text_features):
        modified_image_features = modified_image_features.numpy()
        modified_text_features = modified_text_features.numpy()

    return modified_image_features, modified_text_features


def remove_artifact_features(orig_image_features, orig_text_features, image_features_mean, text_features_mean, both=True):
    if both:
        if isinstance(orig_image_features, np.ndarray):
            image_features = torch.from_numpy(orig_image_features).cuda()
            text_features = torch.from_numpy(orig_text_features).cuda()
        else:
            image_features = orig_image_features.cuda()
            text_features = orig_text_features.cuda()

        image_features_mean = image_features_mean.repeat(image_features.shape[0], 1).cuda()
        text_features_mean = text_features_mean.repeat(text_features.shape[0], 1).cuda()

        new_image_features = image_features - image_features_mean
        new_text_features = text_features - text_features_mean

        new_image_features /= new_image_features.norm(dim=-1, keepdim=True)
        new_text_features /= new_text_features.norm(dim=-1, keepdim=True)

        if isinstance(orig_image_features, np.ndarray):
            new_image_features = new_image_features.detach().cpu().numpy()
            new_text_features = new_text_features.detach().cpu().numpy()
            
        return new_image_features, new_text_features
        
    else:              
        if len(orig_image_features.shape) == 1:
            features = orig_image_features.unsqueeze(0).cuda()
        
        if isinstance(orig_image_features, np.ndarray):
            features = torch.from_numpy(orig_image_features).cuda()
        else:
            features = orig_image_features.cuda()
            
        if isinstance(image_features_mean, np.ndarray):
            image_features_mean = torch.from_numpy(image_features_mean).cuda()
        else:
            image_features_mean = image_features_mean.cuda()

        features_mean = image_features_mean.repeat(features.shape[0], 1).cuda()

        new_features = features - features_mean
        new_features /= new_features.norm(dim=-1, keepdim=True)

        if isinstance(orig_image_features, np.ndarray):
            new_features = new_features.detach().cpu().numpy()

        if len(orig_image_features.shape) == 1:
            new_features = new_features[0]
    
        return new_features