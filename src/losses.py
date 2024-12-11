import torch
import time
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def get_loss(umodel, outputs, criterion, options):  
    if(options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[:len(outputs.image_embeds) // 2], outputs.image_embeds[len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
            
    if(options.distributed):
        if(options.inmodal):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            augmented_gathered_image_embeds = [torch.zeros_like(augmented_image_embeds) for _ in range(options.num_devices)]
            augmented_gathered_text_embeds = [torch.zeros_like(augmented_text_embeds) for _ in range(options.num_devices)]
            
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(augmented_gathered_image_embeds, augmented_image_embeds)
            dist.all_gather(augmented_gathered_text_embeds, augmented_text_embeds)
            
            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
            augmented_image_embeds = torch.cat(augmented_gathered_image_embeds[:options.rank] + [augmented_image_embeds] + augmented_gathered_image_embeds[options.rank + 1:])
            augmented_text_embeds  = torch.cat(augmented_gathered_text_embeds[:options.rank]+ [augmented_text_embeds] + augmented_gathered_text_embeds[options.rank + 1:])      
        else:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
        
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
        
            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
        
    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    if(options.inmodal):
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()

    batch_size = len(logits_text_per_image)
    
    target = torch.arange(batch_size).long().to(options.device, non_blocking = True)
    
    contrastive_loss = torch.tensor(0).to(options.device)
    if(options.inmodal):
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(logits_text_per_augmented_text, target)) / 2
        contrastive_loss = (crossmodal_contrastive_loss + inmodal_contrastive_loss) / 2
    else:
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        contrastive_loss = crossmodal_contrastive_loss
    inmodal_cyclic_loss = torch.tensor(0).to(options.device)
    if(options.cylambda1 > 0):
        logits_image_per_image = umodel.logit_scale.exp() * image_embeds @ image_embeds.t()
        logits_text_per_text = umodel.logit_scale.exp() * text_embeds @ text_embeds.t()
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size
    
    crossmodal_cyclic_loss = torch.tensor(0).to(options.device)
    if(options.cylambda2 > 0):
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size

    cyclic_loss = options.cylambda1 * inmodal_cyclic_loss + options.cylambda2 * crossmodal_cyclic_loss
    loss = contrastive_loss + cyclic_loss
    
    # if(options.use_iiloss):
    #     # iiloss = (criterion(logits_text_per_image, logits_image_per_image) + criterion(logits_image_per_text, logits_image_per_image)) / 2
    #     logits_image_per_image = umodel.logit_scale.exp() * image_embeds @ image_embeds.t()
    #     iiloss = (logits_image_per_text - logits_image_per_image).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size
    
    # if options.use_ttloss:
    #     # ttloss = (criterion(logits_text_per_image, logits_text_per_text) + criterion(logits_image_per_text, logits_text_per_text)) / 2
    #     logits_text_per_text = umodel.logit_scale.exp() * text_embeds @ text_embeds.t()
    #     ttloss = (logits_text_per_image - logits_text_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size

    # cyclic_loss = options.iilosslambda * iiloss + options.ttlosslambda * ttloss
    # loss = contrastive_loss + cyclic_loss
    
    return loss, contrastive_loss, cyclic_loss

def infoLOOB_loss(x, y, i, inv_tau):
    tau = 1 / inv_tau
    k = x @ y.T / tau
    positives = -torch.mean(torch.sum(k * i, dim=1))

    # For logsumexp the zero entries must be equal to a very large negative number
    large_neg = -1000.0
    arg_lse = k * torch.logical_not(i) + i * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))

    return tau * (positives + negatives)


def hopfield(state_patterns, stored_patterns, hopfield_layer):
    retrieved_patterns = hopfield_layer.forward((stored_patterns.unsqueeze(0), state_patterns.unsqueeze(0), stored_patterns.unsqueeze(0))).squeeze()
    # Row vectors -> dim=1 to normalize the row vectors
    retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(dim=1, keepdim=True)
    return retrieved_patterns


def hopfield_retrieval(image_features, text_features, hopfield_layer):
    patterns_xx = hopfield(state_patterns=image_features, stored_patterns=image_features, hopfield_layer=hopfield_layer)
    patterns_yy = hopfield(state_patterns=text_features, stored_patterns=text_features, hopfield_layer=hopfield_layer)
    patterns_xy = hopfield(state_patterns=text_features, stored_patterns=image_features, hopfield_layer=hopfield_layer)
    patterns_yx = hopfield(state_patterns=image_features, stored_patterns=text_features, hopfield_layer=hopfield_layer)
    
    return patterns_xx, patterns_yy, patterns_xy, patterns_yx
    

def cloob(image_features, text_features, inv_tau, hopfield_layer):
    p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(image_features, text_features, hopfield_layer)
    identity = torch.eye(p_xx.shape[0]) > 0.5
    i = identity.to(p_xx.device)
    loss_img = infoLOOB_loss(p_xx, p_xy, i, inv_tau=inv_tau)
    loss_txt = infoLOOB_loss(p_yy, p_yx, i, inv_tau=inv_tau)
    return loss_img + loss_txt