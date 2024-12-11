import os
import argparse
import torch
import torch.distributed as dist
import subprocess
f = open("../configs/root.txt", "r")
data_dir = f.read()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_distributed(backend="nccl", port=None):
    num_gpus = torch.cuda.device_count()
    print("# GPUs:", num_gpus)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if "SLURM_JOB_ID" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = rank % num_gpus
        
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{local_rank}')
    
    print("Rank:", rank)
    print("Local rank:", local_rank)
    return rank, local_rank
        

def get_args_4_evaluate_train():
    parser = argparse.ArgumentParser(description="Args for fine-tuning CLIP")

    parser.add_argument('--lr', default=1e-4, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default='ViT-B/32', help="CLIP Base Model") 
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per gpu.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for.")
    parser.add_argument("--resume", default=False, action='store_true', help="resume training from checkpoint.")
    parser.add_argument("--download-root", default=f'{data_dir}/cache', help="CLIP Base Model download root")
    parser.add_argument("--local-rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--model_name', type=str, default='longclip', choices=[
                        'longclip', 'longcliponly', 
                        'longclip_unifalign',  'longcliponly_unifalign',
                        'cyclip_crossonly', 'cyclip_inmodal', 'cyclip', 'longcliponly_cyclip',
                        'longclip_augment', 'longcliponly_augment'])
    parser.add_argument('--distributed', type=str2bool, default='n', help='distributed training')
    parser.add_argument('--num_devices', type=int, default=2, help='number of devices')
    parser.add_argument('--inmodal', type=str2bool, default='n', help='inmodal training in CyClip')
    parser.add_argument('--cylambda1', type=float, default=0.25, help='lambda1 in CyClip')
    parser.add_argument('--cylambda2', type=float, default=0.25, help='lambda2 in CyClip')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--iilosslambda', type=float, default=0.25, help='lambda for II loss')
    parser.add_argument('--ttlosslambda', type=float, default=0.25, help='lambda for TT loss')
    parser.add_argument("--train_dataset_type", default='coco', choices=['llava', 'sam', 'coco', 'all']) 
    parser.add_argument("--run_name", default='', type=str, help="run name for model saving")
    parser.add_argument("--feature_norm", default=False, action='store_true', help="normalize features")
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--save_dir", default=None, type=str, help="dir name for logging and saving checkpoints")
    
    parser.add_argument("--dropout", default=False, type=str2bool, help="using dropout in CyClip")
    parser.add_argument("--MPCSE", default=False, type=str2bool, help="using MPCSE")
    parser.add_argument("--CLOOB", default=False, type=str2bool, help="using CLOOB")
    parser.add_argument("--color_change", default=False, type=str2bool, help="(additionally) using no colored-captions")
    parser.add_argument("--freeze", default=False, type=str2bool, help="freeze pre-trained layers")
    parser.add_argument("--align_loss", default=False, type=str2bool, help="using align loss")
    parser.add_argument("--uniform_loss", default=False, type=str2bool, help="using uniform loss")
    parser.add_argument("--return_only_cyclic", default=False, type=str2bool, help="return only cyclic loss")
    parser.add_argument("--add_batch_norm", default=False, type=str2bool, help="add batchnorm to the model")
    
    args = parser.parse_args()
    rank, local_rank = setup_distributed()
    print("DDP Done")
    
    return args, rank, local_rank

    
def get_args_4_evaluate_correlation():
    parser = argparse.ArgumentParser(description='Args for image captioning metric evaluation')
    
    parser.add_argument('--datasets', type=str, default='flickr8k_cf', choices=['flickr8k_cf', 'flickr8k_expert'])
    parser.add_argument('--clip_model', type=str, default='our_longclip', choices=['pacscore', 'clip', 'longclip', 'our_longclip', 'our_longcliponly', 'clipcloob', 'cyclip', 'blip', 'blip2'])    
    parser.add_argument('--batch_size_compute_score', type=int, default=128)
    parser.add_argument('--compute_refpac', action='store_true')
    parser.add_argument('--shift_features', type=str2bool, default='n')
    parser.add_argument('--delta', type=int, default=-1)
    parser.add_argument('--random_init', type=str2bool, default='n')
    parser.add_argument('--remove_artifacts', type=str2bool, default='n')
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--is_distil", default=False, type=str2bool, help="after training (False) or after post-training (True)")
    parser.add_argument("--is_batch_norm", default=False, type=str2bool, help="using batch norm")
    parser.add_argument('--ckpt_file_name', type=str, default=None)    
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    return args

    
def get_args_4_evaluate_classification():
    parser = argparse.ArgumentParser(description="Args for classification evaluation")

    parser.add_argument('--dataset_type', type=str, default='cifar100', choices=['cifar100', 'birdsnap'])
    parser.add_argument('--ret_model_name', type=str, default='our_longclip', choices=['pacscore', 'clip', 'longclip', 'our_longclip', 'our_longcliponly', 'clipcloob', 'cyclip', 'blip', 'blip2'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shift_features', type=str2bool, default='n')
    parser.add_argument('--delta', type=int, default=-1)
    parser.add_argument('--random_init', type=str2bool, default='n')
    parser.add_argument('--remove_artifacts', type=str2bool, default='n')
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--is_distil", default=False, type=str2bool, help="after training (False) or after post-training (True)")
    parser.add_argument("--is_batch_norm", default=False, type=str2bool, help="using batch norm")
    parser.add_argument('--ckpt_file_name', type=str, default=None)    
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    return args


def get_args_4_evaluate_retrieval():
    parser = argparse.ArgumentParser(description="Args for ITR task evaluation")

    parser.add_argument('--dataset_type', type=str, default='coco', choices=['coco', 'flickr', 'nocaps'])
    parser.add_argument('--model_name', type=str, default='gold')
    parser.add_argument('--ret_model_name', type=str, default='our_longclip', choices=['pacscore', 'clip', 'longclip', 'our_longclip', 'our_longcliponly', 'clipcloob', 'cyclip', 'blip', 'blip2'])
    parser.add_argument('--num_caption', type=int, default=1)
    parser.add_argument('--shift_features', type=str2bool, default='n')
    parser.add_argument('--delta', type=int, default=-1)
    parser.add_argument('--random_init', type=str2bool, default='n')
    parser.add_argument('--output_folder', type=str, default='../tensors')
    parser.add_argument('--remove_artifacts', type=str2bool, default='n')
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--is_distil", default=False, type=str2bool, help="after training (False) or after post-training (True)")
    parser.add_argument("--is_batch_norm", default=False, type=str2bool, help="using batch norm")
    parser.add_argument('--ckpt_file_name', type=str, default=None)    
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()

    return args