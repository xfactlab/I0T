import os
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator, InitProcessGroupKwargs
from sharegpt4v import share4v_val_dataset, share4v_train_dataset

from tqdm import tqdm
import torch.nn as nn
import sys

from scheduler import cosine_lr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import longclip, longcliponly, longclip_unifalign, longcyclip
from src.args import get_args_4_evaluate_train
from src.load import load_image_captions
from src.utils import set_seed
from src.losses import get_loss
from Github.hopfieldlayers.hflayers import Hopfield
from datetime import timedelta

f = open("../configs/root.txt", "r")
data_dir = f.read()
f = open("../configs/ckpt_file_name.txt", "r")
ckpt_file_name = f.read()
        
class CLIP_Clean_Train():
    def __init__(self, rank, local_rank, args, warmup_length=200):
        self.args = args
        self.rank=rank
        self.local_rank = local_rank
        self.base_model = args.base_model
        
        set_seed(self.args.seed)
        
        self.model_name = args.model_name
        if not args.freeze:
            if 'cyclip' in self.model_name:
                self.model, _ = longcyclip.load_from_clip(self.base_model, device='cpu', download_root=args.download_root, dropout=self.args.dropout, add_batch_norm=self.args.add_batch_norm)
            else:
                self.model, _ = longclip.load_from_clip(self.base_model, device='cpu', download_root=args.download_root, dropout=self.args.dropout, add_batch_norm=self.args.add_batch_norm) 
        else:
            if 'cyclip' in self.model_name:
                self.model, _ = longcyclip.load(f"{data_dir}/checkpoints/{ckpt_file_name}", dropout=self.args.dropout, add_batch_norm=self.args.add_batch_norm)
            else:
                self.model, _ = longclip.load(f"{data_dir}/checkpoints/{ckpt_file_name}", dropout=self.args.dropout, add_batch_norm=self.args.add_batch_norm)  
            
        if 'longcliponly' not in self.model_name:
            self.longonly = False
        else:
            self.longonly = True
        
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model = self.model.cuda()

        if args.freeze:
            print('Freeze pre-trained layers')
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'bn_final_image' in name or 'bn_final_text' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        if args.exp_name == "auto":
            self.logdir = f"{self.args.save_dir}/Vislang/my{self.model_name}/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
        else:
            self.logdir = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        if args.CLOOB:
            self.hopfield_layer = Hopfield(input_size=512,
                                            scaling=14.3,
                                            normalize_hopfield_space=False,
                                            normalize_hopfield_space_affine=False,
                                            normalize_pattern_projection=False,
                                            normalize_pattern_projection_affine=False, 
                                            normalize_state_pattern=False, 
                                            normalize_state_pattern_affine=False, 
                                            normalize_stored_pattern=False, 
                                            normalize_stored_pattern_affine=False,
                                            state_pattern_as_static=True,
                                            pattern_projection_as_static=True,
                                            stored_pattern_as_static=True,
                                            disable_out_projection=True,
                                            num_heads=1,
                                            dropout=False).cuda(local_rank)
        else:
            self.hopfield_layer = None

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = GradScaler()

        trainset = share4v_train_dataset(self.base_model, args.train_dataset_type, args)
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=torch.cuda.device_count(), persistent_workers=True, pin_memory=True)
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(self.train_loader))

        testset = share4v_val_dataset(self.base_model)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=torch.cuda.device_count(), persistent_workers=True, pin_memory=True)
        
        ipg_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=10800)
            )
        
        self.accelerator = Accelerator(kwargs_handlers=[ipg_handler])
        
    def train_epoch(self, dataloader, epoch, start_iter=0):
        running_loss = 0.0
        running_loss_short = 0.0
        num_batches_per_epoch = len(dataloader)
        
        if 'cyclip' in self.model_name:
            self.umodel = self.model.module if self.args.distributed else self.model
            self.criterion = nn.CrossEntropyLoss().to(self.model.device)

        for i, (id, images, texts, short_text) in enumerate(tqdm(dataloader, disable=(self.rank != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            images_short = images.clone()
            texts = longclip.tokenize(texts, truncate=True).to('cuda') 
            if 'longcliponly' not in self.model_name:
                short_text = longclip.tokenize(short_text, truncate=True).to('cuda') 
                
            self.scheduler(step)
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss = self.model(images, texts, short_text, self.rank, self.longonly, self.args, self.hopfield_layer)
                    
            if self.rank == 0:
                self.writer.add_scalar(f"{self.args.run_name}/loss_per_step", loss, step)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        return loss

    @torch.no_grad()
    def test_epoch(self, dataloader):
        temp_corr_dict = dict()
        rank = torch.distributed.get_rank()

        for id, (images, text) in enumerate(tqdm(dataloader, disable=(rank != 0))):

            images = images.cuda()
            image_features = self.model.module.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text = longclip.tokenize(text, truncate=True).to('cuda')
            text_feature = self.model.module.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

            i = 0
            correct = 0
            total = 0

            for i in range(text_feature.shape[0]):
                text = text_feature[i]
                sim = text @ image_features.T
                sim = sim.squeeze()
                correct_i = torch.argmax(sim)

                if i==correct_i:
                    correct = correct + 1
                total = total + 1

        return correct/total
    
    def train(self, resume=False, model_name='longclip'):
        start_epoch = 0
        resume_iter = 0

        for epoch in range(start_epoch, self.num_epoch):  
            self.model.train()
            loss = self.train_epoch(self.train_loader, epoch, start_iter=resume_iter)
            if self.rank == 0:
                self.writer.add_scalar(f"{self.args.run_name}/loss_per_epoch", loss, epoch)
                
                now = datetime.now()
                formatted_date = now.strftime("%m-%d--%H_%M_%S_")
                torch.save(self.model.module.state_dict(), f'{self.args.save_dir}/checkpoints/final/'+str(self.rank)+formatted_date+model_name+'_'+self.args.run_name+'_epoch'+str(epoch)+'.pt')       

            self.model.eval()
            with torch.no_grad():    
                acc = self.test_epoch(self.testloader)
                print("=====================================")
                print(f"test mean of share4v retrieval: {acc}")
                print("=====================================")
        self.writer.flush()
        self.writer.close()

if __name__ == '__main__':
    args, rank, local_rank = get_args_4_evaluate_train()

    trainer = CLIP_Clean_Train(
        rank=rank,
        local_rank=local_rank, 
        args=args,
        warmup_length=args.warmup_length
        )
    trainer.train(resume=args.resume, model_name=args.model_name)