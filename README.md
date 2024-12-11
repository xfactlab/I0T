# I0T: Embedding Standardization Method Towards Zero Modality Gap


## 0. ``Environment`` 

```none
cd environments
conda env create -f environment.yml
conda activate lric
```

```none
cd configs
touch root.txt
vim root.txt # Insert your <data_dir>
touch subroot.txt
vim subroot.txt # Insert your <data_dir_sub>
```


## 1. ``Baselines`` 

We compare our model with the following baselines: [pacscore](https://github.com/aimagelab/pacscore), [Long-CLIP](https://github.com/beichenzbc/Long-CLIP), [BLIP](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip_models/blip_feature_extractor.py), [BLIP2](https://github.com/salesforce/LAVIS/blob/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc/lavis/models/blip2_models/blip2_qformer.py), [cloob](https://github.com/ml-jku/cloob), and [CyCLIP](https://github.com/goel-shashank/CyCLIP). The current checkpoint structure:

```none
<data_dir>
├── pacscore_checkpoints
│   ├── clip_ViT-B-32.pth
├── longclip-B-32.pt
├── cloob
│   ├── cloob_rn50_yfcc_epoch_28.pt
├── cyclip.pt
│   ├── best.pt
```


## 2. ``Datasets`` 

### Training dataset

Download [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md), following the same training procedure as [Long-CLIP](https://github.com/beichenzbc/Long-CLIP). The current data structure:

```none
<data_dir>/sharegpt4v
├── share-captioner_coco_lcs_sam_1246k_1107.json
├── data
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
│   ├── sam
│   │   ├── images
├── ...
```

### Evaluation datasets

- Retrieval
  ```none
  COCO
  <data_dir_sub>/PnP-VQA/lavis/coco/images/val2014
  <data_dir_sub>/Evaluation/COCO_Captions/NewCaptions/COCO_GOLD.json
  
  Flickr
  <data_dir_sub>/PnP-VQA/lavis/flickr30k/images/flickr30k-images
  <data_dir_sub>/Evaluation/Flickr_Captions/NewCaptions/Flickr_GOLD.json

  Nocaps
  <data_dir_sub>/PnP-VQA/lavis/nocaps/images
  <data_dir_sub>/Evaluation/Nocaps_Captions/NewCaptions/NoCaps_GOLD.json
  ```
  
- Classification
  ```none
  # CIFAR100
  ~/.cache
  
  # Birdsnap
  <data_dir>/zeroshot_datasets/birdsnap/test
  ```
  
- Metric
  ```none
  # Flickr8K-Expert
  <data_dir>/pacscore_datasets/flickr8k/images
  <data_dir>/pacscore_datasets/flickr8k/flickr8k.json
  
  # Flickr8K-CF
  <data_dir>/pacscore_datasets/flickr8k/images
  <data_dir>/pacscore_datasets/flickr8k/crowdflower_flickr8k.json
  ```


## 3. ``Training`` 

```
mkdir <data_dir>/checkpoints
mkdir <data_dir>/my<ret_model_name>
cd scripts
bash train.sh
```

After training, please include your checkpoint file name in this [text file](https://github.com/xfactlab/I0T/blob/main/configs/ckpt_file_name.txt) to post-train.


## 4. ``Evaluation``

### Retrieval

Running the script below will save the tensors needed to visualize embeddings on the 3D sphere.

```
mkdir tensors # default output_folder of the below script
cd scripts
bash evaluate_retrieval.sh
```

### Classification

```
cd scripts
bash evaluate_classification.sh
```

### Human Correlation

```
mkdir Github
cd Github
git clone https://github.com/aimagelab/pacscore.git
scp evaluation/pac_score.py ~/I0T/Github/pacscore/evaluation/pac_score/ # overwrite
cd scripts
bash evaluate_correlation.sh
```
