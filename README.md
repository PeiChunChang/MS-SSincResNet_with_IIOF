# MS-SSincResNet with IIOF
# IIOF: Intra- and Inter-feature Orthogonal Fusion of Local and Global Features for Music Emotion Recognition


## Prerequisites
- Ubuntu 22.04
- Python 3.7
- Pytorch 1.10.0
- torchvision 0.11.1
- scipy 1.7.1

## Dataset structure
```
———— $dataset_name$/
    ｜—— anno_mat/
        ｜—— 1.mat
        ｜—— 2.mat
             .
             .
             .
    ｜—— npy/
        ｜—— 1.npy
        ｜—— 2.npy
             .
             .
             .
    ｜—— CV_10_with_val/
        ｜—— fold_0_train.npy
        ｜—— fold_0_val.npy
        ｜—— fold_0_test.npy
        ｜—— fold_1_train.npy
        ｜—— fold_1_val.npy
        ｜—— fold_1_test.npy
             .
             .
             .
 ```     
## Usage:
```
# You can take main_DEAM.py and main_PMEmo.py as references for training on the DEAM and PEMmo datasets, respectively.
# Please check the dataset path
python main_DEAM.py --cv_num 10 --dataset_path $deam_path$ --max_epoch 200 --batch_size 4 --init_lr 0.0001
python main_PMEmo.py --cv_num 10 --dataset_path $deam_path$ --max_epoch 200 --batch_size 4 --init_lr 0.0001
```
  
