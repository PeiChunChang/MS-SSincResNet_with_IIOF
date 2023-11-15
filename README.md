# MS-SSincResNet with IIOF
# IIOF: Intra- and Inter-feature Orthogonal Fusion of Local and Global Features for Music Emotion Recognition
This paper is under review.

## Prerequisites
- Ubuntu 22.04
- Python 3.7
- Pytorch 1.10.0
- torchvision 0.11.1
- scipy 1.7.1

## Usage:
```
# You can take main_DEAM.py and main_PMEmo.py as references for training on the DEAM and PEMmo datasets, respectively.
# Please check the dataset path
python main_DEAM.py --cv_num 10 --dataset_path $deam_path$ --max_epoch 200 --batch_size 4 --init_lr 0.0001
python main_PMEmo.py --cv_num 10 --dataset_path $deam_path$ --max_epoch 200 --batch_size 4 --init_lr 0.0001
```
