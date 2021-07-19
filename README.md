# Condensed Movies Challenge 

The official code repository for the 2021 Condensed Movies Challenge, held at the 4th Workshop on Closing the Loop Between Vision and Language, held in conjunction with ICCV 2021. This repository contains the code and details for the data download, baseline model, and evaluation.

## Dataset Download

To participate in the challenge, you must first download the new **Challenge version of the Condensed Movies dataset.** Use the following instructions to download the challenge version of the Condensed Movies dataset:

1) Clone this repository.

2)  **Download the dataset**. First, you can optionally choose where to download the features to. To do this, in [scripts/download_dataset.sh](https://github.com/m-bain/CondensedMovies-chall/blob/main/scripts/download_dataset.sh "download_dataset.sh") edit the variable "Output_Directory" to give the path to where you would like the visual features to be downloaded to. By default the features will be downloaded to the directory "dataset". The dataset totals xGB.  Second, download the dataset. Run "cd scripts", and "./download_dataset.sh". 

**todo: need to get the text features into this**

## Dataset Overview

Here, we provide an overview and detail for the downloaded dataset. For more details on the features used in the challenge, see [here](https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/features.html "here").

```

├── features
│   ├── pred_audio
│   │      └── vggish   
│   ├── pred_i3d_25fps_256px_stride8_offset0_inner_stride1
│   │      └── i3d  
│   ├── pred_imagenet_25fps_256px_stride25_offset0
│   │      └── resnext101_32x48d  
│   │      └── senet154  
│   ├── pred_r2p1d_30fps_256px_stride16_offset0_inner_stride1
│   │      └── r2p1d-ig65m  
│   └── pred_scene_25fps_256px_stride25_offset0
│   │      └── densenet161  
├── metadata
│   ├── subs_test.json
│   ├── subs_train_val.json
│   ├── test_challf0.csv
│   └── train_val_challf0.csv

```
**Train/Val/Test Splits:** the splits are contained in and read from the text description csv files (e.g. metadata/train_val_challf0.csv & metadata/test_challf0.csv)

**todo**
