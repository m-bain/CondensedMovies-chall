# Condensed Movies Challenge 

The official code repository for the 2021 Condensed Movies Challenge, held at the 4th Workshop on Closing the Loop Between Vision and Language, held in conjunction with ICCV 2021. This repository contains the code and details for the data download, baseline model, and evaluation.

## Dataset Download

To participate in the challenge, you must first download the new **Challenge version of the Condensed Movies dataset.** Use the following instructions to download the challenge version of the Condensed Movies dataset:

1) Clone this repository.

2)  **Download the dataset**. First, you can optionally choose where to download the features to. To do this, in [scripts/download_dataset.sh](https://github.com/m-bain/CondensedMovies-chall/blob/main/scripts/download_dataset.sh "download_dataset.sh") edit the variable "Output_Directory" to give the path to where you would like the visual features to be downloaded to. By default the features will be downloaded to the directory "dataset". The dataset totals xGB.  Second, download the dataset. Run "cd scripts", and "./download_dataset.sh". 

**todo: need to get the text features into this**

## Dataset Overview

Here, we provide an overview and detail for the downloaded dataset. For more information on the following features in the dataset, including details architectures and datsets, see [here](https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/features.html "here"). Below is an overview of the dataset tree structure with details:

```

├── features
│   ├── pred_audio
│   │      └── vggish   (audio features)
│   ├── pred_i3d_25fps_256px_stride8_offset0_inner_stride1
│   │      └── i3d  (action features)
│   ├── pred_imagenet_25fps_256px_stride25_offset0
│   │      └── resnext101_32x48d  (Instagram Hashtags, fine-tuned on ImageNet features)
│   │      └── senet154  (ImageNet features)
│   ├── pred_r2p1d_30fps_256px_stride16_offset0_inner_stride1
│   │      └── r2p1d-ig65m  (Instagram 65m, fine-tuned on Kinetics)
│   └── pred_scene_25fps_256px_stride25_offset0
│   │      └── densenet161  (scene features)
├── metadata
│   ├── subs_test.json (raw text subtitle files for the test set)
│   ├── subs_train_val.json (raw text subtitle files for the train/val set)
│   ├── test_challf0.csv (raw text descriptions for the test set)
│   └── train_val_challf0.csv (raw text descriptions for the train/val set)

```

Below is an overview for the dataset tree structure within a specific feature directory. The features from the train/val videos are further subaranged by year (i.e. 2011 -> 2019 directories). The features from the test videos are found in the 'test' directory.

```

└── vggish 
    ├── 2011 
        ├── sBjNpZ5t9S4.npy
        ├── SBLMTDMdTIU.npy
        ├── SbTWLdT_tgk.npy
        ├── ...
    ├── 2012 
        ├── ...
    ├── 2013
        ├── ...
    ├── 2014
        ├── ...
    ├── 2015
        ├── ...
    ├── 2016
        ├── ...
    ├── 2017
        ├── ...
    ├── 2018
        ├── ...
    ├── 2019
        ├── ...
    └── test (the features for the test videos)
        ├── 1061.pkl.npy
        ├── 1062.pkl.npy
        ├── 1063.pkl.npy
        ├── ...
```


**Train/Val/Test Splits:** the splits are contained in and read from the text description csv files (e.g. metadata/train_val_challf0.csv & metadata/test_challf0.csv)

