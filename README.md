# Condensed Movies Challenge 

The official code repository for the 2021 Condensed Movies Challenge, held at the 4th Workshop on Closing the Loop Between Vision and Language, held in conjunction with ICCV 2021. This repository contains the code and details for the data download, baseline model, and evaluation.

## Dataset Download

To participate in the challenge, you must first download the new **Challenge version of the Condensed Movies dataset.** Use the following instructions to download the challenge version of the Condensed Movies dataset:

1) Clone this repository.

2)  **Download the visual features**. First, you can optionally choose where to download the features to. To do this, in [scripts/download_visual_features.sh](https://github.com/m-bain/CondensedMovies-chall/blob/main/scripts/download_visual_features.sh "download_visual_features.sh") edit the variable "Output_Directory" to give the path to where you would like the visual features to be downloaded to. By default the features will be downloaded to the directory "dataset". The visual features total 124GB.  Second, download the visual features. Run "cd scripts", and "./download_visual_features.sh".

3) **Download the textual features**.  TODO
