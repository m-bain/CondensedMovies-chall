#!/bin/bash
# This script requires that the challenge version of the Condensed Movies Challenge dataset is already downloaded. 
# For details on how to download the dataset, see the challenge website https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/challenge.html

# path to the where the dataset was downloaded to
DATA_DOWNLOAD_DIR="/work/abrown/Condensed_Movies_Challenge/Challenge_parts"
# default data path if not set. This is where the dataset will be unpacked to
DEFAULT_DIR="./data"
DATA_DIR=${DATA_DIR:-$DEFAULT_DIR}

CMD_DIR=$DATA_DIR"/CondensedMovies"
mkdir -p $CMD_DIR

# unpack the dataset
echo "unpacking data in "$CMD_DIR

for FILE_INDEX in {'a','b','c','d','e','f','g','h','i','j','k','l'}
    do
        echo "CondensedMovies_chall.tar.gz.parta"$FILE_INDEX
        tar -xzf $DATA_DOWNLOAD_DIR"/CondensedMovies_chall.tar.gz.parta"$FILE_INDEX -C $CMD_DIR
    done
