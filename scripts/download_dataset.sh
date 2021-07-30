#!/bin/bash
# A script to download and unpack the data for the Condensed Movies Challenge

# default data path if not set
DEFAULT_DIR="./data"
DATA_DIR=${DATA_DIR:-$DEFAULT_DIR}

CMD_DIR=$DATA_DIR"/CondensedMovies"
mkdir -p $CMD_DIR
# 1) visual features

for FILE_INDEX in {'a','b','c','d','e','f','g','h','i','j','k','l'}
    do
        wget -c http://pedro.eng.ox.ac.uk/Condensed_Movies_Challenge/features.tar.gz.parta$FILE_INDEX -O - | tar -xz -C $CMD_DIR
    done

# 2) raw text

wget -c https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/assets_challenge/metadata.tar.gz -O - | tar -xz -C $CMD_DIR

# 3) subtitle features

# do not edit this path
feats_dir=CMD_DIR"/features"

wget -c https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/assets_challenge/pred_subs.tar.gz -O - | tar -xz -C $feats_dir

