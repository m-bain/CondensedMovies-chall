#!/bin/bash
# A script to download and unpack the data for the Condensed Movies Challenge

Output_Directory="../dataset"

# 1) visual features

for FILE_INDEX in {'a','b','c','d','e','f','g','h','i','j','k','l'}
    do
        wget -c http://pedro.eng.ox.ac.uk/Condensed_Movies_Challenge/features.tar.gz.parta$FILE_INDEX -O - | tar -xz -C $Output_Directory
    done

# 2) raw text

wget -c https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/assets_challenge/metadata.tar.gz -O - | tar -xz -C $Output_Directory

