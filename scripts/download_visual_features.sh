#!/bin/bash
# A script to download and unpack the visual data for the Condensed Movies Challenge

Output_Directory="../dataset"

for FILE_INDEX in {'a','b','c','d','e','f','g','h','i','j','k','l'}
    do
        wget -c http://pedro.eng.ox.ac.uk/Condensed_Movies_Challenge/features.tar.gz.parta$FILE_INDEX -O - | tar -xz -C $Output_Directory
    done
