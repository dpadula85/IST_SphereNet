#!/bin/bash

for i in `ls test_xyzs/*.xyz`; do
    j=`echo $i | cut -d / -f2`
    obminimize -ff GAFF -o xyz ${i} > mm_opt/${j%.*}_GAFF.xyz 2> mm_opt/${j%.*}_GAFF.log
done
