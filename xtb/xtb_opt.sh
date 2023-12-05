#!/bin/bash

ulimit -s unlimited
export OMP_NUM_THREADS=1

for i in `ls test_xyzs/*.xyz`; do
    xtb -o -i ${i} > ${i%.*}_xtb.log 2> /dev/null
    j=`echo $i | cut -d / -f2`
    mv xtbopt.xyz xtb_opt/${j%.*}_xtb.xyz
    rm charges wbo xtbopt.log xtbrestart xtbtopo.mol
done
