#!/bin/bash

set -ueo pipefail


monkey=A
method='MUAe'
for ix in {0..25}  # number of parameter combinations
do
for arraynr in 10 11  # arrays
do
python correlation_maps.py scan_monkey_${monkey}${arraynr}_${method}_spont ${monkey} ${arraynr} $ix

for pca in 0 1 2
do
python spontaneous_map.py scan_monkey_${monkey}${arraynr}_${method}_spont $ix $pca

# statistical test
python test_spontaneous_map.py ${monkey}$arraynr $method spontaneous $ix $pca
done
done
done

