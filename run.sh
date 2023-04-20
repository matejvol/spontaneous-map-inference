#!/bin/bash

set -ueo pipefail

monkey=$1
arraynr=$2
method=$3

python correlation_maps.py monkey_${monkey}${arraynr}_${method}_spont ${monkey} ${arraynr}

python spontaneous_map.py monkey_${monkey}${arraynr}_${method}_spont

# statistical test
python test_spontaneous_map.py ${monkey}$arraynr $method spontaneous

