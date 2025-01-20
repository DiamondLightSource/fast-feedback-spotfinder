#!/bin/bash

# Run this file from within the build folder i.e.
# ../baseline_indexer/test_baseline_indexer.sh 

# To run the indexer, we need:
#   a strong reflection table (in DIALS new H5 format)
#   an experiment list (standard dials format)
#   a max cell (i.e. upper limit on basis vector length)
#   a dmin (resolution limit of spots to use in fourier transform)
if test -f "candidate_vectors.json"; then
    rm candidate_vectors.json
fi

./bin/baseline_indexer \
  -r /dls/mx-scratch/jbe/test_cuda_spotfinder/cm37235-2_ins_14_24_rot/strong.refl \
  -e /dls/mx-scratch/jbe/test_cuda_spotfinder/cm37235-2_ins_14_24_rot/imported.expt \
  --max-cell 100 \
  --dmin 1.81

# Read the output from the candidate vecs, which is all we have for now.
output=$(cat candidate_vectors.json)

expected_output='{
    "00": [
        0.3828846032802875,
        16.046345646564774,
        -2.9586537526204055
    ],
    "01": [
        41.00043348644091,
        -6.374347624571426,
        -53.04086788840915
    ],
    "02": [
        25.233528614044186,
        61.114115715026855,
        -14.959117174148561
    ],
    "03": [
        15.894061997532845,
        -63.236873000860214,
        -38.00999879837036
    ],
    "04": [
        0.45249998569488525,
        13.574999570846558,
        -8.144999742507935
    ],
    "05": [
        18.778749406337738,
        87.10624724626541,
        90.95249712467194
    ],
    "06": [
        75.11499762535095,
        18.09999942779541,
        0.0
    ],
    "07": [
        8.144999742507935,
        90.04749715328217,
        25.339999198913574
    ],
    "08": [
        59.72999811172485,
        82.35499739646912,
        37.33124881982803
    ],
    "09": [
        0.0,
        1.809999942779541,
        -9.954999685287476
    ],
    "10": [
        1.809999942779541,
        26.244999170303345,
        36.19999885559082
    ]
}'

# Compare the output with the expected output
if [ "$output" == "$expected_output" ]; then
  echo "#############################################"
  echo "#                                           #"
  echo "#               SUCCESS!!!                  #"
  echo "#                                           #"
  echo "#############################################"
else
  echo "*********************************************"
  echo "*                                           *"
  echo "*               FAILURE!!!                  *"
  echo "*                                           *"
  echo "*********************************************"
fi