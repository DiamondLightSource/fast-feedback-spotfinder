#!/bin/bash

# This script tests the 3d connected component algorithm on a test dataset
# It is meant to be run from the build folder: working_directory/build

# Ensure the output directory exists
mkdir -p _debug_output

# Change to the output directory
cd _debug_output

# Open file descriptor 3 for writing to output_file.txt
exec 3> output_file.txt

../bin/spotfinder /dls/i03/data/2024/mx39148-12/auto/LC_KAT6A/2024-L049_E2a_1/2024-L049_E2a_1_1_master.h5 \
  --min-spot-size 3 \
  --pipe_fd 3 \
  --dmin 4 \
  --algorithm "dispersion" \
  --threads 1 \
  --images 5

# Close file descriptor 3
exec 3>&-

# Read the output from the file
# output=$(cat output_file.txt)

# Define the expected output
# expected_output='{ }'

# Compare the output with the expected output
# if [ "$output" == "$expected_output" ]; then
#   echo "#############################################"
#   echo "#                                           #"
#   echo "#               SUCCESS!!!                  #"
#   echo "#                                           #"
#   echo "#############################################"
# else
#   echo "*********************************************"
#   echo "*                                           *"
#   echo "*               FAILURE!!!                  *"
#   echo "*                                           *"
#   echo "*********************************************"
# fi