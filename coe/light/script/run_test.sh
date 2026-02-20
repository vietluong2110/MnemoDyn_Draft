#!/bin/bash

# Default values
START=${1:-0}				# First argument, default to 0 if not provided
END=${2:-6}				# Second argument, default to 351 if not provided
FOLDER=${3:-"/workspace/media/vhluong_nas/Result/high_dim_BCR/debug_GordonHCP"}		# Third argument, default to original path

for ((i=START; i<=END; i++))
do
	python light_test.py --base_dir "$FOLDER" --version "$i"
done