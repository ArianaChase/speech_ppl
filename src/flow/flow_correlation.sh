#!/bin/bash

root_dir=~/new_work/speech_ppl
cd $root_dir

echo "Starting massive correlation evaluation across all models, batches, and subsets..."

# Just run the python script once. It handles the whole file tree.
python src/flow/flow_correlation_ext.py

echo "Evaluation Complete."