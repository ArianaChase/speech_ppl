#!/bin/bash

# Define directories
ROOT_DIR="$HOME/new_work/speech_ppl"
cd "$ROOT_DIR"

# Define Script Arguments
NAME="taste_likelihood"
INDEX=1
MODEL="TASLM"
CATEGORY="Likelihood_Correlation"

# Define Input Files (Updated to point to the new CSV)
CSV_FILE="$ROOT_DIR/work/data/taste_likelihood_001.csv"
JSON_FILE="$ROOT_DIR/src/scores_enhanced.json"

echo "Starting TASTE Likelihood Correlation Evaluation and Google Sheet Append..."

python "$ROOT_DIR/src/taste/tools/taslm_likelihood_correlation.py" \
    --name "$NAME" \
    --index $INDEX \
    --model "$MODEL" \
    --category "$CATEGORY" \
    --csv_file "$CSV_FILE" \
    --json_file "$JSON_FILE"

echo "Evaluation Complete."