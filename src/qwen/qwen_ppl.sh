root_dir=~/speech_ppl

echo "Running Qwen3-TTS scoring..."
output_dir=$root_dir/work/outputs/qwen3tts/test_B/set_B/235_samples
mkdir -p $output_dir
python $root_dir/src/qwen/qwen_ppl.py \
    --dataset_dir $root_dir/src/stim_final/setB_audio\
    --annotation_dir $root_dir/src/scores_enhanced.json \
    --alignments_file $root_dir/src/mfa/phone_extraction.json \
    --root_dir $root_dir \
    --output_dir $output_dir \
    --test B \
    --set setB