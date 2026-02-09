set -e 

source ./venv/taste/.venv/bin/activate

device=cuda
pretrained_model_dir=./work/pretrained_models/taste/Llama-1B-TASTE-V0
data_sample_dir=./work/data/samples
taslm_output_dir=./work/outputs/taslm
mkdir -p $taslm_output_dir
export CUDA_VISIBLE_DEVICES=1

# test loss calculation and generation on a single sample
python src/taste/tools/taslm_speech_ppl_wrapper.py \
    --pretrained_model_dir $pretrained_model_dir \
    --testing_audio_fpath $data_sample_dir/61-70968-0000_orig.flac \
    --output_dir $taslm_output_dir \
    --device $device \
    --pre_extract_features

python src/taste/tools/taslm_speech_ppl_wrapper.py \
    --pretrained_model_dir $pretrained_model_dir \
    --output_dir $taslm_output_dir \
    --device $device \
    --extract_speech_ppl_results

python src/taste/tools/taslm_speech_ppl_wrapper.py \
    --pretrained_model_dir $pretrained_model_dir \
    --output_dir $taslm_output_dir \
    --device $device \
    --run_continuation

python src/taste/tools/taslm_speech_ppl_wrapper.py \
    --pretrained_model_dir $pretrained_model_dir \
    --output_dir $taslm_output_dir \
    --device $device \
    --extract_additional_ppl_results

python src/taste/tools/taslm_speech_ppl_wrapper.py \
    --pretrained_model_dir $pretrained_model_dir \
    --output_dir $taslm_output_dir \
    --device $device \
    --calculate_sanity_only

