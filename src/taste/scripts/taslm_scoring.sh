set -e 

source ./venv/taste/.venv/bin/activate

device=cpu
pretrained_model_dir=./work/pretrained_models/taste/Llama-1B-TASTE-V0
data_sample_dir=./work/data/samples
taslm_output_dir=./work/outputs/taslm
mkdir -p $taslm_output_dir
export CUDA_VISIBLE_DEVICES=0

# test loss calculation and generation on a single sample
python /Users/hermitcrab/speech_ppl/src/taste/tools/taslm_scoring.py \
    --pretrained_model_dir $pretrained_model_dir \
    --testing_audio_fpath $data_sample_dir/speech.wav \
    --output_dir $taslm_output_dir \
    --device $device \
