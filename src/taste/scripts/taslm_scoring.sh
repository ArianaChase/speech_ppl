set -e 

echo "Activating virtual environment..."
root_dir=~/speech_ppl
cd $root_dir
source $root_dir/venv/taste/.venv/bin/activate

device=cuda
pretrained_model_dir=$root_dir/work/pretrained_models/taste/Llama-1B-TASTE-V0
data_sample_dir=$root_dir/work/data/samples
taslm_output_dir=$root_dir/work/outputs/taslm
mkdir -p $taslm_output_dir
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
nvidia-smi

echo "Running TASTE scoring..."
python $root_dir/src/taste/tools/taslm_scoring.py \
    --pretrained_model_dir $pretrained_model_dir \
    --testing_audio_fpath $data_sample_dir/speech.wav \
    --output_dir $taslm_output_dir \
    --device $device \
    --dataset_dir $root_dir/speechocean762/WAVE \
    --labels_dir $root_dir/speechocean762/resource/scores.json \
