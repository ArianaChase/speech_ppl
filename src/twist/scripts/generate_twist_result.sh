set -e 

root_dir=~/speech_ppl
cd $root_dir
source $root_dir/textlesslib/.venv/bin/activate
device=cuda:1

twist_pretrained_model_dir=$root_dir/work/pretrained_models/twist/TWIST-1.3B
data_sample_dir=$root_dir/work/data/samples
twist_output_dir=$root_dir/work/outputs/twist
mkdir -p $twist_output_dir
python src/twist/tools/twist_speech_ppl_wrapper.py \
    --twist_model_pretrained_path $twist_pretrained_model_dir \
    --input_audio_fpath $data_sample_dir/61-70968-0000_orig.flac \
    --output_dir $twist_output_dir \
    --prompt_duration_sec 3 \
    --device $device
