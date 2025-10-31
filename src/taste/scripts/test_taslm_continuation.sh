set -e 

root_dir=/mnt/storages/nfs/projects/speech_ppl
cd $root_dir
source $root_dir/venv/taste/.venv/bin/activate

pretrained_model_dir=$root_dir/work/pretrained_models
data_sample_dir=$root_dir/work/data/samples

taslm_output_dir=$root_dir/work/outputs/taslm/test_continuation
mkdir -p $taslm_output_dir
python ./src/taste/tools/test_taslm_continuation.py \
	--pretrained_model $pretrained_model_dir/taste/Llama-1B-TASTE-V0 \
	--input_dir $data_sample_dir \
	--output_dir $taslm_output_dir \
    --copy_src_to_output_dir
