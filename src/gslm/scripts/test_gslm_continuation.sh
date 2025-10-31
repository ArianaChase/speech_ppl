set -e 

root_dir=~/speech_ppl
cd $root_dir
source $root_dir/venv/gslm/.venv/bin/activate

pretrained_model_dir=$root_dir/work/pretrained_models
data_sample_dir=$root_dir/work/data/samples

gslm_dir=$root_dir/textlesslib/examples/gslm
gslm_output_dir=$root_dir/work/outputs/gslm
mkdir -p $gslm_output_dir
python $gslm_dir/sample.py \
	--language-model-data-dir $pretrained_model_dir/gslm/hubert100_lm \
	--input-file $data_sample_dir/61-70968-0000_orig.flac \
	--output-file $gslm_output_dir/output_new.wav \
	--prompt-duration-sec=3 \
	--temperature=0.7 \
	--vocab-size=100