set -e 

echo "Activating virtual environment..."
root_dir=~/speech_ppl
cd $root_dir
source $root_dir/venv/gslm/.venv/bin/activate

pretrained_model_dir=$root_dir/work/pretrained_models
data_sample_dir=$root_dir/work/data/samples

gslm_dir=$root_dir/textlesslib/examples/gslm
gslm_output_dir=$root_dir/work/outputs/gslm

mkdir -p $gslm_output_dir

echo "Running gslm_scoring..."
python /Users/hermitcrab/speech_ppl/src/gslm/tools/gslm_scoring.py \
	--dataset_dir $root_dir/speechocean762/WAVE/ \
    --language_model_dir $pretrained_model_dir/gslm/hubert100_lm \
	--output_dir $gslm_output_dir \
	--device cpu \
	--test_only \
