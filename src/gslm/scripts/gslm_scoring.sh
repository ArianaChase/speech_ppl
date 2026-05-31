set -e 

echo "Activating virtual environment..."
root_dir=~/new_work/speech_ppl
cd $root_dir
source $root_dir/venv/gslm/.venv/bin/activate

pretrained_model_dir=$root_dir/work/pretrained_models
data_sample_dir=$root_dir/work/data/samples

gslm_dir=$root_dir/textlesslib/examples/gslm
gslm_output_dir=$root_dir/work/outputs/gslm

category="Likelihood_Correlation"
name="gslm_likelihood_agednot18"
index=001
model="GSLM"

mkdir -p $gslm_output_dir
export CUDA_VISIBLE_DEVICES=0

echo "Running gslm_scoring..."
python $root_dir/src/gslm/tools/gslm_scoring.py --help 
python $root_dir/src/gslm/tools/gslm_scoring_aged.py \
	--name $name \
	--dataset_dir $root_dir/speechocean762/WAVE/ \
    --language_model_dir $pretrained_model_dir/gslm/hubert100_lm \
	--output_dir $gslm_output_dir \
	--labels_dir $root_dir/src/scores_enhanced.json \
	--device cuda \
	--index $index \
	--category $category \
	--model $model

