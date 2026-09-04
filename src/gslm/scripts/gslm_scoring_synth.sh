set -e 

echo "Activating virtual environment..."
root_dir=~/speech_ppl
cd $root_dir
source $root_dir/venv/gslm-hopper/venv/gslm-hopper/bin/activate

pretrained_model_dir=$root_dir/work/pretrained_models
gslm_dir=$root_dir/textlesslib/examples/gslm
gslm_output_dir=$root_dir/work/outputs/gslm
name="GSLM_scoring"

mkdir -p $gslm_output_dir
export CUDA_VISIBLE_DEVICES=0

echo "Running gslm_scoring..."
python $root_dir/src/gslm/tools/gslm_scoring_phone_synth.py \
	--name $name \
	--root_dir $root_dir \
	--dataset_dir $root_dir/src/stim_final/setB_audio \
    --language_model_dir $pretrained_model_dir/gslm/hubert100_lm \
	--output_dir $gslm_output_dir \
	--labels_dir $root_dir/src/scores_enhanced.json \
	--device cuda \
	--alignments $root_dir/src/mfa/phone_extraction.json

