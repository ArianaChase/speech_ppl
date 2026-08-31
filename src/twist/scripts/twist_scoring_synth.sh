set -e 

echo "Activating virtual environment..."
root_dir=/home/ubuntu/speech_ppl
cd $root_dir
source $root_dir/venv/twist/.venv/bin/activate

twist_pretrained_model_dir=$root_dir/work/pretrained_models/twist/TWIST-1.3B
twist_output_dir=$root_dir/work/outputs/twist
name="TWIST_scoring"

mkdir -p $twist_output_dir
export CUDA_VISIBLE_DEVICES=0
echo $SETUPTOOLS_USE_DISTUTILS
unset SETUPTOOLS_USE_DISTUTILS
echo $SETUPTOOLS_USE_DISTUTILS


echo "Running twist_scoring..."
python $root_dir/src/twist/tools/twist_scoring_local_synth.py \
	--name $name \
	--root_dir $root_dir \
    --language_model_dir $twist_pretrained_model_dir \
	--dataset_dir $root_dir/src/stim_final/setB_audio \
	--output_dir $root_dir/work/outputs/twist/ \
	--labels_dir $root_dir/src/scores_enhanced.json \
	--device cuda \
	--alignments $root_dir/src/mfa/phone_extraction.json \

