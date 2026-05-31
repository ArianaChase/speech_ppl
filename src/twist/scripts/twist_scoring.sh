set -e 

echo "Activating virtual environment..."
root_dir=/home/u5504709/new_work/speech_ppl
cd $root_dir
source $root_dir/venv/twist/.venv/bin/activate

twist_pretrained_model_dir=$root_dir/work/pretrained_models/flow-slm/TWIST-1.3B
data_sample_dir=$root_dir/work/data/samples
twist_output_dir=$root_dir/work/outputs/twist
category="Likelihood_Correlation"
name="twist1B_likelihood_aged18"
index=001
model="TWIST"

mkdir -p $twist_output_dir
export CUDA_VISIBLE_DEVICES=0
echo $SETUPTOOLS_USE_DISTUTILS
unset SETUPTOOLS_USE_DISTUTILS
echo $SETUPTOOLS_USE_DISTUTILS


echo "Running twist_scoring..."
python $root_dir/src/twist/tools/twist_scoring_aged.py \
	--name $name \
    --language_model_dir $twist_pretrained_model_dir \
	--dataset_dir $root_dir/speechocean762/WAVE \
	--output_dir $root_dir/work/outputs/twist/ \
	--labels_dir $root_dir/src/scores_enhanced.json \
	--device cuda \
	--category $category \
	--name $name \
	--index $index \
	--model $model \
