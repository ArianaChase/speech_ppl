set -e 

echo "Activating virtual environment..."
root_dir=~/speech_ppl
cd $root_dir
source $root_dir/venv/twist/.venv/bin/activate

twist_pretrained_model_dir=$root_dir/work/pretrained_models/twist/TWIST-1.3B
data_sample_dir=$root_dir/work/data/samples
twist_output_dir=$root_dir/work/outputs/twist

mkdir -p $twist_output_dir
<<<<<<< HEAD

echo "Running twist_scoring..."
python /Users/hermitcrab/speech_ppl/src/twist/tools/twist_scoring.py \
    --language_model_dir $twist_pretrained_model_dir \
	--dataset_dir $root_dir/speechocean762/WAVE \
	--output_dir $root_dir/work/outputs/twist/ \
	--device cpu \
=======
export CUDA_VISIBLE_DEVICES=0
echo $SETUPTOOLS_USE_DISTUTILS
unset SETUPTOOLS_USE_DISTUTILS
echo $SETUPTOOLS_USE_DISTUTILS


echo "Running twist_scoring..."
python $root_dir/src/twist/tools/twist_scoring.py \
    --language_model_dir $twist_pretrained_model_dir \
	--dataset_dir $root_dir/speechocean762/WAVE \
	--output_dir $root_dir/work/outputs/twist/ \
	--labels_dir $root_dir/speechocean762/resource/scores.json \
	--device cuda \
>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))
