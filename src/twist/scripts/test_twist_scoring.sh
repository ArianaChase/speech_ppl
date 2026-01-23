set -e 

echo "Activating virtual environment..."
root_dir=~/speech_ppl
cd $root_dir
source $root_dir/venv/twist/.venv/bin/activate

twist_pretrained_model_dir=$root_dir/work/pretrained_models/twist/TWIST-1.3B
data_sample_dir=$root_dir/work/data/samples
twist_output_dir=$root_dir/work/outputs/twist

mkdir -p $twist_output_dir

echo "Running twist_scoring..."
python /Users/hermitcrab/speech_ppl/src/twist/tools/twist_scoring.py \
	--input_audio_fpath $data_sample_dir/test_tts.wav \
    --language_model_dir $twist_pretrained_model_dir \
	--device cpu \
	--test_only \
