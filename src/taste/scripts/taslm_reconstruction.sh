set -e 

echo "Activating virtual environment..."
root_dir=/home/u5504709/new_work/speech_ppl
cd $root_dir
source $root_dir/venv/taste/.venv/bin/activate

echo "Running TASTE reconstruction..."
python $root_dir/src/taste/tools/taslm_reconstruction_dataloader.py \