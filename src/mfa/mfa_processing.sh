set -e 

echo "Activating virtual environment..."
root_dir=~/new_work/speech_ppl
cd $root_dir
source $root_dir/venv/gslm/.venv/bin/activate

python /home/u5504709/new_work/speech_ppl/src/mfa/mfa_processing.py