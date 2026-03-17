set -e 

echo "Activating virtual environment..."
root_dir=~/new_work/speech_ppl
cd $root_dir

corpus_directory=$root_dir/src/mfa/WAVE
dictionary_path=$root_dir/speechocean762/resource/lexicon.txt
acoustic_model_name=english_us_arpa
output_path=$root_dir/work/outputs

# mfa validate $corpus_directory $dictionary_path $acoustic_model_name

mfa align $corpus_directory $dictionary_path $acoustic_model_name $output_path