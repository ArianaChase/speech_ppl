root_dir=~/speech_ppl
cd $root_dir
source $root_dir/venv/gslm-hopper/venv/gslm-hopper/bin/activate

set_name="setB"

echo "Calculating metrics..."
python $root_dir/src/metrics/metrics_synth_A.py \
	--root_dir $root_dir \
	--dataset_dir $root_dir/speechocean762/WAVE/ \
	--labels_dir $root_dir/src/scores_enhanced.json \
	--alignments $root_dir/src/metrics/alignments.json \
	--metadata $root_dir/src/stim_final/setB_stimuli_list.csv \
    --evaluation_file /home/ubuntu/speech_ppl/work/outputs/cosyvoice/COSYVOICE_COSYVOICE_domestic_retrieval_per_token_losses.csv \
	--evaluation_dir $root_dir/work/outputs/flow-slm/test_A_acoustic \
	--set_name $set_name