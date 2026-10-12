root_dir=~/speech_ppl
cd $root_dir
source $root_dir/venv/gslm-hopper/venv/gslm-hopper/bin/activate

set_name="setB"

echo "Calculating metrics..."
python $root_dir/src/metrics/metrics_synth_B.py \
	--root_dir $root_dir \
	--dataset_dir $root_dir/speechocean762/WAVE/ \
	--labels_dir $root_dir/src/scores_enhanced.json \
	--alignments $root_dir/src/metrics/alignments.json \
	--metadata $root_dir/src/stim_final/setB_stimuli_list.csv \
    --evaluation_file /home/ubuntu/speech_ppl/work/outputs/flow-slm/1b_ext/Flow-SLM_Flow-SLM-1Bext-semantic_per_token_losses.csv \
	--evaluation_dir $root_dir/work/outputs/cosyvoice2/test_B/set_B/235_samples \
	--set_name $set_name \
