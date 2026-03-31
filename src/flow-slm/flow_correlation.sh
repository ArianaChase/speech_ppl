root_dir=~/new_work/speech_ppl
cd $root_dir

name="flow_correlation"
loss=$root_dir/work/outputs/flow-slm/predictions.csv

python $root_dir/src/flow-slm/flow_correlation.py \
    --name $name \
    --loss_file $loss \
	--labels_dir $root_dir/speechocean762/resource/scores.json \
