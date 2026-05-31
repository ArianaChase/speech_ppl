root_dir=~/new_work/speech_ppl
cd $root_dir

name="flow1bext_likelihood"
index=6
model="Flow-SLM"
category="Likelihood_Correlation"
loss=$root_dir/work/outputs/flow-slm/predictions_1bext_normal_$index.csv

python $root_dir/src/flow/flow_correlation.py \
    --name $name \
    --index $index \
    --model $model \
    --category $category \
    --loss_file $loss \
	--labels_dir $root_dir/speechocean762/resource/scores.json \
