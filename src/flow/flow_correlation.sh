root_dir=~/new_work/speech_ppl
cd $root_dir

name="flow1bext_likelihood"
index=8
model="Flow-SLM"
category="Likelihood_Correlation"
loss=$root_dir/work/data/speechocean/1b_extend/2/train/flow_loss.txt

python $root_dir/src/flow/flow_correlation_ext.py \
    --name $name \
    --index $index \
    --model $model \
    --category $category \
    --loss_file $loss \
	--labels_dir $root_dir/src/scores_enhanced.json \
