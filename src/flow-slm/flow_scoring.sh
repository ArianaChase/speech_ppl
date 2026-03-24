root_dir=~/new_work/speech_ppl
cd $root_dir

output_dir="$root_dir/work/outputs/flow-slm"

if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

conf_path="$root_dir/src/flow-slm/conf/1b_extended.yaml"
data_dir="/home/u5504709/new_work/speech_ppl/speechocean762/WAVE"
id="$root_dir/src/flow-slm/predict_id.txt"
ckpt_path="$root_dir/work/pretrained_models/flow-slm/1b_extend.bin"
k_future_tokens=4
batch_size=1

python $root_dir/src/flow-slm/dataloader_trainer.py \
    --data_dir $data_dir \
    --conf $conf_path  \
    --predict_id_file $id \
    --ckpt_path $ckpt_path \
    --prediction_output_dir $output_dir \
    --use_k_future_tokens $k_future_tokens \
    --ignore_eos \
