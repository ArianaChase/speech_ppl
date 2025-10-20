
# GSLM
## Setup Environment
```bash
# cd to your working directory
# clone speech_ppl
git clone https://github.com/andybi7676/speech_ppl.git
# update submodules recursively
cd speech_ppl
git submodule update --init --recursive
# create the venv under textlesslib for gslm
mkdir -p venv/gslm
cd venv/gslm
uv python install 3.9
uv venv --python 3.9
source .venv/bin/activate
# download older version of torch first
uv pip install torch==1.13.1 torchaudio==0.13.1 datasets==3.6.0
cd ../.. # go back to the root dir
# install fairseq mannually
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout 3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99 # use the specific hash
# install fairseq (should have venv (gslm) activated)
uv pip install -e fairseq
# install textlesslib (should have venv (gslm) activated)
uv pip install -e textlesslib
```

## Prepare GSLM Pretrained Model
```bash
mkdir -p ./work/pretrained_models/gslm
wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/lm_km100/hubert100_lm.tgz -O ./work/pretrained_models/gslm/hubert100_lm.tgz
cd ./work/pretrained_models/gslm
tar -xzvf hubert100_lm.tgz
cd ../../..
```

## Prepare Your Own Audio Sample for Testing

## Test Model Continuation
```bash
# under speech_ppl
# NOTE: you may need to modify the $root_dir in the below script
bash ./src/gslm/scripts/test_gslm_continuation.sh
```

## Speech PPL (greedy) on salmon
```bash
bash ./src/gslm/scripts/generate_greedy_result.sh
```


# TWIST

## Setup Environment
```bash
# cd to your working directory
# clone speech_ppl
git clone https://github.com/andybi7676/speech_ppl.git
# update submodules recursively
cd speech_ppl
git submodule update --init --recursive
# create the venv under textlesslib for twist
mkdir -p venv/twist
cd venv/twist
uv python install 3.9
uv venv --python 3.9
source .venv/bin/activate
# download older version of torch first
uv pip install torch==2.6.0 torchaudio==2.6.0 transformers==4.53.0 datasets==3.6.0
cd ../.. # go back to the root dir
# install fairseq mannually
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout 3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99 # use the specific hash
# install fairseq (should have venv (twist) activated)
uv pip install -e fairseq
# install textlesslib (should have venv (twist) activated)
uv pip install -e textlesslib
```

## Prepare TWIST Pretrained Model
```bash
mkdir -p ./work/pretrained_models/twist
cd ./work/pretrained_models/twist
wget https://dl.fbaipublicfiles.com/textless_nlp/twist/lms/TWIST-1.3B.zip
mkdir TWIST-1.3B
unzip TWIST-1.3B.zip -d TWIST-1.3B
cd ../../..
```