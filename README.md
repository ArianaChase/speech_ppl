
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

## Generate Speech PPL on salmon
```bash
bash ./src/gslm/scripts/generate_gslm_result.sh
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
# optional download other model sizes TWIST-{350M,7B}
```

# TASTE
## Setup Environment
```bash
git clone https://github.com/mtkresearch/TASTE-SpokenLM.git
mkdir -p venv/taste
cd venv/taste
uv venv --python 3.10
source .venv/bin/activate
cd ../.. # back to root
# NOTE: we assume that CUDA driver >= 12.x and CuDNN >= 9.x (use torch>=2.4.0) when installing onnxruntime-gpu==1.21.0
uv pip install -e TASTE-SpokenLM
uv pip install install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0
uv pip install transformers==4.51.1 datasets==3.6.0
uv pip install einx==0.3.0 HyperPyYAML==1.2.2 openai-whisper==20231117 \
    onnxruntime-gpu==1.21.0 conformer==0.3.2 lightning==2.2.4 numpy==1.26.4 \
    matplotlib==3.10.3 librosa==0.11.0 omegaconf==2.3.0 diffusers==0.33.1 peft==0.15.2
```
## (Optional: Test ONNX on CUDA)
To speed up the s3 tokenizer encoding using onnxruntime, we can use onnxruntime-gpu for cuda. 
Even if ONNX on CUDA fails, one can always fall back to using the CPU. 
```python
# prepare a onnx model
import onnxruntime
# automatic search fo dll (onnxruntime-gpu>=1.21.0 required)
onnxruntime.preload_dlls() # <- This could be important if onnx cannot find your cuDNN path
model_fpath = "your/path/to/any/onnx_model.onnx"
options = onnxruntime.SessionOptions()
onnx_model = onnxruntime.InferenceSession(model_fpath, sess_options=options, providers=["CUDAExecutionProvider"])
# output: warnings are fine, but there should be no error message. 
# automatically fall back to using CPU
# onnx_model = onnxruntime.InferenceSession(model_fpath, sess_options=options, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
```
[Reference to onnxruntime-gpu](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

## Prepare TASTE Pretrained Model
```bash
cd ./work/pretrained_models
mkdir -p taste
cd taste
git lfs install
git clone https://huggingface.co/MediaTek-Research/Llama-1B-TASTE-V0
cd ../../.. # back to root
```

## Test Model Continuation
```bash
bash ./src/taste/scripts/test_taslm_continuation.sh
```