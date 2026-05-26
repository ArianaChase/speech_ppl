import torch.nn.functional as F  
import torch
from moshi.models import loaders 
from huggingface_hub import hf_hub_download  
import sphn  

mimi_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", loaders.MIMI_NAME)  
mimi = loaders.get_mimi(mimi_path, device="cuda")
mimi.set_num_codebooks(8)  

audio_path = "/home/u5504709/new_work/speech_ppl/speechocean762/WAVE/SPEAKER0001/000010011.WAV"
wav, sr = sphn.read(audio_path, sample_rate=mimi.sample_rate)  

original_audio = torch.from_numpy(wav).to(device='cuda')  
original_audio = original_audio[None, 0:1] 

frame_size = mimi.frame_size  
padding = (frame_size - (original_audio.shape[-1] % frame_size)) % frame_size  
if padding > 0:  
    original_audio = torch.nn.functional.pad(original_audio, (0, padding))  


with torch.no_grad():  
    codes = mimi.encode(original_audio)  
    reconstructed_audio = mimi.decode(codes)  

codes = mimi.encode(original_audio) 

reconstruction_loss = torch.nn.functional.mse_loss(  
    reconstructed_audio,   
    original_audio  
)  

print(f"Reconstruction loss: {reconstruction_loss.item()}")
