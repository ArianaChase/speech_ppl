import torchaudio  
import librosa  
import numpy as np  
from taste_speech import TasteForCausalLM, TasteProcessor  
from torch.nn.utils.rnn import pad_sequence  
import torch  
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")
    
device = 0
model_id = 'MediaTek-Research/Llama-1B-TASTE-V0'  
sampling_rate = 16000 

model = TasteForCausalLM.from_pretrained(  
    model_id,   
    torch_dtype=torch.bfloat16,  # or torch.bfloat16
    low_cpu_mem_usage=False  
).to(device).eval()  

processor = TasteProcessor.from_pretrained(model_id) # preparation for the audio
generator = processor.get_generator(device=device) # decoder - tokens to audio

audio_path = '/home/u5504709/new_work/speech_ppl/speechocean762/WAVE/SPEAKER0001/000010011.WAV'  

# 1. Prepare inputs (processor handles Whisper features, ASR tokens, speaker embed)  
inputs = processor(audio_path, sampling_rate, ref_audio_list=[audio_path])  
cols = ['speaker_embeds', 'audio_features', 'audio_feature_lengths',  
        'asr_token_ids', 'asr_token_lengths', 'asr_word_ids',  
        'llm_token_ids', 'llm_token_lengths', 'llm_word_ids']  

# Get model's precision and ensure all tensors match  
model_dtype = next(model.parameters()).dtype  
batch = {}  
for k in cols:  
    tensor = torch.tensor(inputs[k][0]).unsqueeze(0).to(device)  
    if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:  
        # Only convert floating point tensors  
        batch[k] = tensor.to(model_dtype)  
    else:  
        # Keep integer tensors as they are  
        batch[k] = tensor 

# Force all model parameters to bfloat16  
for param in model.parameters():  
    param.data = param.data.to(torch.bfloat16)  
  
# Also ensure all buffers are bfloat16  
for buffer in model.buffers():  
    buffer.data = buffer.data.to(torch.bfloat16)
  
output = model.inference_reconstruction(**batch)

# 3. Synthesize waveform from speech tokens (Flow + HiFTGenerator)  
tts_speech, tts_sr = generator.inference(  
    speech_token_ids=output['speech_token_ids'],  
    speech_token_lengths=output['speech_token_lengths'],  
    flow_embedding=batch['speaker_embeds']  
)  

tts_speech = tts_speech.to(torch.float32)

torchaudio.save('/home/u5504709/new_work/speech_ppl/work/outputs/reconstructed.wav', tts_speech, tts_sr)

# Load original and resample to match reconstructed  
orig_wav, orig_sr = torchaudio.load(audio_path)  
# print("orig_wav:", orig_wav.shape)
# print("orig_sr:", orig_sr)
# print("tts_wav:", tts_speech.shape)
# print("tts_sr:", tts_sr)
orig_wav_resampled = torchaudio.transforms.Resample(orig_sr, tts_sr)(orig_wav)  
  
# Compute mel spectrograms using librosa (consistent parameters)  
def get_mel(wav_tensor, sr, n_fft=1024, hop=256, n_mels=80):  
    wav_np = wav_tensor.squeeze().numpy()  
    mel = librosa.feature.melspectrogram(  
        y=wav_np, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels  
    )  
    return librosa.power_to_db(mel, ref=np.max)  
  
orig_mel = get_mel(orig_wav_resampled, tts_sr)  
recon_mel = get_mel(tts_speech.cpu(), tts_sr)  

min_len = min(orig_mel.shape[1], recon_mel.shape[1])

orig_mel = orig_mel[:, :min_len]
recon_mel = recon_mel[:, :min_len]

#score = np.mean(np.abs(orig_mel - recon_mel))

score = mean_squared_error(orig_mel, recon_mel)
print(score)