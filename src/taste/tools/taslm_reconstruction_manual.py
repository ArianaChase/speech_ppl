import torchaudio  
import librosa  
import numpy as np  
from taste_speech import TasteForCausalLM, TasteProcessor  
from torch.nn.utils.rnn import pad_sequence  
import torch  
from sklearn.metrics import mean_squared_error
import glob 
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
import csv
import json
import pandas as pd
import time
import datetime
import scipy.stats
from datasets import Dataset  
from torch.utils.data import DataLoader  

start_time = time.time()

def parse_accuracy_scores(filename):
    accuracy_scores = {}
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            accuracy_scores[os.path.basename(audio_file)] = value["accuracy"]

    return accuracy_scores

def create_csv_file(output_dir, name):
    filename = '%s/%s' % (output_dir, name)

    print("Creating csv with file name: ", filename, " ...")

    with open(filename, mode="w") as csvfile:
        fieldnames = ["Speaker", "Audio filename", "Raw MSE", "Human Annotation", "Correlation Score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filename

def pad_seq_collate_fn(batch, device=None):
    padded = {}
    for key in batch[0].keys():
        packed_list = [
            x[key][0].clone().detach() if isinstance(x[key][0], torch.Tensor) else torch.tensor(x[key][0]) 
            for x in batch
        ]
        if 'length' in key:
            padded_tensor = torch.tensor(packed_list)
        else:
            padded_tensor = pad_sequence(packed_list, batch_first=True, padding_value=0)

        padded[key] = padded_tensor.to(device) if device is not None else padded_tensor
    return padded

def getWAVfiles(input_dataset, amt):

    audio_paths = []

    speakerWAV = tqdm(sorted(os.listdir(input_dataset)))

    counter = 0;

    for dir in speakerWAV:
        if (counter >= amt):
            break
        speaker = dir[7:None]
        speakerWAV.set_description(f"Appending speaker: {speaker}")
        speakerWAV_path = os.path.join(input_dataset, dir)
        for file in os.listdir(speakerWAV_path):
            audio_paths.append({
                "path" : os.path.join(input_dataset, dir, file),
                "speaker" : speaker,
                "filename" : file[0:10]
            })
        counter += 1

    return audio_paths

# Compute mel spectrograms using librosa (consistent parameters)  
def get_mel(wav_tensor, sr, n_fft=1024, hop=256, n_mels=80):  
    wav_np = wav_tensor.squeeze().numpy()  
    mel = librosa.feature.melspectrogram(  
        y=wav_np, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels  
    )  
    return librosa.power_to_db(mel, ref=np.max)  

def get_MSE(original, recon):
    orig_wav, orig_sr = torchaudio.load(original["path"])  
    orig_wav_resampled = torchaudio.transforms.Resample(orig_sr, recon["sr"])(orig_wav)  

    orig_mel = get_mel(orig_wav_resampled, recon["sr"])  
    recon_mel = get_mel(recon["tensor"].cpu(), recon["sr"])  

    min_len = min(orig_mel.shape[1], recon_mel.shape[1])

    orig_mel = orig_mel[:, :min_len]
    recon_mel = recon_mel[:, :min_len]

    score = mean_squared_error(orig_mel, recon_mel)
    return score

# ======== 

input_dataset = "/home/u5504709/new_work/speech_ppl/speechocean762/WAVE/"
score_labels = "/home/u5504709/new_work/speech_ppl/speechocean762/resource/scores.json"
audio_info = getWAVfiles(input_dataset, 5)

print("Saved ", len(audio_info), " audio files from input dataset.")
print(audio_info)

output_csv = create_csv_file("/home/u5504709/new_work/speech_ppl/work/outputs", "taslm_reconstruction_001")

accuracy_scores = parse_accuracy_scores(score_labels)
accuracy_scores = dict(sorted(accuracy_scores.items()))
human_annotations_values = []
for key, value in accuracy_scores.items():
    human_annotations_values.append(value)

# 1. Loading model
print("Loading model...")
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

# 2. Inferencing batches 
print("Inferencing...")

batch_size = 32
cols = ['speaker_embeds', 'audio_features', 'audio_feature_lengths',  
        'asr_token_ids', 'asr_token_lengths', 'asr_word_ids',  
        'llm_token_ids', 'llm_token_lengths', 'llm_word_ids']  


for i in tqdm(range(0, len(audio_info), batch_size), desc="Going through batches", leave=True):  
    print("Batch processing and inferencing, iteration: ", i + 1)

    batch_files = audio_info[i:i+batch_size] # slicing list
    
    batch_data = []  
    for file_info in tqdm(batch_files, desc="Processing batches", leave=True):  
        # Process audio file through TASTE processor  
        inputs = processor(  
            file_info["path"],
            sampling_rate,          # Audio sample rate (16000 Hz)  
            ref_audio_list=[file_info["path"]]  # Reference audio for speaker embedding  
        )  
        inputs.update(file_info)   # Add speaker metadata to processed data  
        batch_data.append(inputs)  # Add to batch data list  
      
    # Collate individual files into batch tensors using TASTE's batching function  
    batch = pad_seq_collate_fn(  
        [{k: item[k][0] for k in cols} for item in batch_data],  # Extract required columns  
        device=device  # Move tensors to specified device (GPU/CPU)  
    )  
      
    # Run inference without gradient computation (saves memory)  
    with torch.no_grad():  
        # Reconstruct speech tokens from audio features  
        output = model.inference_reconstruction(**batch)  
          
        # Generate actual audio waveform from speech tokens  
        tts_speech, tts_sr = generator.inference(  
            speech_token_ids=output['speech_token_ids'],      # Generated speech tokens  
            speech_token_lengths=output['speech_token_lengths'], # Token sequence lengths  
            flow_embedding=batch['speaker_embeds']           # Speaker characteristics  
        ) 
        tts_speech = tts_speech.to(torch.float32)

        for j in tqdm(range(0, len(tts_speech)), desc="Calculating Scores: "):
            original_info = file_info
            recon_info = {
                "tensor" : tts_speech[i],
                "sr" : tts_sr
            }

            print(f"Getting scores for original speaker: { original_info['speaker'] }")
            score = get_MSE(original_info, recon_info)

            with open(output_csv, mode="a", newline="") as csvfile:
                fieldnames = ["Speaker", "Audio filename", "Raw MSE", "Human Annotation"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"Speaker": original_info["speaker"], "Audio filename": original_info["filename"], "Raw MSE": score, "Human Annotation": accuracy_scores[file_info["filename"]]})

output_csv_df = pd.read_csv(output_csv)
x = output_csv_df["Raw MSE"].values
y = human_annotations_values

print("Correlation x len: ", len(x))
print("Correlation y len: ", len(y))
    
print(f"Correlation value is: {scipy.stats.pearsonr(x, y)}")

now = datetime.now() 
finish_time = now.strftime("%m-%d-%Y %H:%M") 
print(f"Date and time at completion: {finish_time}") 
print(f"Program 'taste_reconstruction' finished executing in {time.time() - start_time} seconds.")

# # Get model's precision and ensure all tensors match  
# model_dtype = next(model.parameters()).dtype  
# for k in cols:  
#     tensor = torch.tensor(inputs[k][0]).unsqueeze(0).to(device)  
#     if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:  
#         # Only convert floating point tensors  
#         batch[k] = tensor.to(model_dtype)  
#     else:  
#         # Keep integer tensors as they are  
#         batch[k] = tensor 

# # Force all model parameters to bfloat16  
# for param in model.parameters():  
#     param.data = param.data.to(torch.bfloat16)  
  
# # Also ensure all buffers are bfloat16  
# for buffer in model.buffers():  
#     buffer.data = buffer.data.to(torch.bfloat16)  
