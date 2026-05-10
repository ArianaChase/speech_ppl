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
from datetime import datetime
import scipy.stats
from datasets import Dataset  
from torch.utils.data import DataLoader  
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

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
        fieldnames = ["Speaker", "Audio filename", "Raw MSE", "Human Annotation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filename

def pad_seq_collate_fn(batch):
    padded = {}
    for key in batch[0].keys():
        packed_list = []
        for x in batch:
            val = x[key]
            
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val)
            else:
                val = val.clone().detach()
                
            if val.ndim > 1 and val.shape[0] == 1:
                val = val.squeeze(0)
                
            packed_list.append(val)

        if 'length' in key:
            padded_tensor = torch.tensor([v.flatten()[0].item() for v in packed_list], dtype=torch.long)
        else:
            padded_tensor = pad_sequence(packed_list, batch_first=True, padding_value=0)

        padded[key] = padded_tensor
        
    return padded

def process_audio_item(item):
    print(f"Processing {item['filename']}...")
    
    # This uses the 'processor' defined in your main script

    inputs = processor(
        item["path"],
        sampling_rate,          
        ref_audio_list=[item["path"]]  
    )
    
    # Foolproof Filtering
    cols_to_check = ['audio_features', 'asr_token_ids', 'llm_token_ids']
    for col in cols_to_check:
        if len(inputs[col][0]) == 0:
            return None
    
    inputs.update(item)

    tensor_only_dict = {k: inputs[k] for k in cols if k in inputs}

    padded_batch = pad_seq_collate_fn([tensor_only_dict])

    padded_batch["path"] = item["path"]
    padded_batch["speaker"] = item["speaker"]
    padded_batch["filename"] = item["filename"]

    #print("batch: ", padded_batch)

    return padded_batch

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
                "filename" : file[0:9]
            })
        counter += 1

    return audio_paths  

def get_MSE(original, recon):
    # 1. Load original and immediately move to GPU
    orig_wav, orig_sr = torchaudio.load(original["path"])
    orig_wav = orig_wav.to(device)
    
    # 2. Fast resampling (Avoids the transforms.Resample instantiation trap)
    if orig_sr != recon["sr"]:
        orig_wav = torchaudio.functional.resample(orig_wav, orig_sr, recon["sr"])
        
    # 3. Calculate Mels entirely on GPU
    orig_mel = mel_transform(orig_wav)
    recon_mel = mel_transform(recon["tensor"])
    
    # 4. Replicate librosa.power_to_db(ref=np.max) on GPU
    # torchaudio amplitude_to_DB is highly optimized
    orig_mel_db = torchaudio.functional.amplitude_to_DB(orig_mel, multiplier=10.0, amin=1e-10, db_multiplier=1.0, top_db=80.0)
    orig_mel_db -= orig_mel_db.max() # Equivalent to ref=np.max
    
    recon_mel_db = torchaudio.functional.amplitude_to_DB(recon_mel, multiplier=10.0, amin=1e-10, db_multiplier=1.0, top_db=80.0)
    recon_mel_db -= recon_mel_db.max() # Equivalent to ref=np.max

    # 5. Trim to match lengths
    min_len = min(orig_mel_db.shape[-1], recon_mel_db.shape[-1])
    orig_mel_db = orig_mel_db[..., :min_len]
    recon_mel_db = recon_mel_db[..., :min_len]

    # 6. Calculate MSE on GPU and return just the single float value
    score = torch.nn.functional.mse_loss(orig_mel_db, recon_mel_db)
    return score.item()

start_time = time.time()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) 
    # =================1. Preparing initial data

    input_dataset = "/home/u5504709/new_work/speech_ppl/speechocean762/WAVE/"
    score_labels = "/home/u5504709/new_work/speech_ppl/speechocean762/resource/scores.json"
    data_size = 1
    audio_info = getWAVfiles(input_dataset, data_size) # returns a list of dictionaries in the format of {"path" : path to audio file}
    print("Saved ", len(audio_info), " audio files from input dataset.")
    print(audio_info)
    output_csv = create_csv_file("/home/u5504709/new_work/speech_ppl/work/outputs", "taslm_reconstruction_001")

    # -- Saving human annotated scores --
    accuracy_scores = parse_accuracy_scores(score_labels)
    accuracy_scores = dict(sorted(accuracy_scores.items()))
    human_annotations_values = []

    for key, value in accuracy_scores.items():
        human_annotations_values.append(value)

    # ===================== 2. Loading model

    print("Loading model...")
    device = 0
    model_id = 'MediaTek-Research/Llama-1B-TASTE-V0'  
    model_dtype = torch.bfloat16
    sampling_rate = 16000 

    model = TasteForCausalLM.from_pretrained(  
        model_id,   
        torch_dtype=model_dtype,
        low_cpu_mem_usage=False  
    ).to(device).eval()  

    # Force all model parameters to bfloat16  
    for param in model.parameters():  
        param.data = param.data.to(model_dtype)  

    # Also ensure all buffers are bfloat16  
    for buffer in model.buffers():  
        buffer.data = buffer.data.to(model_dtype) 

    processor = TasteProcessor.from_pretrained(model_id) # preparation for the audio
    generator = processor.get_generator(device=device) # decoder - tokens to audio

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, 
        n_fft=1024, 
        hop_length=256, 
        n_mels=80,
        power=2.0 # Librosa default is power=2.0
    ).to(device)

    # ============== 3. Create Dataset and DataLoader  

    batch_size = 32
    cols = ['speaker_embeds', 'audio_features', 'audio_feature_lengths',  
            'asr_token_ids', 'asr_token_lengths', 'asr_word_ids',  
            'llm_token_ids', 'llm_token_lengths', 'llm_word_ids']  

    results_to_write = []

    with ThreadPoolExecutor(max_workers=4) as executor:

        batch_idx = 0

        for batch in tqdm(executor.map(process_audio_item, audio_info), total=len(audio_info)):

            print("Batch inferencing, iteration: ", batch_idx)
            if batch is None: # Skip files that failed filtering
                batch_idx += 1
                continue

            # Safely move batch to GPU AND ensure correct precision matching
            formatted_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device) # Move to GPU
                    # Only convert floating point tensors to match the model (bfloat16)
                    if v.dtype in [torch.float32, torch.float16, torch.float64]:  
                        v = v.to(model_dtype)
                formatted_batch[k] = v
            batch = formatted_batch

            # Run inference without gradient computation (saves memory)  
            with torch.no_grad():  
                batch = {k: batch[k] for k in cols if k in batch}

                # Reconstruct speech tokens from audio features  
                output = model.inference_reconstruction(**batch)  
                
                # Generate actual audio waveform from speech tokens  
                tts_speech, tts_sr = generator.inference(  
                    speech_token_ids=output['speech_token_ids'],         # Generated speech tokens  
                    speech_token_lengths=output['speech_token_lengths'], # Token sequence lengths  
                    flow_embedding=batch['speaker_embeds']               # Speaker characteristics  
                ) 
                tts_speech = tts_speech.to(torch.float32).to(device)

                print(f" Scoring index: { batch_idx }, speaker is { audio_info[ batch_idx ]['speaker'] }")
                
                original_info = audio_info[batch_idx]
                recon_info = {
                    "tensor" : tts_speech,
                    "sr" : tts_sr
                }

                score = get_MSE(original_info, recon_info)

                results_to_write.append({
                    "Speaker": original_info["speaker"], 
                    "Audio filename": original_info["filename"], 
                    "Raw MSE": score, 
                    "Human Annotation": accuracy_scores[original_info["filename"]]
                    })
                    
            print(f"Finished processing batch {batch_idx}!")
            batch_idx += 1

    with open(output_csv, mode="a", newline="") as csvfile:
        fieldnames = ["Speaker", "Audio filename", "Raw MSE", "Human Annotation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerows(results_to_write)

    output_csv_df = pd.read_csv(output_csv)
    x = output_csv_df["Raw MSE"].values
    y = human_annotations_values[0:data_size*20]

    print("Correlation x len: ", len(x))
    print("Correlation y len: ", len(y))
        
    print(f"Correlation value is: {scipy.stats.pearsonr(x, y)}")

    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    print(f"Program 'taste_reconstruction' finished executing in {time.time() - start_time} seconds.")
