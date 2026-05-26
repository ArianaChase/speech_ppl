import torch.nn.functional as F  
import torch
from moshi.models import loaders 
from huggingface_hub import hf_hub_download  
import sphn 
from operator import itemgetter
import json 
import os
from tqdm import tqdm
import csv
from datetime import datetime
import time
import pandas as pd
import scipy.stats

def parse_human_annotations(filename):
    human_scores = []
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            human_scores.append({
                "filename" : os.path.basename(audio_file),
                "accuracy" : value["accuracy"],
                "fluency" : value["fluency"],
                "prosodic" : value["prosodic"],
                "completeness" : value["completeness"]
            })
    return human_scores

def create_csv_file(output_dir, name):
    filename = '%s/%s' % (output_dir, name)

    print("Creating csv with file name: ", filename, " ...")

    with open(filename, mode="w") as csvfile:
        fieldnames = ["Speaker", "Audio filename", "original_path", "recon_path", "Raw MSE", "Human Annotation (Accuracy)", "Human Annotation (Fluency)", "Human Annotation (Prosody)", "Human Annotation (Completeness)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filename

def getWAVfiles(input_dataset, amt, labels_list):

    audio_paths = []

    speakerWAV = tqdm(sorted(os.listdir(input_dataset)))

    counter = 0;

    for dir in speakerWAV:
        if (counter >= amt):
            break
        speaker = dir[7:None]

        if (speaker != "1076"):
            speakerWAV.set_description(f"Appending speaker: {speaker}")
            speakerWAV_path = os.path.join(input_dataset, dir)
            for file in os.listdir(speakerWAV_path):
                file_info = {
                    "path" : os.path.join(input_dataset, dir, file),
                    "speaker" : speaker,
                    "filename" : file[0:9]
                }

                if (any(file_info["filename"] == x["filename"] for x in labels_list)):
                    audio_paths.append(file_info)
                
        
        counter += 1

    return audio_paths  

start_time = time.time()

# -- Prepare tokenizer --
mimi_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", loaders.MIMI_NAME)  
mimi = loaders.get_mimi(mimi_path, device="cuda")
mimi.set_num_codebooks(8)  

# -- Saving human annotated scores --
score_labels = "/home/u5504709/new_work/speech_ppl/speechocean762/resource/scores.json"
human_scores = parse_human_annotations(score_labels)
human_scores = sorted(human_scores, key=itemgetter("filename"))
human_score_map = {
    x["filename"]: x
    for x in human_scores
}

# -- Initializing data --
input_dataset = "/home/u5504709/new_work/speech_ppl/speechocean762/WAVE"
data_size = 250
audio_info = getWAVfiles(input_dataset, data_size, human_scores)
audio_data_size = len(audio_info)
print("Saved ", audio_data_size, " audio files from input dataset.")
output_csv = create_csv_file("/home/u5504709/new_work/speech_ppl/work/outputs", "mimi_reconstruction_001")
output_folder = "/home/u5504709/new_work/speech_ppl/work/mimi_reconstructed_audio"

# Checking
for thing in audio_info:
    if (not any(thing["filename"] == x["filename"] for x in human_scores)):
        print(f"FILE {thing['filename']} NOT FOUND IN HUMAN ANNOTATED SCORES")

print("EXISTENCE CHECK FINISHED.")

# -- Reconstruction --

num_files = 0  
batch_size = 8
results_to_write = []

with torch.no_grad():  

    for i in tqdm(range(0, len(audio_info), batch_size), desc="Reconstructing Audio"):  
        batch_files = audio_info[i:i + batch_size]  
        batch_info = []  

        for file_info in batch_files:
            wav, sr = sphn.read(file_info["path"], sample_rate=mimi.sample_rate)  

            original = torch.from_numpy(wav).to(device='cuda')  
            original = original[None, 0:1] 

            # padding
            frame_size = mimi.frame_size  
            padding = (frame_size - (original.shape[-1] % frame_size)) % frame_size  
            if padding > 0:  
                original = torch.nn.functional.pad(original, (0, padding))  

            file_length = original.shape[-1] - padding

            batch_info.append({
                "path" : file_info["path"],
                "speaker" : file_info["speaker"],
                "filename" : file_info["filename"],
                "audio" : original,
                "length" : file_length,
            })

        # max padded length in batch
        max_len = max(item["audio"].shape[-1] for item in batch_info)

        # batch tensor
        batch_tensor = torch.cat([
            torch.nn.functional.pad(
                item["audio"],
                (0, max_len - item["audio"].shape[-1])
            )
            for item in batch_info
        ], dim=0)  

        codes = mimi.encode(batch_tensor)
        reconstructed = mimi.decode(codes)  

        for j, item in enumerate(batch_info):

            length = item["length"]

            recon = reconstructed[j:j+1, :, :length]
            orig = item["audio"][:, :, :length]

            loss = F.mse_loss(recon, orig).item()

            output_path = os.path.join(output_folder, item["filename"] + "_recon.wav")
            sphn.write_wav(str(output_path), recon[0, 0].cpu().numpy(), mimi.sample_rate)

            human_annotation_obj = human_score_map[item["filename"]]

            results_to_write.append({
                "Speaker" : item["speaker"],
                "Audio filename" : item["filename"],
                "original_path": item["path"],
                "recon_path" : output_path,
                "Raw MSE": loss,
                "Human Annotation (Accuracy)" : human_annotation_obj["accuracy"],
                "Human Annotation (Fluency)" : human_annotation_obj["fluency"],
                "Human Annotation (Prosody)" : human_annotation_obj["prosodic"],
                "Human Annotation (Completeness)" : human_annotation_obj["completeness"],
            })

# -- Writing --
with open(output_csv, mode="a", newline="") as csvfile:
    fieldnames = ["Speaker", "Audio filename", "original_path", "recon_path", "Raw MSE", "Human Annotation (Accuracy)", "Human Annotation (Fluency)", "Human Annotation (Prosody)", "Human Annotation (Completeness)"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    print(f"Writing {len(results_to_write)} items to csv...")
    writer.writerows(results_to_write)

# -- Correlation --
output_csv_df = pd.read_csv(output_csv)
x = output_csv_df["Raw MSE"].values

def calc_correlation(x, dim):
    if (dim == "accuracy"):
        y = output_csv_df["Human Annotation (Accuracy)"]
    elif (dim == "fluency"):
        y = output_csv_df["Human Annotation (Fluency)"]
    elif (dim == "prosodic"):
        y = output_csv_df["Human Annotation (Prosody)"]
    elif (dim == "completeness"):
        y = output_csv_df["Human Annotation (Completeness)"]
    else:
        print(f"Invalid dimension")
        return
    
    print(f"=== Correlation for dimension {dim} ===")
    print("Correlation x len: ", len(x))
    print("Correlation y len: ", len(y))
    print(f"Correlation value is: {scipy.stats.pearsonr(x, y)}")

calc_correlation(x, "accuracy")
calc_correlation(x, "fluency")
calc_correlation(x, "prosodic")
calc_correlation(x, "completeness")

now = datetime.now() 
finish_time = now.strftime("%m-%d-%Y %H:%M") 
print(f"Date and time at completion: {finish_time}") 
print(f"Program 'taste_reconstruction' finished executing in {time.time() - start_time} seconds.")

