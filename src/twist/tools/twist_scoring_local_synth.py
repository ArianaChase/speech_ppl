import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import scipy.stats
from transformers import AutoModelForCausalLM
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder
from textless.data.speech_encoder import SpeechEncoder
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import csv 
import pandas as pd
import json 
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import math
from wordfreq import word_frequency
from sklearn.metrics import roc_auc_score
from pathlib import Path

start_time = time.time()

MODEL_TYPE="TWIST"
MODEL_NAME="TWIST-1.3B"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

class TwistSpeechPPLWrapper:
    def __init__(
        self,
        twist_model_pretrained_path,
        dense_model="mhubert-base-25hz",
        quantizer_model="kmeans",
        vocab=500,
        device=None,
    ):
        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")

        # Load speech encoder and vocoder
        self.encoder = SpeechEncoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer_model,
            vocab_size=vocab,
            deduplicate=False, # set to False but mannually deduplicate later if needed
            need_f0=False,
            add_bos_eos=False,
        ).eval().to(self.device)

        # build twist unit lm
        self.twist_lm = AutoModelForCausalLM.from_pretrained(twist_model_pretrained_path).to(self.device) # this is a text LLM
        self.twist_lm.eval()
    
    @torch.no_grad()
    def get_per_token_losses(
        self,
        audio_sample
    ) -> dict:
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)

        # get input ids for unit lm
        units = self.encoder(raw_audio)['units']

        # perform deduplication
        input_ids, _durations = torch.unique_consecutive(units, return_counts=True)
        input_ids = input_ids.unsqueeze(0) + self.twist_lm.config.offset # Speech vocab embeddings live behind the text embeddings, offset by a certain number. this is ensuring that the input ids are not clashing with pretrained text embeddings // (1, seq_len)

        # deal with durations and convert to seconds
        FRAMERATE = 1 / 25
        token_ends_frames = torch.cumsum(_durations, dim=0) # [5, 3+5, 1+3+5, ...]
        token_starts_frames = token_ends_frames - _durations # [0, 5, 8, ...]
        t_start = token_starts_frames * FRAMERATE  # seconds
        t_end = token_ends_frames * FRAMERATE
        
        # prepare labels
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit lm logits
        logits = self.twist_lm(input_ids)[0]
        logits_reshaped = logits.reshape(-1, logits.size(-1))

        # calcuate CE loss
        loss_all_tokens = F.cross_entropy(
            logits_reshaped,
            labels.reshape(-1).long(),
            ignore_index=-100,
            reduction='none',
        )

        # merge duration info
        loss_with_timestamps = []
        for i in range(0, len(loss_all_tokens) - 1): # don't consider the EOS
            loss_with_timestamps.append((loss_all_tokens[i].item(), t_start[i+1].item(), t_end[i+1].item()))
        
        return {
            "logits": logits,
            "loss_all_tokens": loss_all_tokens,
            "loss_with_timestamps": loss_with_timestamps
        }

def create_csv_file(output_dir, name):
    filename = '%s/%s' % (output_dir, name)

    print("Creating csv with file name: ", filename, " ...")

    with open(filename, mode="w") as csvfile:
        fieldnames = ["Speaker", "Audio filename", "Raw Mean of Per Token Losses", "Human Annotation (Accuracy)", "Human Annotation (Fluency)", "Human Annotation (Prosody)", "Human Annotation (Completeness)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filename

def is_overlapping(a_start, a_end, b_start, b_end):
    if (a_end >= b_start and a_start <= b_end):
        return True
    else:
        return False

def strip_stress(phone_label):
    if phone_label[-1].isdigit():
        return phone_label[:-1]
    else:
        return phone_label

def process_synth(input_dataset, audio_version):

        audio_file_info = []

        pbar = tqdm(os.listdir(input_dataset))

        for file_path in pbar:
            audio_path = f"{input_dataset}/{file_path}"
            file_metadata = os.path.basename(audio_path).split("_")
            filename = file_metadata[0]
            ver = Path(file_metadata[1]).stem

            if ver != audio_version:
                continue

            audio_file_info.append({
                "filename" : filename,
                "audio_version" : ver,
                "path" : audio_path
            })
        return {
            "processed" : audio_file_info,
        }

def get_losses(dataset, alignments_path, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    alignments_path : to filter the dataset to those that have alignments to compare to downstream
    '''
    per_token_losses = []
    file_count = 0
    lim = limit if limit != None else len(dataset)


    pbar = tqdm(dataset)
    for sample in pbar:
        if file_count >= lim:
            break
        
        # info
        file_path = sample["path"]
        filename = sample["filename"]
        ver = sample['audio_version']
        pbar.set_description(f"Getting per phone losses for file: {filename}")

        # load audio
        audio, sr = torchaudio.load(file_path)
        audio = audio.to(device)
        
        # get ppl_losses per token + timestamps
        losses_with_timestamps = get_per_token_losses(audio)["loss_with_timestamps"]

        for idx, loss in enumerate(losses_with_timestamps):
            per_token_losses.append({
                "audio_version" : ver,
                "file_path" : file_path,
                "filename" : filename,
                "token_idx" : idx,
                "ppl_loss" : loss[0],
                "start" : loss[1],
                "end" : loss[2]
            })

        file_count += 1

    return {
        "results" : per_token_losses,
    }

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Twist Speech PPL Wrapper Test")
    argparser.add_argument("--name", type=str, required=True)
    argparser.add_argument("--language_model_dir", type=str, required=True, help="Path to pretrained twist model")
    argparser.add_argument("--dataset_dir", type=str, required=True, help="Path to input dataset")
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--labels_dir", type=str, required=True)
    argparser.add_argument("--device", type=str, default=None, help="Device to use, e.g., 'cpu' or 'cuda'")
    argparser.add_argument("--alignments", type=str)
    argparser.add_argument("--root_dir", type=str)

    args = argparser.parse_args()

    open(f'{args.root_dir}/src/twist/tools/error_log', 'w').close()
    
    # get device
    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # build model
    model = TwistSpeechPPLWrapper(
        twist_model_pretrained_path=args.language_model_dir,
        device=device
    )
    
    def get_per_token_losses(
        audio_sample: torch.Tensor,
    ) -> dict:
        return model.get_per_token_losses(audio_sample)
    
    # process dataset
    input_dataset = args.dataset_dir
    AUDIO_VERSION = "dist"
    processed = process_synth(input_dataset, AUDIO_VERSION)
    processed_dataset = processed["processed"]
    print(f"Processed {len(processed_dataset)} samples.")

    # calculate losses

    OUTPUT_DIR = args.output_dir
    
    csv_path = f"{OUTPUT_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{AUDIO_VERSION}_setB_per_token_losses.csv"   

    results = get_losses(
        dataset=processed_dataset, 
        alignments_path=args.alignments, 
        limit=None,
        )
            
    ppl_results = results["results"]

    with open(csv_path, "w") as f:
        fieldnames = ppl_results[0].keys()
        dict_writer = csv.DictWriter(f, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(ppl_results)
                
    print(f"File count: {len(processed_dataset)}")

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    duration = time.time() - start_time
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")

