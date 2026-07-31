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
import re
import warnings
from wordfreq import word_frequency
import math
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
        for i in range(0, len(loss_all_tokens)):
            loss_with_timestamps.append((loss_all_tokens[i].item(), t_start[i].item(), t_end[i].item()))
        
        return {
            "logits": logits,
            "loss_all_tokens": loss_all_tokens,
            "loss_with_timestamps": loss_with_timestamps
        }

def is_overlapping(a_start, a_end, b_start, b_end):
    if (a_end >= b_start and a_start <= b_end):
        return True
    else:
        return False


def process_librispeech(input_dataset):

    audio_file_info = []

    for sample in input_dataset:

        audio_file_info.append({
            "speaker" : sample["speaker_id"],
            "array" : sample["audio"]["array"],
            "sr" : sample["audio"]["sampling_rate"],
            "text" : sample["text"],
            "filename" : sample["id"] 
        }) 

    return audio_file_info

def process_alignments_ds(input_dataset):
    
    alignments = []

    for sample in input_dataset:
        phone_list = []
        word_list = []

        for phone_alignment in sample['phonemes']:
            phone_list.append({
                "start" : phone_alignment['start'],
                "end" : phone_alignment["end"],
                "label" : phone_alignment["phoneme"]
            })

        for word_alignment in sample["words"]:
            word_list.append({
                "start" : word_alignment['start'],
                "end" : word_alignment["end"],
                "label" : word_alignment['word']
            })

        alignments.append({
            "audio_id" : sample['id'],
            "phone_alignment" : phone_list,
            "word_alignment" : word_list
        })
    
    return alignments
def strip_stress(phone_label):
    if phone_label[-1].isdigit():
        return phone_label[:-1]
    else:
        return phone_label


def get_losses(dataset, granularity, alignments_ext, pooling, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    alignments      : alignments
    pooling         : pooling method (max/mean/std)
    '''
    result_dict = {}
    file_count = 0
    error_log = []
    nan_count = 0
    lim = limit if limit != None else len(dataset)

    pbar = tqdm(dataset)
    for sample in pbar:
        if file_count >= lim:
            break
        
        # info
        speaker = sample["speaker"]
        filename = sample["filename"]
        pbar.set_description(f"Getting per phone losses for file: {filename}")

        # load audio
        audio = torch.Tensor(sample["array"])
        sr = sample["sr"]
        audio = audio.to(device)

        # external preparation
        alignment_obj = next((item for item in alignments_ext if item.get('audio_id') == filename), None)
        phone_alignments = alignment_obj["phone_alignment"] # type: ignore # list of phone objects {start, end, label}
        word_alignments = alignment_obj["word_alignment"] # type: ignore # list of phone objects {start, end, label}

        if granularity == "phone":
            alignments = phone_alignments
        elif granularity == "word":
            alignments = word_alignments
        else:
            raise Exception("Invalid granularity.")

        # get ppl_losses per token + timestamps
        losses_with_timestamps = get_per_token_losses(audio)["loss_with_timestamps"]

        # aggregate
        for i in range(0, len(alignments)):
            current_alignment = alignments[i]

            a_start = current_alignment["start"]
            a_end = current_alignment["end"]
            losses = []

            for loss_item in losses_with_timestamps:
                t_start = loss_item[1]
                t_end = loss_item[2]
                if is_overlapping(a_start, a_end, t_start, t_end):
                    losses.append(loss_item[0])
            
            # pooling
            loss_pooled = np.nan
            
            if pooling == "mean":
                loss_pooled = np.mean(losses) if len(losses) > 0 else np.nan
            elif pooling == "max":
                loss_pooled = np.max(losses) if len(losses) > 0 else np.nan
            elif pooling == "std":
                loss_pooled = np.std(losses) if len(losses) > 1 else np.nan
            else:
                raise Exception("No pooling method specified.")
            
            # if np.isnan(loss_pooled):
            #     nan_count += 1

            if granularity == "phone":
                phone_label = strip_stress(alignments[i]['label'])

                if phone_label in result_dict:
                    result_dict[phone_label]['count'] += 1
                    result_dict[phone_label]['losses'].append(loss_pooled)
                else:
                    result_dict[phone_label] = {
                        "count" : 1,
                        "losses" : [loss_pooled]
                    }
            elif granularity == "word":
                # TODO: Implement wordfreq normalization here
                word = alignments[i]['label']
                freq = word_frequency(word, 'en')
                neg_log_freq = -math.log(freq) if freq > 0 else np.nan  # guard against unknown words

                if word in result_dict:
                    result_dict[word]['freq'] = neg_log_freq
                    result_dict[word]['losses'].append(loss_pooled)
                else:
                    result_dict[word] = {
                        'freq' : neg_log_freq,
                        'losses' : [loss_pooled]
                    }

        file_count += 1

    with open("/home/u5504709/new_work/speech_ppl/src/gslm/tools/error_log", "a") as f:
        for i in error_log:
            f.write(i)
            f.write("\n")

    return result_dict

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Twist Speech PPL Wrapper Test")
    argparser.add_argument("--name", type=str, required=True)
    argparser.add_argument("--language_model_dir", type=str, required=True, help="Path to pretrained twist model")
    argparser.add_argument("--dataset_dir", type=str, required=True, help="Path to input dataset")
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--labels_dir", type=str, required=True)
    argparser.add_argument("--device", type=str, default=None, help="Device to use, e.g., 'cpu' or 'cuda'")
    argparser.add_argument("--index", type=int, required=True)
    argparser.add_argument("--category", type=str, required=True)
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--alignments", type=str)

    args = argparser.parse_args()

    open('/home/u5504709/new_work/speech_ppl/src/twist/tools/error_log', 'w').close()
    
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
    
    # alignments
    from datasets import load_dataset
    alignments_ds = load_dataset("gilkeyio/librispeech-alignments", streaming=True)
    processed_alignments_ds = process_alignments_ds(alignments_ds["dev_clean"])

    # process dataset
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)

    processed_dataset = process_librispeech(dataset)
    print(f"Processed {len(processed_dataset)} samples.")

    # calculate losses

    NORM = False
    GRANULARITY = "phone"
    result_dicts_path = "/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts"

    for pool in ["std"]:
        result_dict = get_losses(
            dataset=processed_dataset, 
            granularity=GRANULARITY,
            alignments_ext=processed_alignments_ds,
            pooling=pool,
            limit=None,
            )

        if GRANULARITY == "phone":
            result_dict.pop("spn")

            for key, phone_info in result_dict.items():
                result_dict[key]['mean'] = np.mean(result_dict[key]['losses'])
                result_dict[key]['std'] = np.std(result_dict[key]['losses'])

            with open(f"{result_dicts_path}/{MODEL_NAME}_phone_{pool}_norm.json", "w") as f:
                json.dump(result_dict, f)

            with open("/home/u5504709/new_work/speech_ppl/src/gslm/tools/error_log", "a") as f:
                f.write(f"In total, there are {len(result_dict)} unique phones in the {pool} dictionary.")
                f.write("\n")
                f.write(f"{sorted(list(result_dict.keys()))}\n")

        elif GRANULARITY == "word":

            # TODO: Implement bucketing here, then create a dictionary with only the buckets

            NUM_BUCKETS = 5

            all_neg_log_freqs = [item['freq'] for word, item in result_dict.items()]
            nan_count = np.isnan(all_neg_log_freqs).sum()
            x_series = pd.Series(all_neg_log_freqs)
            nan_count = x_series.isna().sum()
            all_neg_log_freqs = x_series.dropna().to_numpy() 
            
            bucket_boundaries = pd.qcut(all_neg_log_freqs, q=NUM_BUCKETS)

            print(f"all neg log freqs: {all_neg_log_freqs}, nan count {nan_count}")
            print(f"Boundaries: {bucket_boundaries}, type {type(bucket_boundaries)}")


            bucketed_word_dict = {}


            for idx, bucket in enumerate(np.unique(bucket_boundaries)):
                print(f"Bucket {idx}: {bucket}")
                print(f"Type: {type(bucket)}")

                bucket_losses = []
                bucket_word_count = 0

                for word, info in result_dict.items():
                    if info['freq'] in bucket:
                        bucket_losses.append(np.nanmean(info['losses'])) 
                        bucket_word_count += 1

                bucketed_word_dict[idx] = {
                    'freq_range' : str(bucket),
                    'losses' : bucket_losses,
                    'mean' : np.mean(bucket_losses),
                    'std' : np.std(bucket_losses),
                    'count' : bucket_word_count
                }
            
            with open(f"{result_dicts_path}/{MODEL_NAME}_word_{pool}_norm.json", "w") as f:
                json.dump(bucketed_word_dict, f)
    

    print(f"File count: {len(processed_dataset)}")

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    duration = time.time() - start_time
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")