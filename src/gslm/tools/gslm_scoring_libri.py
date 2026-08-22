import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import logging
import argparse
import json
import scipy.stats
from omegaconf import OmegaConf
from fairseq import utils
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.tacotron2.vocoder import TacotronVocoder
from sampler import UnitLanguageModelSampler
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from operator import itemgetter
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import re
import warnings
from wordfreq import word_frequency
import math

warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

start_time = time.time()

log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger(__name__)

print(torch.cuda.is_available())  # True if a GPU is detected
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.current_device())  # Index of the current device
print(torch.cuda.get_device_name(0))  # Name of GPU 0

MODEL_TYPE="GSLM"
MODEL_NAME="GSLM"
GSLM_INPUT_SAMPLE_RATE = 16000
FIELDNAMES = ["filename", "speaker", "ppl_loss", "human_annotated_accuracy"]

class GslmSpeechPplWrapper:
    def __init__(
        self, 
        language_model_dir: str,
        seed: int = None,
        temperature: float = 0.7,
        vocab_size: int = 100,
        device: str = "cpu",
    ):
        logger.info("Initializing the GSLM pipeline.")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            utils.set_torch_seed(seed)
        self.input_sample_rate = GSLM_INPUT_SAMPLE_RATE
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.tokens_framerate = 0.02  # HuBERT framerate
        self.max_length = 1000
        self.trim_trailing_audio_frames = 200
        self.sampling_kwargs = {
            "temperature": self.temperature,
            "sampling": True,
            "beam": 1,
            "prefix_size": -1,
            "max_len_a": 0.0,
            "max_len_b": self.max_length,
        }
        logger.info("... Loading the language model")
        self.sampler = UnitLanguageModelSampler.from_pretrained(
            language_model_dir,
        )
        logger.info("=> Done!")
        logger.info("... Loading the encoder")

        self.speech_encoder = SpeechEncoder.by_name(
            dense_model_name="hubert-base-ls960",
            quantizer_model_name="kmeans",
            vocab_size=vocab_size,
            need_f0=False,
            deduplicate=False, # set to False to mannually deduplicate later if needed
            f0_normalizer=None,
            f0_quantizer=None,
        )

        # move to device and eval mode
        self.device = device
        self.speech_encoder = self.speech_encoder.to(self.device)
        # self.sampler.model = self.sampler.model.to(self.device)
        self.sampler = self.sampler.to(self.device) # make sure the sampler knows the device
        logger.info(f"Sampler model device: {self.sampler.device}")
        self.speech_encoder.eval()
        self.sampler.model.eval()

        logger.info("=> Done!")
        logger.info("GSLM pipeline initialized!")

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
        
        # turn two channeled audio into 1 channel (just one tensor)
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        
        # get audio units... tokenize, if you will
        encoder_output = self.speech_encoder(raw_audio)
        units = encoder_output['units']
        
        # perform deduplication [4,4,3,3,3,3,3,1,2,2...] -> [4,3,1,2]
        input_ids, _durations = torch.unique_consecutive(units, return_counts=True) # return the durations of each token (how many copies)
        input_ids = input_ids.unsqueeze(0)  # add batch dim (1, seq_len)

        # deal with durations and convert to seconds
        token_ends_frames = torch.cumsum(_durations, dim=0) # [5, 3+5, 1+3+5, ...]
        token_starts_frames = token_ends_frames - _durations # [0, 5, 8, ...]
        t_start = token_starts_frames * self.tokens_framerate  # seconds
        t_end = token_ends_frames * self.tokens_framerate

        # making training samples!!!
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit language model logits (Hubert units)
        logits = self.sampler.model(input_ids)[0] # raw predicted scores!! No softmax becaue cross_entropy does it
        
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
    parser = argparse.ArgumentParser()
    #parser.add_argument("--testing_audio_fpath", type=str, default=None)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=False)
    parser.add_argument("--language_model_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=False)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--alignments", type=str)

    args = parser.parse_args()
    
    open('/home/u5504709/new_work/speech_ppl/src/gslm/tools/error_log', 'w').close()

    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model
    model = GslmSpeechPplWrapper(
        language_model_dir=args.language_model_dir,
        seed=None,
        temperature=0.7,
        vocab_size=100,
        device=device,
    )

    # info about the program
    print(f"Language model: {MODEL_NAME}")
    print(f"Model Input Sample Rate: {model.input_sample_rate}")
    print(f"Device: {device}")

    # function for localizing ppl function
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
    GRANULARITY = "word"
    result_dicts_path = "/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts"

    for pool in ["mean", "max"]:
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

            print(f"Words to be sorted: {len(result_dict)}")
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