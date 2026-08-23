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
from wordfreq import word_frequency
import math
from sklearn.metrics import roc_auc_score

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

        # token : 1, 2, 3, 4, 5, 6, 7, 8
        # label : 2, 3, 4, 5, 6, 7, 8, E
        # loss  : 2, 3, 4, 5, 6, 7, 8, E
        # start : 1, 2, 3, 4, 5, 6, 7, 8

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

def strip_stress(phone_label):
    if phone_label[-1].isdigit():
        return phone_label[:-1]
    else:
        return phone_label

def process_speechocean(input_dataset):

        audio_file_info = []
        spk_count = 0
        ignored_speakers = ["1076"]
        removed = []

        pbar = tqdm(sorted(os.listdir(input_dataset)))

        for spk_dir in pbar:
            spk_count += 1
            speaker = spk_dir[7:None]
            pbar.set_description(f"Processing speaker: {speaker}")
            spk_dir_path = os.path.join(input_dataset, spk_dir)
            for audio_file in os.listdir(spk_dir_path):
                audio_path = os.path.join(spk_dir_path, audio_file)
                filename = os.path.basename(audio_path)[0:9]
                audio_file_info.append({
                    "speaker" : speaker,
                    "filename" : filename,
                    "path" : audio_path
                })
        
        for i in range(len(audio_file_info) - 1, -1, -1):
            if audio_file_info[i]["speaker"] in ignored_speakers:
                removed.append(audio_file_info.pop(i))
                          

        return {
            "processed" : audio_file_info,
            "ignored" : removed,
            "spk_count" : spk_count - len(ignored_speakers)
        }


    
def get_losses(dataset, labels_dict, alignments_path, granularity, pooling, norm_dict=None, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    labels_dict     : dictionary of human annotated information, sorted by filename 
    alignments_path : json file of phone-level alignment boundaries
    granularity     : phone/word/utterance level
    pooling         : pooling method (max/mean/std)
    norm_dict      : dict for normalization
    '''
    ppl_info = []
    file_count = 0
    error_log = []
    nan_count = 0
    lim = limit if limit != None else len(dataset)


    with open(alignments_path, 'r') as f:
        alignment_list = json.load(f)

    dataset_cleaned = []

    for sample in dataset:
        for idx in range(len(alignment_list) -1, -1, -1):
            if sample['filename'] == alignment_list[idx]['audio_id']:
                dataset_cleaned.append(sample)
            if alignment_list[idx]['speaker'] == '1076':
                alignment_list.pop(idx)
                
    if len(dataset_cleaned) != len(alignment_list):
        raise Exception(f"Length mismatch between alignments ({len(alignment_list)}) and dataset ({len(dataset_cleaned)})")

    pbar = tqdm(dataset_cleaned)
    for sample in pbar:
        if file_count >= lim:
            break
        
        # info
        speaker = sample["speaker"]
        file_path = sample["path"]
        filename = sample["filename"]
        utterance_ppl_info = []
        pbar.set_description(f"Getting per phone losses for file: {filename}")

        # load audio
        audio, sr = torchaudio.load(file_path)
        audio = audio.to(device)

        # external preparation
        alignment_obj = next((item for item in alignment_list if item.get('audio_id') == filename), None)
        phone_alignments = alignment_obj["phone_alignment"] # list of phone objects {start, end, label}
        word_alignments = alignment_obj["word_alignment"] # list of word objects {start, end, label}
        alignments = None

        human_annotation_obj = labels_dict.get(filename)
        phone_scores = []
        word_scores = []
        utterance_score = human_annotation_obj["accuracy"]
        auc_threshold = None

        for word_obj in human_annotation_obj["words"]:
            for i in range(0, len(word_obj["phones"])):
                phone_scores.append({
                    "phone" : word_obj["phones"][i], 
                    "accuracy" : word_obj["phones-accuracy"][i]
                }) 
            word_scores.append({
                "word" : word_obj["text"],
                "accuracy" : word_obj["accuracy"],
                "stress" : word_obj["stress"] # unused for now
            })

        if granularity == "phone":
            # align canonical phonemes and phone alignments
            phone_alignments_labels = [item['label'] for item in phone_alignments]
            phone_scores_labels = [item['phone'] for item in phone_scores]

            matcher = SequenceMatcher(None, phone_scores_labels, phone_alignments_labels)
            opcodes = matcher.get_opcodes()
            has_error = False
            matched_alignments = []
            matched_scores = []
            for tag, a_idx1, a_idx2, b_idx1, b_idx2 in opcodes:
                if tag == "equal":
                    matched_scores.extend(phone_scores[a_idx1:a_idx2])
                    matched_alignments.extend(phone_alignments[b_idx1:b_idx2])
            phone_alignments = matched_alignments
            phone_scores = matched_scores

            if len(phone_alignments) != len(phone_scores):
                error_log.append(f"Alignment mismatch at file {filename}. {len(phone_alignments)} alignments but {len(phone_scores)} scores.")
                error_log.append(f"{[item['label'] for item in phone_alignments]}\n{[item['phone'] for item in phone_scores]}")

            human_scores = phone_scores
            alignments = phone_alignments
            auc_threshold = 0.5

        elif granularity == "word":
            if len(word_scores) != len(word_alignments):
                raise Exception("Human word annotations cannot be aligned with word alignments.")
            human_scores = word_scores
            alignments = word_alignments
            auc_threshold = 3

        elif granularity == "utterance":
            human_scores = utterance_score
            auc_threshold = 3
        else:
            raise Exception("Invalid granularity")
        
        # get ppl_losses per token + timestamps
        losses_with_timestamps = get_per_token_losses(audio)["loss_with_timestamps"]

        # aggregate
        if granularity != "utterance":
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
                
                if np.isnan(loss_pooled):
                    nan_count += 1

                # normalization
                if granularity == "phone":
                    # z-score normalization
                    phone_label = strip_stress(alignments[i]['label']) # type: ignore
                    p_mean = norm_dict[phone_label]['mean'] # type: ignore
                    p_std = norm_dict[phone_label]['std'] # type: ignore
                    loss_pooled_norm = ((loss_pooled - p_mean) / p_std) if p_std > 0 else np.nan
                else :
                    word = alignments[i]['label']
                    freq = word_frequency(word, 'en')
                    neg_log_freq = -math.log(freq) if freq > 0 else np.nan  # guard against unknown words
                    w_mean = None
                    w_std = None

                    for bucket, item in norm_dict.items():
                        s = item['freq_range']
                        clean_s = s.strip("()[]")
                        left_str, right_str = clean_s.split(",")
                        left = float(left_str)
                        right = float(right_str)
                        interval = pd.Interval(left, right, closed="right")

                        if neg_log_freq in interval:
                            w_mean = item['mean']
                            w_std = item['std']

                    if w_mean != None and w_std != None and w_std > 0:
                        loss_pooled_norm = (loss_pooled - w_mean) / w_std
                    else:
                        loss_pooled_norm = np.nan

                utterance_ppl_info.append({
                    "speaker" : speaker,
                    "filename" : filename,
                    "label" : current_alignment['label'],
                    'auc_label' : 1 if human_scores[i]['accuracy'] > auc_threshold else 0,
                    "ppl_loss" : -loss_pooled,
                    "ppl_loss_norm" : -loss_pooled_norm,
                    "human_score": human_scores[i]['accuracy']
                })
        else:
            losses = []
            for loss_item in losses_with_timestamps:
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
            
            if np.isnan(loss_pooled):
                nan_count += 1

            utterance_ppl_info.append({
                "speaker" : speaker,
                "filename" : filename,
                "label" : human_annotation_obj["text"],
                'auc_label' : 1 if human_scores > auc_threshold else 0,
                "ppl_loss" : -loss_pooled,
                "ppl_loss_norm" : np.nan,
                "human_score": human_scores
            })

        if len(alignments) != len(utterance_ppl_info):
            error_log.append(f"[LOSS/ALIGNMENT MISMATCH] Phone segmentation does not match at file {filename}. {len(alignments)} phone alignments but {len(utterance_ppl_info)} losses") 
                
        ppl_info += utterance_ppl_info
        file_count += 1

    with open("/home/u5504709/new_work/speech_ppl/src/gslm/tools/error_log", "a") as f:
        for i in error_log:
            f.write(i)
            f.write("\n")

    return {
        "results" : ppl_info,
        "nan_count" : nan_count
    }

def parse_human_annotations(filename):
    human_scores = {}
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            human_scores[audio_file] = {
                "filename" : audio_file,
                "accuracy" : value["accuracy"],
                "fluency" : value["fluency"],
                "prosodic" : value["prosodic"],
                "completeness" : value["completeness"],
                "words" : value["words"],
                "text" : value["text"]
            }
    return human_scores

def append_to_sheet(
    row_data,
    spreadsheet_name="ICASSP 2026 Experiment Results",
    worksheet_name="main",
    service_account_file="/home/u5504709/new_work/speech_ppl/src/service_account.json"
):
    # Authenticate
    creds = Credentials.from_service_account_file(
        service_account_file,
        scopes=SCOPES
    )

    client = gspread.authorize(creds)

    # Open sheet
    spreadsheet = client.open(spreadsheet_name)
    worksheet = spreadsheet.worksheet(worksheet_name)

    # Append row
    worksheet.append_row(row_data)

    print("Spreadsheet updated successfully.")

def per_phone_auc(results):

    phone_auc_dict = {}

    for result in results:
        if result['label'] not in list(phone_auc_dict.keys()):
            phone_auc_dict[result['label']] = {
                'auc_labels' : [result['auc_label']],
                'ppl_losses' : [result['ppl_loss']],
                'ppl_norm_losses' : [result['ppl_loss_norm']],
            }
        else:
            phone_auc_dict[result['label']]['auc_labels'].append(result['auc_label'])
            phone_auc_dict[result['label']]['ppl_losses'].append(result['ppl_loss'])
            phone_auc_dict[result['label']]['ppl_norm_losses'].append(result['ppl_loss_norm'])
    
    roc_auc_scores = []
    roc_auc_scores_norm = []

    for phone, item in phone_auc_dict.items():
        df = pd.DataFrame(item)
        df = df.dropna(axis=0, subset=['ppl_losses', 'auc_labels'])

        y_true = df['auc_labels']
        y_score = df['ppl_losses']

        if len(np.unique(y_true)) != 1 and len(y_score) >= 1:
            auc = roc_auc_score(y_true, y_score)
            roc_auc_scores.append(auc)

        df = pd.DataFrame(item)
        df = df.dropna(axis=0, subset=['ppl_norm_losses', 'auc_labels'])
        y_true_norm = df['auc_labels']
        y_score_norm = df['ppl_norm_losses']

        if len(np.unique(y_true_norm)) != 1 and len(y_score_norm) >= 1:
            auc = roc_auc_score(y_true_norm, y_score_norm) # because 0 - 1 is akin to big loss - small loss
            roc_auc_scores_norm.append(auc)

    return {
        'auc' : np.nanmean(roc_auc_scores) if len(roc_auc_scores) >= 1 else "n/a",
        'auc_norm' : np.nanmean(roc_auc_scores_norm) if len(roc_auc_scores_norm) >= 1 else "n/a"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--testing_audio_fpath", type=str, default=None)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--language_model_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
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
    
    # get labels to compare to
    score_labels = args.labels_dir
    human_scores = parse_human_annotations(score_labels)

    # process dataset
    input_dataset = args.dataset_dir

    processed = process_speechocean(input_dataset)
    processed_dataset = processed["processed"]
    ignored_samples = processed["ignored"]
    spk_count = processed["spk_count"]
    print(f"Processed {len(processed_dataset)} samples.")

    # calculate losses

    NORM_DICT_DIR = "/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts"
    OUTPUT_DIR = args.output_dir
    
    for granularity in ["phone", "word"]:
        for pool in ["mean", "max", "std"]:

            csv_path = f"{OUTPUT_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{granularity}_{pool}_losses.csv"

            if granularity == "phone" or granularity == "word":
                with open(f"{NORM_DICT_DIR}/{MODEL_NAME}_{granularity}_{pool}_norm.json", "r") as f:
                    norm_dict = json.load(f)
            else:
                norm_dict = None

            results = get_losses(
                dataset=processed_dataset, 
                labels_dict=human_scores, 
                alignments_path=args.alignments, 
                granularity=granularity,
                pooling=pool,
                norm_dict=norm_dict,
                limit=20,
                )
            
            ppl_results = results["results"]
            nan_percent = (results["nan_count"] / len(ppl_results)) * 100

            with open(csv_path, "w") as f:
                fieldnames = ppl_results[0].keys()
                dict_writer = csv.DictWriter(f, fieldnames)
                dict_writer.writeheader()
                dict_writer.writerows(ppl_results)
            
            # correlate
            df = pd.DataFrame(ppl_results)
            df.dropna(axis=0, subset=df.columns.drop('ppl_loss_norm'), inplace=True)
            x = df["ppl_loss"]
            y = df["human_score"]
            pcc = scipy.stats.pearsonr(x, y)

            # auc
            y_score = df["ppl_loss"]
            y_true = df["auc_label"]
            if len(np.unique(y_true)) != 1:
                auc = roc_auc_score(y_true, y_score)
            else:
                auc = "n/a"

            if granularity == "phone" or granularity == "word":
                df_norm = pd.DataFrame(ppl_results)
                df_norm.dropna(axis=0, inplace=True)
                x_norm = df_norm["ppl_loss_norm"]
                y_norm = df_norm["human_score"]
                pcc_norm = scipy.stats.pearsonr(x_norm, y_norm)
                pcc_norm_stats = pcc_norm.statistic
                pcc_norm_pvalue = pcc_norm.pvalue

                y_score_norm = df_norm["ppl_loss_norm"]
                y_true_norm = df_norm["auc_label"]
                if len(np.unique(y_true_norm)) != 1:
                    auc_norm = roc_auc_score(y_true_norm, y_score_norm)
                else:
                    auc_norm = "n/a"
            
            else:
                pcc_norm_stats = "n/a"
                pcc_norm_pvalue = "n/a"
                auc_norm = "n/a"

            if granularity == "phone":
                per_phone_auc_result = per_phone_auc(ppl_results)
            else:
                per_phone_auc_result = {
                    "auc" : "n/a",
                    "auc_norm" : "n/a"
                }
                
            # Record in CSV
            #append_to_sheet([MODEL_TYPE, MODEL_NAME, granularity, pool, pcc.statistic, pcc.pvalue, pcc_norm_stats, pcc_norm_pvalue, auc, per_phone_auc_result['auc'], auc_norm, per_phone_auc_result['auc_norm'], f"{nan_percent:2f}" + "%", len(df)])
            
    
    print(f"Speaker count: {spk_count}")
    print(f"File count: {len(processed_dataset)}")

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    duration = time.time() - start_time
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")