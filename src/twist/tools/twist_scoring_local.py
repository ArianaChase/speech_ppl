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
    norm            : normalization
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

                            error_log.append(f"{word} has entered bucket {bucket}, freq is {neg_log_freq}, range is {s}")

                    if w_mean != None and w_std != None and w_std > 0:
                        loss_pooled_norm = (loss_pooled - w_mean) / w_std
                    else:
                        loss_pooled_norm = np.nan

                utterance_ppl_info.append({
                    "speaker" : speaker,
                    "filename" : filename,
                    "label" : current_alignment['label'],
                    "auc_label" : 1 if human_scores[i]['accuracy'] > auc_threshold else 0,
                    "ppl_loss" : -loss_pooled,
                    "ppl_loss_norm" : -loss_pooled_norm,
                    "human_score": human_scores[i]['accuracy']
                })

                if len(alignments) != len(utterance_ppl_info):
                    error_log.append(f"[LOSS/ALIGNMENT MISMATCH] Phone segmentation does not match at file {filename}. {len(phone_alignments)} phone alignments but {len(ppl_info)} losses") 
        
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
                "auc_label" : 1 if human_scores > auc_threshold else 0,
                "ppl_loss" : -loss_pooled,
                "ppl_loss_norm" : np.nan,
                "human_score": human_scores
            })
    
        ppl_info += utterance_ppl_info
        file_count += 1

    with open(f"{args.root_dir}/src/gslm/tools/error_log", "a") as f:
        for i in error_log:
            f.write(i)
            f.write("\n")

    return {
        "results" : ppl_info,
        "nan_count" : nan_count
    }

def append_to_sheet(
    row_data,
    service_account_file,
    spreadsheet_name="ICASSP 2026 Experiment Results",
    worksheet_name="main",
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

    NORM_DICT_DIR = f"{args.root_dir}/src/gslm/tools/result_dicts"
    SERVICE_ACCOUNT = f"{args.root_dir}/src/service_account.json"
    OUTPUT_DIR = args.output_dir

    for granularity in ["word", "utterance"]:
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
                limit=10,
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
            append_to_sheet([MODEL_TYPE, MODEL_NAME, granularity, pool, pcc.statistic, pcc.pvalue, pcc_norm_stats, pcc_norm_pvalue, auc, per_phone_auc_result['auc'], auc_norm, per_phone_auc_result['auc_norm'], f"{nan_percent:2f}" + "%", len(df)], SERVICE_ACCOUNT)
    print(f"Speaker count: {spk_count}")
    print(f"File count: {len(processed_dataset)}")

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    duration = time.time() - start_time
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")

