import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
from wordfreq import word_frequency
import math
from sklearn.metrics import roc_auc_score
import scipy.stats
import json
import csv
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from collections import defaultdict
import re

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

PHONE_CLASSES = {
    "stop": set("P B T D K G".split()),
    "fricative": set("F V TH DH S Z SH ZH HH".split()),
    "affricate": set("CH JH".split()),
    "nasal": set("M N NG".split()),
    "liquid": set("L R".split()),
    "glide": set("W Y".split()),
    "high_vowel": set("IY IH UW UH".split()),
    "mid_vowel": set("EH ER AH EY OW".split()),
    "low_vowel": set("AE AA AO AW AY OY".split()),
}
PHONE_TO_CLASS = {p: cls for cls, phones in PHONE_CLASSES.items() for p in phones}

# utils
def is_overlapping(a_start, a_end, b_start, b_end):
    if (a_end >= b_start and a_start <= b_end):
        return True
    else:
        return False

def get_class(phone_label):
    base = re.sub(r"\d", "", phone_label)  # strip stress digits, e.g. "IY0" -> "IY"
    return PHONE_TO_CLASS.get(base, "unknown")

def strip_stress(phone_label):
    if phone_label[-1].isdigit():
        return phone_label[:-1]
    else:
        return phone_label

# main functions
def align_and_pool(losses, alignments_dict, human_scores, granularity, pooling, model_type):
    '''
    Returns a list of dicts containing loss scores aggregated by specified granularity and pooling
    '''
    data = losses.to_dict('records')

    results = []

    # sort by file
    tokens_by_file = defaultdict(list)
    for tok in data:
        tokens_by_file[tok["filename"]].append(tok)
    

    # aggregate by file
    for file, tokens in tokens_by_file.items():

        utterance_results = []
        speaker = tokens[0]['speaker']
        filename = tokens[0]['filename']
        human_annotation_obj = human_scores.get(filename)

        phone_scores = []
        word_scores = []

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
            phone_alignments = alignments_dict.get(filename)['phone_alignment']
            
            # align canonical phonemes and phone alignments
            phone_alignments_labels = [item['label'] for item in phone_alignments]
            phone_scores_labels = [item['phone'] for item in phone_scores]

            matcher = SequenceMatcher(None, phone_scores_labels, phone_alignments_labels)
            opcodes = matcher.get_opcodes()
            matched_alignments = []
            matched_scores = []
            for tag, a_idx1, a_idx2, b_idx1, b_idx2 in opcodes:
                if tag == "equal":
                    matched_scores.extend(phone_scores[a_idx1:a_idx2])
                    matched_alignments.extend(phone_alignments[b_idx1:b_idx2])
            phone_alignments = matched_alignments
            phone_scores = matched_scores
            alignments = phone_alignments
            human_score = phone_scores

        elif granularity == "word":
            alignments = alignments_dict.get(filename)['word_alignment']
            human_score = word_scores

        else:
            pass

        if granularity == "phone" or granularity == "word":  
            if model_type == "TASLM":
                for idx in range(0, len(tokens)):
                    utterance_results.append({
                        'speaker' : speaker,
                        'filename' : filename,
                        'label' : human_score[idx]['word'],
                        'human_score' : human_score[idx]['accuracy'],
                        'score' : tokens[idx]['ppl_loss']
                    })
            else:
                for idx, current_alignment in enumerate(alignments):     # type: ignore     
                    a_start = current_alignment["start"] # type: ignore
                    a_end = current_alignment["end"]     # type: ignore
                    cur_losses = []

                    for i, loss_item in enumerate(tokens):
                        # if i == len(tokens) - 1:
                        #     continue
                        token_loss = loss_item['ppl_loss']
                        t_start = loss_item['start']
                        t_end = loss_item['end']
                        if is_overlapping(a_start, a_end, t_start, t_end):
                            cur_losses.append(token_loss)

                    # pooling
                    loss_pooled = np.nan
                    
                    if pooling == "mean":
                        loss_pooled = np.mean(cur_losses) if len(cur_losses) > 0 else np.nan
                    elif pooling == "max":
                        loss_pooled = np.max(cur_losses) if len(cur_losses) > 0 else np.nan
                    elif pooling == "std":
                        loss_pooled = np.std(cur_losses) if len(cur_losses) > 1 else np.nan
                    else:
                        raise Exception("No pooling method specified.")
                    
                    if np.isnan(loss_pooled):
                        continue

                    utterance_results.append({
                        'speaker' : speaker,
                        'filename' : filename,
                        'label' : current_alignment['label'],
                        'human_score' : human_score[idx]['accuracy'],
                        'score' : loss_pooled
                    })
        elif granularity == "utterance":
            cur_losses = []
            for idx, loss_item in enumerate(tokens):
                # if idx == len(tokens) - 1:
                #     continue
                cur_losses.append(loss_item['ppl_loss'])

            loss_pooled = np.nan
                            
            if pooling == "mean":
                loss_pooled = np.mean(cur_losses) if len(cur_losses) > 0 else np.nan
            elif pooling == "max":
                loss_pooled = np.max(cur_losses) if len(cur_losses) > 0 else np.nan
            elif pooling == "std":
                loss_pooled = np.std(cur_losses) if len(cur_losses) > 1 else np.nan
            else:
                raise Exception("No pooling method specified.")
            
            if np.isnan(loss_pooled):
                continue

            utterance_results.append({
                'speaker' : speaker,
                'filename' : filename,
                'label' : human_annotation_obj['text'],
                'human_score' : human_annotation_obj['accuracy'],
                'score' : loss_pooled
            })

        results.append(utterance_results)

    return {
        'results' : results,
    }
def enrollment_calibration(scores, granularity):
    '''
    Normalizes loss scores against the mean and std of every other score under the same speaker and class.
    Returns a list of dicts containing calibrated loss scores.

    MUST be phone-level.
    '''
    MIN_SAMPLES = 8
    calibrated_results = []

    if granularity == "phone":
        by_speaker = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for utterance in scores:
            for tok in utterance:
                speaker = tok["speaker"]
                filename = tok["filename"]
                cls = get_class(tok["label"])
                by_speaker[speaker][filename][cls].append(tok)
    else:
        by_speaker = defaultdict(lambda: defaultdict(list))

        for utterance in scores:
            for tok in utterance:
                speaker = tok["speaker"]
                filename = tok["filename"]
                by_speaker[speaker][filename].append(tok)

    for file_scores in scores:
        speaker_dict = by_speaker.get(file_scores[0]['speaker'])

        merged_dict = {}
        all_speaker_scores = []  # flat fallback pool: every score, any class

        if granularity == "phone":
            for filename, file_classes in speaker_dict.items():
                if filename == file_scores[0]['filename']:
                    continue
                for cat_name, loss_items in file_classes.items():
                    merged_dict.setdefault(cat_name, [])
                    for item in loss_items:
                        merged_dict[cat_name].append(item['score'])
                        all_speaker_scores.append(item['score'])

        elif granularity == "word":
            for filename, losses in speaker_dict.items():
                if filename == file_scores[0]['filename']:
                                continue
                for item in losses:
                    all_speaker_scores.append(item['score'])

        for score_info in file_scores:

            if granularity == "phone":
                cls = get_class(score_info['label'])
                class_scores = merged_dict.get(cls, [])

                if len(class_scores) < MIN_SAMPLES:
                    # fallback: speaker's all-phone stats instead of this class
                    cls_mean = np.mean(all_speaker_scores)
                    cls_std = np.std(all_speaker_scores)
                else:
                    cls_mean = np.mean(class_scores)
                    cls_std = np.std(class_scores)
            else:
                cls_mean = np.mean(all_speaker_scores)
                cls_std = np.std(all_speaker_scores)

            calibrated_score = (score_info['score'] - cls_mean) / cls_std

            if cls_std == 0 or np.isnan(calibrated_score): # guard against 0 std
                continue

            score_info.update({'score': calibrated_score})
            calibrated_results.append(score_info)

    return calibrated_results

def normalize(scores, granularity, norm_dict):
    '''
    Returns a list of dicts containing frequency-normalized loss scores.
    '''
    normalized_results = []

    for score_item in scores:
        score = score_item['score']
        if granularity == "phone":
            phone_label = strip_stress(score_item['label']) # type: ignore
            p_mean = norm_dict[phone_label]['mean'] # type: ignore
            p_std = norm_dict[phone_label]['std'] # type: ignore
            normalized_score = ((score - p_mean) / p_std) if p_std > 0 else np.nan
            
        elif granularity == "word":
            word = score_item['label']
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
                normalized_score = (score - w_mean) / w_std
            else:
                normalized_score = np.nan
        else:
            raise Exception("Invalid granularity")

        score_item.update({'score' : normalized_score})
        normalized_results.append(score_item)

    return normalized_results
    
def calculate_pcc(data):
    '''
    Returns a PCC value
    '''
    df = pd.DataFrame(data)
    df.dropna(axis=0, inplace=True)
    x = df["score"]
    y = df["human_score"]
    pcc = scipy.stats.pearsonr(x, y)

    return pcc.statistic, pcc.pvalue

def calculate_auc(data, granularity, threshold):
    '''
    Returns an AUC value
    '''

    if granularity == 'phone':
        thres = 2 * threshold
    else:
        thres = 10 * threshold

    df = pd.DataFrame(data)
    df.dropna(axis=0, inplace=True) 
    y_score = -df["score"]                           # higher score = higher loss = worse pronunciation, so reversed
    y_true = (df["human_score"] > thres).astype(int) # higher score = better pronunciation

    if len(np.unique(y_true)) != 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = "n/a"

    return auc

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
    service_account_file,
    spreadsheet_name="ICASSP 2026 Experiment Results",
    worksheet_name="run_3",
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

# ======== main =========

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--labels_dir", type=str)
    parser.add_argument("--alignments", type=str)
    parser.add_argument("--evaluation_file", type=str)
    args = parser.parse_args()

    CSV_PATH = args.evaluation_file
    model_metadata = os.path.basename(CSV_PATH).split('_')
    MODEL_TYPE = model_metadata[0]
    MODEL_NAME = model_metadata[1]
    NORM_DICT_DIR = f"{args.root_dir}/src/metrics/result_dicts"
    SERVICE_ACCOUNT = f"{args.root_dir}/src/service_account.json"

    losses_df = pd.read_csv(CSV_PATH, dtype={'token_id': str, 'filename': str, 'speaker': str})

    # prepare alignments
    with open(args.alignments, 'r') as f:
        alignments = json.load(f)

    # human annotations
    score_labels = args.labels_dir
    human_scores = parse_human_annotations(score_labels)

    # aggregate and normalize
    for granularity in ['phone', 'word', 'utterance']:
        for pooling in ['mean', 'max', 'std']:
            for calibration in [True, False]:
                for norm in [True, False]:

                    if calibration and norm:
                        continue
                    elif MODEL_TYPE == "TASLM" and (granularity == "phone"):
                        continue
                    elif MODEL_TYPE == "TASLM" and granularity == "word" and (pooling == "max" or pooling == "std"):
                        continue
                    elif granularity == 'utterance' and (calibration or norm):
                        continue
                        continue

                    print(f"GRANULARITY: {granularity}")

                    aggregated_losses = align_and_pool( # num of losses will equal num of alignments
                        losses=losses_df, 
                        alignments_dict=alignments, 
                        human_scores=human_scores,
                        granularity=granularity, 
                        pooling=pooling,
                        model_type=MODEL_TYPE)

                    #print(f"AGGREGATION: {aggregated_losses['results'][0]}")

                    if calibration == True and granularity != "utterance":
                        calibration_losses = enrollment_calibration(scores=aggregated_losses['results'], granularity=granularity)
                    else:
                        temp = []
                        for obj in aggregated_losses['results']:
                            temp += obj
                        calibration_losses = temp

                    #print(f"CALIBRATION: {calibration_losses[0]}")

                    if MODEL_TYPE == "TASLM":
                        pooling = "none"

                    if norm == True and granularity != 'utterance':
                        norm_dict_path = f"{NORM_DICT_DIR}/{MODEL_NAME}_{granularity}_{pooling}_norm.json"
                        with open(norm_dict_path, "r") as f:
                            norm_dict = json.load(f)
                        norm_losses = normalize(scores=calibration_losses, granularity=granularity, norm_dict=norm_dict)
                    else:
                        norm_losses = calibration_losses

                    #print(f"NORM: {norm_losses[0]}")

                    final_data = norm_losses

                    # evaluation metrics
                    pcc_value, pvalue = calculate_pcc(data=final_data)
                    auc_value = calculate_auc(data=final_data, granularity=granularity, threshold=0.5)
                    auc_value_2 = calculate_auc(data=final_data,  granularity=granularity, threshold=0.25)

                    # record results
                    append_to_sheet([MODEL_TYPE, MODEL_NAME, granularity, pooling, calibration, norm, pcc_value, pvalue, auc_value, auc_value_2, len(final_data)], SERVICE_ACCOUNT)



                    

    