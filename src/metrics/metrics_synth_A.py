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


# utils
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

# main functions
def align_and_pool(losses, alignments_dict):
    '''
    Returns a list of dicts containing loss scores aggregated by specified granularity and pooling
    '''
    data = losses.to_dict('records')

    results = {}

    # sort by file
    tokens_by_file = defaultdict(list)
    for tok in data:
        tokens_by_file[tok["filename"]].append(tok)
    

    # aggregate by file
    for file, tokens in tokens_by_file.items(): # tokens in every file

        utterance_results = []
        filename = tokens[0]['filename']
        current_alignment = alignments_dict.get(filename)
        idx = current_alignment['idx']

        if current_alignment == None:
            #print(f"{filename} HAS NO ALIGNMENT")
            continue

        a_start = current_alignment["start"] # type: ignore
        a_end = current_alignment["end"]     # type: ignore
        cur_losses = []

        if MODEL_NAME == "TASLM":
            cur_losses = tokens[idx]['ppl_loss']
        else:
            for i, loss_item in enumerate(tokens): # extracting only the loss within the alignments
                token_loss = loss_item['ppl_loss']
                t_start = loss_item['start']
                t_end = loss_item['end']
                if is_overlapping(a_start, a_end, t_start, t_end):
                    cur_losses.append(token_loss)

        target_loss = np.mean(cur_losses)

        results[filename] = {
            'label' : current_alignment['label'],
            'score' : target_loss
        }

    return {
        'results' : results,
}

def parse_final_data(data, metadata):
    clean_dict = data['clean']
    sub_dict = data['sub']
    dist_dict = data['dist']

    final_data = []

    for index, file in metadata.iterrows():
        filename = file['stim_id']

        if clean_dict.get(filename) == None or sub_dict.get(filename) == None or dist_dict.get(filename) == None:
            continue

        final_data.append({
            'filename' : filename,
            'clean' : clean_dict.get(filename)['score'],
            'sub' : sub_dict.get(filename)['score'],
            'dist' : dist_dict.get(filename)['score']
        })

    return final_data

def parse_delta(data):
    df = pd.DataFrame(data)
    print(df.columns)
    df['clean_sub_delta'] = df['sub'] - df['clean'] # final: sub, initial: clean -> if final > initial (positive) = correct
    df['clean_dist_delta'] = df['dist'] - df['clean']

    df.dropna(inplace=True)

    sub_percent = (df["clean_sub_delta"] > 0).mean() * 100 # percentage of positive values
    dist_percent = (df["clean_dist_delta"] > 0).mean() * 100

    return sub_percent, dist_percent, len(df)


def parse_alignments(metadata, audio_version):

    alignments = {}

    for index, file in metadata.iterrows():
        if audio_version == 'clean':
            alignments[file['stim_id']] = {
                'start' : file['clean_word_start'],
                'end' : file['clean_word_end'],
                'label' : file['original_word'],
                'idx' : file['word_index']
            }
        elif audio_version == 'sub':
            alignments[file['stim_id']] = {
                'start' : file['sub_word_start'],
                'end' : file['sub_word_end'],
                'label' : file['original_word'],
                'idx' : file['word_index']

            }
        elif audio_version == 'dist':
            alignments[file['stim_id']] = {
                'start' : file['dist_word_start'],
                'end' : file['dist_word_end'],
                'label' : file['original_word'],
                'idx' : file['word_index']

            }

    return alignments

def calculate_auc(data, granularity, threshold):
    '''
    Returns an AUC value
    '''

    df = pd.DataFrame(data)
    df.dropna(axis=0, inplace=True) 
    y_score = -df["score"]                           # higher score = higher loss = worse pronunciation, so reversed

    if len(np.unique(y_true)) != 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = "n/a"

    return auc

def append_to_sheet(
    row_data,
    service_account_file,
    spreadsheet_name="ICASSP 2026 Experiment Results",
    worksheet_name="test_A",
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
    parser.add_argument("--metadata", type=str)
    parser.add_argument("--evaluation_file")
    parser.add_argument("--evaluation_dir")
    parser.add_argument("--set_name")
    args = parser.parse_args()

    CSV_DIR = args.evaluation_dir
    CSV_PATH = os.listdir(CSV_DIR)[0]
    print(CSV_PATH)
    model_metadata = os.path.basename(CSV_PATH).split('_')
    MODEL_TYPE = model_metadata[0]
    MODEL_NAME = model_metadata[1]
    SERVICE_ACCOUNT = f"{args.root_dir}/src/service_account.json"
    METADATA = args.metadata

    metadata_df = pd.read_csv(METADATA)

    data_store = {
        "clean" : [],
        "sub" : [],
        "dist" : []
    }
    
    for audio_version in ['clean', 'sub', 'dist']:
        alignments = parse_alignments(metadata_df, audio_version)
        print(alignments)
        if MODEL_NAME == "TASLM":
            csv_path = f"{CSV_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{audio_version}_{args.set_name}_per_token_losses.csv"
        else:
            csv_path = f"{CSV_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{audio_version}_{args.set_name}_per_token_losses.csv"

        print(f"Current file: {csv_path}")

        losses_df = pd.read_csv(csv_path, dtype={'token_id': str, 'filename': str, 'speaker': str})

        print(losses_df.head())

        target_word_data = align_and_pool( # num of losses will equal num of alignments
            losses=losses_df, 
            alignments_dict=alignments,
            )['results']

        data_store[audio_version] = target_word_data

    final_data = parse_final_data(data_store, metadata_df)
    sub_percent, dist_percent, size = parse_delta(final_data)

    print(f"{sub_percent}\n{dist_percent}\n{size}")

    # record results
    append_to_sheet([MODEL_TYPE, MODEL_NAME, sub_percent, dist_percent, size, "A", args.set_name], SERVICE_ACCOUNT)



    

