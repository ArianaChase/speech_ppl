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
    for file, tokens in tokens_by_file.items():

        utterance_results = []
        filename = tokens[0]['filename']
        current_alignment = alignments_dict.get(filename)

        a_start = current_alignment["start"] # type: ignore
        a_end = current_alignment["end"]     # type: ignore
        cur_losses = []

        for i, loss_item in enumerate(tokens):
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

    if args.set_name == "setC":
        real_real = data['real_real']
        real_foil = data['real_foil']
        real_ref = data['real_ref']

        final_data = []
        for index, file in metadata.iterrows():
            filename = file['utt_id']

            if real_real.get(filename) == None or real_foil.get(filename) == None or real_ref.get(filename) == None:
                continue

            final_data.append({
                'filename' : filename,
                'real_real' : real_real.get(filename)['score'],
                'real_foil' : real_foil.get(filename)['score'],
                'real_ref' : real_ref.get(filename)['score'],
            })

    else:
        clean_clean = data['clean_clean']
        clean_sub = data['clean_sub']
        sub_sub = data['sub_sub']
        sub_clean = data['sub_clean']

        final_data = []

        for index, file in metadata.iterrows():
            filename = file['stim_id']

            if clean_clean.get(filename) == None or clean_sub.get(filename) == None or sub_sub.get(filename) == None or sub_clean.get(filename) == None:
                continue

            final_data.append({
                'filename' : filename,
                'clean_clean' : clean_clean.get(filename)['score'],
                'clean_sub' : clean_sub.get(filename)['score'],
                'sub_clean' : sub_clean.get(filename)['score'],
                'sub_sub' : sub_sub.get(filename)['score']
            })

    return final_data

def parse_delta(data):
    df = pd.DataFrame(data)
    print(df.columns)

    if args.set_name == "setC":
        df['foil_delta'] = df['real_foil'] - df['real_real'] # if final > initial (positive) = correct
        df['ref_delta'] = df['real_ref'] - df['real_real']

        foil_percent = (df["foil_delta"] > 0).mean() * 100
        ref_percent = (df["ref_delta"] > 0).mean() * 100
        size = len(df)

        return {
            'foil_percent': foil_percent, 
            'ref_percent' : ref_percent, 
            'size' : size
        }
    else:
        df['clean_delta'] = df['clean_sub'] - df['clean_clean'] # final: sub, initial: clean -> if final > initial (positive) = correct
        df['sub_delta'] = df['sub_sub'] - df['sub_clean'] # final: sub, initial: clean -> if final < initial (negative) = correct
        df['total_delta'] = df['sub_delta'] - df['clean_delta'] # if clean delta is pos, sub delta is neg = negative = correct
        df.dropna(inplace=True)

        clean_percent = (df["clean_delta"] > 0).mean() * 100 # percentage of positive values
        sub_percent = (df["sub_delta"] < 0).mean() * 100
        total_percent = (df['total_delta'] < 0).mean() * 100 
        size = len(df)

        return {
            'clean_percent' : clean_percent, 
            'sub_percent' : sub_percent, 
            'total_percent' : total_percent, 
            'size' : size
        }

def parse_alignments(metadata, audio_version):

    alignments = {}

    for index, file in metadata.iterrows():
        if audio_version == 'clean':
            alignments[file['stim_id']] = {
                'start' : file['clean_word_start'],
                'end' : file['clean_word_end'],
                'label' : file['original_word']
            }
        elif audio_version == 'sub':
            alignments[file['stim_id']] = {
                'start' : file['sub_word_start'],
                'end' : file['sub_word_end'],
                'label' : file['original_word']
            }
        elif audio_version == 'dist':
            alignments[file['stim_id']] = {
                'start' : file['dist_word_start'],
                'end' : file['dist_word_end'],
                'label' : file['original_word']
            }
        elif audio_version == 'real':
            alignments[file['utt_id']] = {
                'start' : file['real_target_start'],
                'end' : file['real_target_end'],
                'label' : file['target_word']
            }
        elif audio_version == 'foil':
            alignments[file['utt_id']] = {
                'start' : file['real_foil_start'],
                'end' : file['real_foil_end'],
                'label' : file['target_word']
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
    worksheet_name="test_B",
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

    metadata_df = pd.read_csv(METADATA, dtype={0:str, 1:str})

    if args.set_name == "setC":
        data_store = {
            "real_real" : {},
            "real_foil" : {},
            "real_ref" : {}
        }

        for audio_version in ['real']:
            for text_version in ['real', 'foil', 'ref']:
                alignments = parse_alignments(metadata_df, audio_version)
                csv_path = f"{CSV_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{audio_version}_{text_version}_{args.set_name}_per_token_losses.csv"
                losses_df = pd.read_csv(csv_path, dtype={'token_id': str, 'filename': str, 'speaker': str})

                target_word_data = align_and_pool( # num of losses will equal num of alignments
                    losses=losses_df, 
                    alignments_dict=alignments,
                    )['results']

                data_store[f"{audio_version}_{text_version}"] = target_word_data

        final_data = parse_final_data(data_store, metadata_df)
        results = parse_delta(final_data)

        # record results
        append_to_sheet([MODEL_TYPE, MODEL_NAME, results['foil_percent'], results['ref_percent'], "", results['size'], "B"], SERVICE_ACCOUNT)


    else:
        data_store = {
            "clean_clean" : {},
            "clean_sub" : {},
            "sub_sub" : {},
            "sub_clean" : {},
        }
        
        for audio_version in ['clean', 'sub']:
            for text_version in ['clean', 'sub']:
                alignments = parse_alignments(metadata_df, audio_version)
                csv_path = f"{CSV_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{audio_version}_{text_version}_{args.set_name}_per_token_losses.csv"
                losses_df = pd.read_csv(csv_path, dtype={'token_id': str, 'filename': str, 'speaker': str})

                target_word_data = align_and_pool( # num of losses will equal num of alignments
                    losses=losses_df, 
                    alignments_dict=alignments,
                    )['results']

                data_store[f"{audio_version}_{text_version}"] = target_word_data

        final_data = parse_final_data(data_store, metadata_df)
        results = parse_delta(final_data)

        # record results
        append_to_sheet([MODEL_TYPE, MODEL_NAME, results['clean_percent'], results['sub_percent'], results['total_percent'], results['size'], "B"], SERVICE_ACCOUNT)



        

