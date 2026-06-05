import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import argparse
import json
from operator import itemgetter
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import time

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

service_account_file="/home/u5504709/new_work/speech_ppl/src/service_account.json"

# Authenticate
creds = Credentials.from_service_account_file(
    service_account_file,
    scopes=SCOPES
)

client = gspread.authorize(creds)

start_time = time.time()

def parse_human_annotations(filename):
    human_scores = []
    unique_speakers = set()

    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            print(audio_file)
            value = data[audio_file]
            unique_speakers.add(audio_file[1:5])

            #if value["gender"] == "m":
            human_scores.append({
                "filename" : audio_file,
                "accuracy" : value["accuracy"],
                "fluency" : value["fluency"],
                "prosodic" : value["prosodic"],
                "completeness" : value["completeness"]
            })

    return human_scores, unique_speakers


def calc_correlation(output_df, dim):
    x = output_df["loss"]

    if dim == "accuracy":
        y = output_df["Human Annotation (Accuracy)"]
    elif dim == "fluency":
        y = output_df["Human Annotation (Fluency)"]
    elif dim == "prosodic":
        y = output_df["Human Annotation (Prosody)"]
    elif dim == "completeness":
        y = output_df["Human Annotation (Completeness)"]
    else:
        print("Invalid dimension")
        return

    # force numeric + clean
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = x.notna() & y.notna()
    x = x[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=float)

    result = scipy.stats.pearsonr(x, y)

    print(f"=== Correlation for dimension {dim} ===")
    print("Correlation x len:", len(x))
    print("Correlation y len:", len(y))
    print("Correlation value is:", result, "\n")

    return result

def losses_to_dict(loss_file, human_scores):
    # Create a lookup dictionary for O(1) matching instead of nested loops
    human_dict = {obj["filename"]: obj for obj in human_scores}
    loss_objects = []

    with open(loss_file, mode='r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            # Split line by whitespace: expected format "test_0_000030012 -0.30289340019226074"
            parts = line.split()
            if len(parts) != 2:
                continue
                
            raw_id, loss_val = parts[0], parts[1]
            
            # Extract the 9-digit filename from the raw_id (e.g., gets "000030012")
            filename = raw_id.split('_')[-1]
            spk = filename[1:5]

            if filename in human_dict and spk != "1076":
                human_annotation_obj = human_dict[filename]

                loss_objects.append({
                    "filename" : filename,
                    "loss" : loss_val,
                    "Human Annotation (Accuracy)" : human_annotation_obj["accuracy"],
                    "Human Annotation (Fluency)" : human_annotation_obj["fluency"],
                    "Human Annotation (Prosody)" : human_annotation_obj["prosodic"],
                    "Human Annotation (Completeness)" : human_annotation_obj["completeness"],
                })
    
    print("Length of list: ", len(loss_objects))
    print("Length of labels: ", len(human_scores))
    if len(loss_objects) > 0:
        print(loss_objects[0])

    return loss_objects

def append_to_sheet(
    row_data,
    spreadsheet_name="Pronunciation Evaluation Results",
    worksheet_name="main",
):
    # Open sheet
    spreadsheet = client.open(spreadsheet_name)
    worksheet = spreadsheet.worksheet(worksheet_name)

    # Append row
    worksheet.append_row(row_data)

    print("Spreadsheet updated successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--name", type=str, default="experiment")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default="Target_Model")
    parser.add_argument("--category", type=str, default="None")

    parser.add_argument("--loss_file", help="Path to losses", default=None)
    parser.add_argument("--labels_dir", help="Path to labels", default=None)

    args = parser.parse_args()

    loss_file = args.loss_file

    # get labels to compare to
    score_labels = args.labels_dir
    human_scores, unique_speakers = parse_human_annotations(score_labels)
    human_scores = sorted(human_scores, key=itemgetter("filename"))

    loss_dict = losses_to_dict(loss_file, human_scores)
    output_csv_df = pd.DataFrame.from_dict(loss_dict)

    print(output_csv_df.head())

    speaker_count = len(unique_speakers) - 1
    sample_count = len(loss_dict)

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    duration = time.time() - start_time
    print(f"Program '{args.name}' finished executing in {duration} seconds.")

    # == Calculate Single Loss Correlations ==
    MODEL_NAME = "Flow-SLM-1bext_semantic" # Set dynamically based on args

    accuracy_result = calc_correlation(output_csv_df, "accuracy")
    fluency_result = calc_correlation(output_csv_df, "fluency")
    prosody_result = calc_correlation(output_csv_df, "prosodic")
    completeness_result = calc_correlation(output_csv_df, "completeness")

    # Format the name identifier securely
    run_identifier = f"{args.name}_{args.index}"
    category_str = str(args.category)

    # Append to Sheets
    append_to_sheet([f"Accuracy-{category_str}", run_identifier, finish_time, args.model, MODEL_NAME, speaker_count, sample_count, accuracy_result.statistic, accuracy_result.pvalue, duration])
    append_to_sheet([f"Fluency-{category_str}", run_identifier, finish_time, args.model, MODEL_NAME, speaker_count, sample_count, fluency_result.statistic, fluency_result.pvalue, duration])
    append_to_sheet([f"Prosody-{category_str}", run_identifier, finish_time, args.model, MODEL_NAME, speaker_count, sample_count, prosody_result.statistic, prosody_result.pvalue, duration])
    append_to_sheet([f"Completeness-{category_str}", run_identifier, finish_time, args.model, MODEL_NAME, speaker_count, sample_count, completeness_result.statistic, completeness_result.pvalue, duration])