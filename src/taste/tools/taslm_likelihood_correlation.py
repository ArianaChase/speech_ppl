import os
import json
import argparse
import time
from datetime import datetime
import pandas as pd
import scipy.stats
import gspread
from google.oauth2.service_account import Credentials

# --- Google Sheets Authentication Setup ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Update this path if necessary
service_account_file = "/home/u5504709/new_work/speech_ppl/src/service_account.json"

creds = Credentials.from_service_account_file(
    service_account_file,
    scopes=SCOPES
)
client = gspread.authorize(creds)

# Start execution timer
start_time = time.time()

def append_to_sheet(
    row_data,
    spreadsheet_name="Pronunciation Evaluation Results",
    worksheet_name="main",
):
    """Opens the target Google Sheet and appends the provided row data."""
    spreadsheet = client.open(spreadsheet_name)
    worksheet = spreadsheet.worksheet(worksheet_name)
    worksheet.append_row(row_data)
    print(f"Spreadsheet updated successfully: {row_data[0]} | {row_data[4]}")

def load_and_merge_data(csv_path, json_path):
    """Loads CSV and JSON, merging Age and Gender into the dataframe."""
    # Force 'Audio filename' to be read as string to preserve leading zeros (e.g. 000010011)
    df = pd.read_csv(csv_path, dtype={'Audio filename': str})
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    # Map the age and gender based on 'Audio filename' matching the JSON keys
    # .get() prevents KeyError if an audio file is missing from the JSON
    df['age'] = df['Audio filename'].apply(lambda x: json_data.get(x, {}).get('age', None))
    df['gender'] = df['Audio filename'].apply(lambda x: json_data.get(x, {}).get('gender', None))
    
    # Ensure age is numeric so we can mathematically evaluate >= 18 and < 18
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    return df

def calc_correlation(df, metric_col, human_col):
    """Calculates Pearson correlation between a metric and human dimension."""
    x = pd.to_numeric(df[metric_col], errors='coerce')
    y = pd.to_numeric(df[human_col], errors='coerce')
    
    # Create a boolean mask to keep only rows where both x and y have valid data
    mask = x.notna() & y.notna()
    x = x[mask].to_numpy(dtype=float)
    y = y[mask].to_numpy(dtype=float)
    
    # Return None if we don't have enough valid data points to compute correlation
    if len(x) < 2:
        return None

    # returns (statistic, pvalue)
    return scipy.stats.pearsonr(x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Demographics Correlations for TASTE Likelihood")
    
    # Arguments for tracking in gspread
    parser.add_argument("--name", type=str, default="Taste_Likelihood_Demographics")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--model", type=str, default="TASTE")
    parser.add_argument("--category", type=str, default="Demographics")
    
    # Data arguments
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the TASTE likelihood CSV")
    parser.add_argument("--json_file", type=str, required=True, help="Path to JSON file with annotations")
    
    args = parser.parse_args()

    # 1. Load and merge the demographic data with the likelihood metrics
    df = load_and_merge_data(args.csv_file, args.json_file)
    print(df.head())

    # 2. Define the requested subsets using Pandas filters
    categories = {
        "genderedm": df[df['gender'] == 'm'],
        "genderedf": df[df['gender'] == 'f'],
        "aged18": df[df['age'] >= 18],
        "agednot18": df[df['age'] < 18]
    }

    # Only 1 metric for this file compared to the TASLM file
    metrics = ["Raw Mean of Per Token Losses"]
    
    # Capture final execution time and calculate duration
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    duration = time.time() - start_time

    print(f"Date and time at completion: {finish_time}")
    print(f"Executing append operations...")

    # 3. Iterate through subgroups, compute scores, and append to gspread
    for cat_name, cat_df in categories.items():
        
        # Determine number of unique speakers and valid samples in this specific slice
        speaker_count = cat_df['Speaker'].nunique() if 'Speaker' in cat_df.columns else 0
        print(f"{cat_name} Speaker count: {speaker_count}")
        sample_count = len(cat_df)
        print(f"{cat_name} Sample count: {sample_count}")

        
        if sample_count < 2:
            print(f"Skipping {cat_name}: Not enough samples ({sample_count}).")
            continue

        for metric in metrics:
            
            MODEL_NAME = "TASLM"
            # Creates an ID like 'taste_likelihood_genderedm_Raw Mean of Per Token Losses_1'
            MODEL_ID = f"{args.name}_{cat_name}_{str(args.index)}"

            # Calculate Pearson Correlations
            accuracy_result = calc_correlation(cat_df, metric, "Human Annotation (Accuracy)")
            fluency_result = calc_correlation(cat_df, metric, "Human Annotation (Fluency)")
            prosody_result = calc_correlation(cat_df, metric, "Human Annotation (Prosody)")
            completeness_result = calc_correlation(cat_df, metric, "Human Annotation (Completeness)")

            # Append to sheet (only if calculations were successfully generated)
            #if accuracy_result and fluency_result and prosody_result and completeness_result:
                #append_to_sheet(["Accuracy-" + args.category, MODEL_ID, finish_time, args.model, MODEL_NAME, speaker_count, sample_count, accuracy_result.statistic, accuracy_result.pvalue, duration])
                #append_to_sheet(["Fluency-" + args.category, MODEL_ID, finish_time, args.model, MODEL_NAME, speaker_count, sample_count, fluency_result.statistic, fluency_result.pvalue, duration])
                #append_to_sheet(["Prosody-" + args.category, MODEL_ID, finish_time, args.model, MODEL_NAME, speaker_count, sample_count, prosody_result.statistic, prosody_result.pvalue, duration])
                #append_to_sheet(["Completeness-" + args.category, MODEL_ID, finish_time, args.model, MODEL_NAME, speaker_count, sample_count, completeness_result.statistic, completeness_result.pvalue, duration])

    print(f"\nProgram '{args.name}' finished executing in {time.time() - start_time:.2f} seconds.")