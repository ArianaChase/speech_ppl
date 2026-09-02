import os
import json
import time
import numpy as np
import scipy.stats
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# --- CONFIGURATION ---
ROOT_DIR = "/home/ubuntu/speech_ppl/work/speechocean"
LABELS_PATH = "/home/ubuntu/speech_ppl/src/scores_enhanced.json"
SERVICE_ACCOUNT = "/home/ubuntu/speech_ppl/src/service_account.json"
SPREADSHEET_NAME = "Pronunciation Evaluation Results"
WORKSHEET_NAME = "revised_flow"

# Model name mappings
MODELS = {
    "1b_extend": {"short": "1bext", "full": "Flow-SLM-1Bext"},
    "1b": {"short": "1b", "full": "Flow-SLM-1B"},
    "270m": {"short": "270m", "full": "Flow-SLM-270M"}
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT,
    scopes=SCOPES
)

client = gspread.authorize(creds)

# The 5 filters you requested
FILTERS = {
    "all": lambda x: True,
    "genderedf": lambda x: x.get("gender") == "f",
    "genderedm": lambda x: x.get("gender") == "m",
    "aged18": lambda x: int(x.get("age", 0)) >= 18,
    "agednot18": lambda x: int(x.get("age", 0)) < 18
}

def load_human_scores(json_path):
    with open(json_path) as f:
        return json.load(f)

def safe_pearson(x, y):
    """Calculates PCC, returning 0 if arrays are invalid (e.g., zero variance)."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    res = scipy.stats.pearsonr(x, y)
    return res.statistic, res.pvalue

def read_loss_file(filepath):
    """Extracts dict of {filename_id: loss_value} from text format."""
    data = {}
    if not os.path.exists(filepath): return data
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # Extracts '000030012' from 'test_0_000030012'
                file_id = parts[0].split('_')[-1]
                data[file_id] = float(parts[1])
    return data

def process_batch():
    start_time = time.time()
    human_scores = load_human_scores(LABELS_PATH)
    all_rows = []

    # Traverse Model -> Batch -> Loss Type
    for model_dir, model_info in MODELS.items():
        for batch_idx in range(1, 11):
            batch_path = os.path.join(ROOT_DIR, model_dir, str(batch_idx))
            if not os.path.exists(batch_path): continue

            # Combine test and train sets for this batch
            for loss_type, filename in [("acoustic", "loss.txt"), ("semantic", "token_loss.txt")]:
                combined_losses = {}
                combined_losses.update(read_loss_file(os.path.join(batch_path, "test", filename)))
                combined_losses.update(read_loss_file(os.path.join(batch_path, "train", filename)))

                if not combined_losses: continue

                # Evaluate all 5 filters
                for filter_name, filter_func in FILTERS.items():
                    # Align scores based on the filter
                    x_losses, y_acc, y_flu, y_pro, y_com = [], [], [], [], []
                    unique_speakers = set()

                    for file_id, loss_val in combined_losses.items():
                        # Determine human dict key (usually matches exactly, but check if it's nested)
                        h_data = human_scores.get(file_id) or human_scores.get(f"/{file_id}.wav")
                        if h_data and filter_func(h_data):
                            x_losses.append(loss_val)
                            y_acc.append(h_data["accuracy"])
                            y_flu.append(h_data["fluency"])
                            y_pro.append(h_data["prosodic"])
                            y_com.append(h_data["completeness"])
                            unique_speakers.add(file_id[1:5]) # standard speaker ID extraction

                    sample_count = len(x_losses)
                    speaker_count = len(unique_speakers)
                    if sample_count < 2: continue

                    # Calculate Correlations
                    acc_r, acc_p = safe_pearson(x_losses, y_acc)
                    flu_r, flu_p = safe_pearson(x_losses, y_flu)
                    pro_r, pro_p = safe_pearson(x_losses, y_pro)
                    com_r, com_p = safe_pearson(x_losses, y_com)

                    # Create identifiers (e.g., flow1bext_likelihood_genderedf_7)
                    run_id = f"flow{model_info['short']}_likelihood_{filter_name}_{batch_idx}"
                    if filter_name == "all": run_id = f"flow{model_info['short']}_likelihood_{batch_idx}"
                    
                    finish_time = datetime.now().strftime("%m-%d-%Y %H:%M")
                    m_name = f"{model_info['full']}_{loss_type}"
                    duration = round(time.time() - start_time, 2)

                    # Build row payloads
                    base_row = [run_id, finish_time, "Flow-SLM", m_name, speaker_count, sample_count]
                    all_rows.append([f"Accuracy-Likelihood_Correlation"] + base_row + [acc_r, acc_p, duration])
                    all_rows.append([f"Fluency-Likelihood_Correlation"] + base_row + [flu_r, flu_p, duration])
                    all_rows.append([f"Prosody-Likelihood_Correlation"] + base_row + [pro_r, pro_p, duration])
                    all_rows.append([f"Completeness-Likelihood_Correlation"] + base_row + [com_r, com_p, duration])

    # Batch push to Google Sheets
    print(f"Computed {len(all_rows)} metric rows. Pushing to Google Sheets...")
    worksheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
    worksheet.append_rows(all_rows)
    print(f"Successfully appended all rows in {round(time.time() - start_time, 2)} seconds.")

if __name__ == "__main__":
    process_batch()