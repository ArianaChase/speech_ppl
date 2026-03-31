import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import argparse
import json

def calc_correlation(file, labels, mode="combined"):
    loss_df = pd.read_csv(file)
    y = labels

    losses = []

    if (mode == "utt_token"):
        losses = loss_df["utt_token_loss"].astype(float).values
    elif (mode == "utt_flow"):
        losses = loss_df["utt_flow_loss"].astype(float).values
    elif (mode == "combined"):
        losses = loss_df["weighted_combined_score"].astype(float).values
    else:
        print("Mode not valid.")

    print(loss_df.head())
    x = -np.array(losses)

    correlation_value = scipy.stats.pearsonr(x, y)

    return len(x), len(y), correlation_value



def parse_accuracy_scores(filename):
    accuracy_scores = {}
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            accuracy_scores[os.path.basename(audio_file)] = value["accuracy"]

    return accuracy_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--loss_file", help="Path to losses", default=None)
    parser.add_argument("--labels_dir", help="Path to labels", default=None)

    args = parser.parse_args()

    loss_file = args.loss_file

    # get labels to compare to
    score_labels = args.labels_dir
    accuracy_scores = parse_accuracy_scores(score_labels)
    accuracy_scores = dict(sorted(accuracy_scores.items()))
    y = []
    for key, value in accuracy_scores.items():
        if key[1:5] != "1076":
            y.append(value)

    utt_token_correlation = calc_correlation(loss_file, y, "utt_token")
    utt_flow_correlation = calc_correlation(loss_file, y, "utt_flow")
    combined_correlation = calc_correlation(loss_file, y, "combined")

    print(f"Sample count: {utt_flow_correlation[0]}") 
    print(f"Labels count: {utt_flow_correlation[1]}") 
    print(f"Correlation value for utterance-level literary tokens is: {utt_token_correlation[2]}") 
    print(f"Correlation value for utterance-level acoustic tokens is: {utt_flow_correlation[2]}")
    print(f"Weighted and combined correlation value is: {combined_correlation[2]}")





